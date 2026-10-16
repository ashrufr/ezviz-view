use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::auth::{encode_userinfo, scramble_password};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    #[default]
    Tcp,
    Udp,
}

impl Transport {
    pub fn as_str(self) -> &'static str {
        match self {
            Transport::Tcp => "tcp",
            Transport::Udp => "udp",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Brand {
    /// Hikvision-style URL (/Streaming/Channels/101), scrambled passwords.
    #[default]
    Ezviz,
    /// JOOAN / CloudEdge style URL (/live/ch00_0), plain RTSP auth.
    Jooan,
}

/// ONVIF PTZ settings for pan/tilt/zoom control.
#[derive(Debug, Clone, Deserialize)]
pub struct PtzConfig {
    /// ONVIF HTTP port. JOOAN models commonly use 8899; other brands 80/8080.
    pub onvif_port: Option<u16>,
    /// PTZ profile token (e.g. "Profile_0"). Auto-detected when omitted.
    pub profile_token: Option<String>,
    /// Pan/tilt/zoom velocity, 0.1 (slow) .. 1.0 (fast).
    #[serde(default = "default_ptz_speed")]
    pub speed: f64,
}

fn default_ptz_speed() -> f64 {
    0.5
}

#[derive(Debug, Clone, Deserialize)]
pub struct Camera {
    /// Display name shown in logs.
    pub name: String,
    /// Camera IP or hostname on the LAN.
    pub host: String,
    #[serde(default)]
    pub brand: Brand,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Camera channel/lens: 1 = primary (pan/tilt lens on dual-lens JOOAN).
    #[serde(default = "default_channel")]
    pub channel: u16,
    /// 1 = main stream, 2 = sub stream (lower resolution, lighter CPU).
    #[serde(default = "default_stream")]
    pub stream: u16,
    pub username: Option<String>,
    /// Device password / verification code / ONVIF password.
    pub password: Option<String>,
    /// Apply the Hikvision/EZVIZ RTSP password scramble. Defaults to true for
    /// brand "ezviz" and false for "jooan".
    pub scramble_password: Option<bool>,
    /// Full RTSP path override, e.g. "/live/ch00_0" or
    /// "/user=admin_password=x_channel=1_stream=0.sdp".
    pub path: Option<String>,
    /// Enable client-side motion alerts for this camera (frame differencing;
    /// toggle at runtime with the m key). Default false.
    #[serde(default)]
    pub motion_alert: bool,
    /// Motion sensitivity: mean per-pixel change needed to alert.
    /// Lower = more sensitive. Default 0.012 (quiet scene ~0.002,
    /// person in frame ~0.005..0.03). Raise it if you get false alarms.
    pub motion_threshold: Option<f64>,
    /// Publish at most this many frames/s to the UI (CPU saver). Default 15.
    pub max_fps: Option<u32>,
    /// Downscale decoded frames wider than this before display (CPU saver).
    /// 0 = keep native size. Default 1280.
    pub max_width: Option<u32>,
    /// Pan/tilt/zoom control via ONVIF. Presence of this table enables PTZ.
    pub ptz: Option<PtzConfig>,
}

fn default_port() -> u16 {
    554
}
fn default_channel() -> u16 {
    1
}
fn default_stream() -> u16 {
    1
}

impl Camera {
    pub fn scramble_enabled(&self) -> bool {
        self.scramble_password
            .unwrap_or(matches!(self.brand, Brand::Ezviz))
    }

    fn stream_path(&self) -> String {
        if let Some(path) = &self.path {
            return path.clone();
        }
        match self.brand {
            Brand::Ezviz => format!("/Streaming/Channels/{}", self.channel * 100 + self.stream),
            // JOOAN: channel 1 -> ch00, channel 2 -> ch01; stream 1 -> _0, stream 2 -> _1
            Brand::Jooan => format!(
                "/live/ch{:02}_{}",
                self.channel.saturating_sub(1),
                self.stream.saturating_sub(1)
            ),
        }
    }

    /// Default ONVIF port for PTZ when not configured explicitly.
    pub fn default_onvif_port(&self) -> u16 {
        match self.brand {
            Brand::Jooan => 8899,
            Brand::Ezviz => 80,
        }
    }

    /// Decode/display CPU limits for this camera.
    pub fn decode_limits(&self) -> crate::stream::DecodeLimits {
        crate::stream::DecodeLimits {
            max_fps: self.max_fps.unwrap_or(15).clamp(1, 60),
            max_width: self.max_width.unwrap_or(1280),
        }
    }

    /// Build the RTSP URL for this camera using the configured scramble mode.
    #[cfg(test)]
    pub fn rtsp_url(&self, transport: Transport) -> String {
        self.rtsp_url_with_scramble(transport, self.scramble_enabled())
    }

    /// Build the RTSP URL, forcing the password scramble on or off. Used to
    /// retry with the alternate auth scheme after a 401.
    pub fn rtsp_url_with_scramble(&self, _transport: Transport, scramble: bool) -> String {
        let path = self.stream_path();
        let mut url = String::from("rtsp://");

        if let Some(user) = &self.username {
            url.push_str(&encode_userinfo(user));
            if let Some(pass) = &self.password {
                url.push(':');
                let effective = if scramble {
                    scramble_password(pass)
                } else {
                    pass.clone()
                };
                url.push_str(&encode_userinfo(&effective));
            }
            url.push('@');
        }

        let _ = write!(url, "{}:{}{path}", self.host, self.port);
        url
    }

    pub fn from_url(url: &str) -> Result<Self> {
        let rest = url
            .strip_prefix("rtsp://")
            .context("URL must start with rtsp://")?;
        let (userinfo, tail) = match rest.find('@') {
            Some(idx) => (&rest[..idx], &rest[idx + 1..]),
            None => ("", rest),
        };
        let (hostport, path) = match tail.find('/') {
            Some(idx) => (&tail[..idx], &tail[idx..]),
            None => (tail, ""),
        };
        let (host, port) = match hostport.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), p.parse().context("invalid port in URL")?),
            None => (hostport.to_string(), default_port()),
        };
        let channel_stream = path
            .rsplit('/')
            .find(|s| !s.is_empty())
            .and_then(|s| s.parse::<u32>().ok());
        let (channel, stream) = match channel_stream {
            Some(id) => ((id / 100) as u16, (id % 100).max(1) as u16),
            None => (default_channel(), default_stream()),
        };
        let (username, password) = match userinfo.split_once(':') {
            Some((u, p)) => (Some(u.to_string()), Some(p.to_string())),
            None if !userinfo.is_empty() => (Some(userinfo.to_string()), None),
            None => (None, None),
        };

        Ok(Camera {
            name: host.clone(),
            host,
            brand: Brand::Ezviz,
            port,
            channel,
            stream,
            username,
            password,
            // URLs pasted from other tools usually already contain a final
            // password, so do not scramble them again.
            scramble_password: Some(false),
            path: if path.is_empty() {
                None
            } else {
                Some(path.to_string())
            },
            motion_alert: false,
            motion_threshold: None,
            max_fps: None,
            max_width: None,
            ptz: None,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub transport: Transport,
    #[serde(default)]
    pub cameras: Vec<Camera>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        let config: Config =
            toml::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.cameras.is_empty() {
            bail!("config contains no [[cameras]] entries");
        }
        for cam in &self.cameras {
            if cam.host.is_empty() {
                bail!("camera {:?} has an empty host", cam.name);
            }
            if !(1..=2).contains(&cam.stream) {
                bail!(
                    "camera {:?}: stream must be 1 (main) or 2 (sub), got {}",
                    cam.name,
                    cam.stream
                );
            }
            if let Some(ptz) = &cam.ptz {
                if !(0.1..=1.0).contains(&ptz.speed) {
                    bail!(
                        "camera {:?}: ptz.speed must be between 0.1 and 1.0, got {}",
                        cam.name, ptz.speed
                    );
                }
            }
            if let Some(thr) = cam.motion_threshold {
                if !(0.001..=1.0).contains(&thr) {
                    bail!(
                        "camera {:?}: motion_threshold must be between 0.001 and 1.0, got {}",
                        cam.name,
                        thr
                    );
                }
            }
            if let Some(fps) = cam.max_fps {
                if fps == 0 || fps > 60 {
                    bail!(
                        "camera {:?}: max_fps must be 1..=60, got {}",
                        cam.name, fps
                    );
                }
            }
        }
        Ok(())
    }
}

pub const SAMPLE_CONFIG: &str = r#"# ezviz-view configuration
# transport applies to all cameras unless overridden per camera via CLI.
transport = "tcp"

# EZVIZ camera on the LAN. Use stream = 2 (sub stream) if CPU usage is high.
[[cameras]]
name = "Front door"
host = "192.168.1.64"
brand = "ezviz"
username = "admin"
password = "VERIFYCODE"      # verification code from the camera label
# scramble_password defaults to true for ezviz

[[cameras]]
name = "Backyard"
host = "192.168.1.65"
brand = "ezviz"
username = "admin"
password = "VERIFYCODE"
channel = 1
stream = 2
 motion_alert = true          # flash + audible beep when something moves
 motion_threshold = 0.012     # lower = more sensitive (0.001 .. 1.0)
# max_fps = 15               # CPU saver: publish at most 15 fps (default)
# max_width = 1280           # CPU saver: downscale wide streams (0 = native)

# JOOAN / CloudEdge pan-tilt camera with ONVIF PTZ control.
# RTSP: rtsp://user:pass@ip:554/live/ch00_0   (ch01_0 = second lens)
# PTZ:  arrows pan/tilt, +/- zooms (hold key to move). Press p to switch
#       which camera the arrow keys control.
[[cameras]]
name = "Driveway"
host = "192.168.1.70"
brand = "jooan"
username = "admin"
password = "admin123"        # ONVIF user set in the Cam720/Jooan app
channel = 1                  # 1 = pan/tilt lens, 2 = fixed lens
stream = 1

[cameras.ptz]
onvif_port = 8899            # common JOOAN ONVIF port; also try 80/8080/5000
# profile_token = "Profile_0"  # auto-detected when omitted
speed = 0.5
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> Camera {
        Camera {
            name: "test".into(),
            host: "192.168.1.64".into(),
            brand: Brand::Ezviz,
            port: 554,
            channel: 1,
            stream: 1,
            username: Some("admin".into()),
            password: Some("abc123".into()),
            scramble_password: Some(true),
            path: None,
            motion_alert: false,
            motion_threshold: None,
            max_fps: None,
            max_width: None,
            ptz: None,
        }
    }

    #[test]
    fn url_scrambles_password() {
        let url = cam().rtsp_url(Transport::Tcp);
        assert_eq!(
            url,
            "rtsp://admin:a99ad0a45a70ab9a6fa62ab6ae7a5ea0da74a13a83a64ad8@192.168.1.64:554/Streaming/Channels/101"
        );
    }

    #[test]
    fn url_plain_password_and_substream() {
        let mut c = cam();
        c.scramble_password = Some(false);
        c.stream = 2;
        c.password = Some("p@ss".into());
        let url = c.rtsp_url(Transport::Tcp);
        assert!(url.ends_with("/Streaming/Channels/102"));
        assert!(url.contains(":p%40ss@"));
    }

    #[test]
    fn jooan_url_and_plain_auth_by_default() {
        let mut c = cam();
        c.brand = Brand::Jooan;
        c.scramble_password = None;
        c.host = "192.168.1.70".into();
        c.password = Some("admin123".into());
        assert!(!c.scramble_enabled());
        assert_eq!(
            c.rtsp_url(Transport::Tcp),
            "rtsp://admin:admin123@192.168.1.70:554/live/ch00_0"
        );

        c.channel = 2;
        c.stream = 2;
        assert!(c
            .rtsp_url(Transport::Tcp)
            .ends_with("/live/ch01_1"));
    }

    #[test]
    fn path_override_wins() {
        let mut c = cam();
        c.path = Some("/custom/stream".into());
        assert!(c.rtsp_url(Transport::Tcp).ends_with("@192.168.1.64:554/custom/stream"));
    }

    #[test]
    fn parse_inline_config_with_jooan_ptz() {
        let cfg: Config = toml::from_str(SAMPLE_CONFIG).unwrap();
        assert_eq!(cfg.cameras.len(), 3);
        let backyard = &cfg.cameras[1];
        assert!(backyard.motion_alert);
        assert_eq!(backyard.motion_threshold, Some(0.012));
        let jooan = &cfg.cameras[2];
        assert_eq!(jooan.brand, Brand::Jooan);
        let ptz = jooan.ptz.as_ref().expect("ptz config");
        assert_eq!(ptz.onvif_port, Some(8899));
        assert_eq!(ptz.speed, 0.5);
        assert!(ptz.profile_token.is_none());
    }

    #[test]
    fn parse_quick_url() {
        let c =
            Camera::from_url("rtsp://admin:secret@10.0.0.9:554/Streaming/Channels/102").unwrap();
        assert_eq!(c.host, "10.0.0.9");
        assert_eq!(c.username.as_deref(), Some("admin"));
        assert_eq!(c.password.as_deref(), Some("secret"));
        assert_eq!(c.channel, 1);
        assert_eq!(c.stream, 2);
        assert!(!c.scramble_enabled());
    }
}
