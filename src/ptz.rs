use std::sync::mpsc::{channel, Sender};
use std::sync::Mutex;
use std::thread::JoinHandle;

use anyhow::{anyhow, bail, Result};
use log::{debug, info};

use crate::config::{Camera, PtzConfig};
use crate::soap::{escape_xml, soap_envelope, HTTP_TIMEOUT};

/// Tokens commonly returned (or accepted) by budget cameras when the
/// capabilities/discovery layer is stripped down.
const FALLBACK_TOKENS: &[&str] = &[
    "Profile_0",
    "MainProfile",
    "000",
    "Profile_1",
    "1",
    "profile1",
    "main",
];

/// PTZ service paths to try, in order.
const PTZ_PATHS: &[&str] = &["/onvif/ptz_service", "/onvif/PTZ", "/onvif/ptz"];

/// Media service paths used to discover the profile token via GetProfiles.
const MEDIA_PATHS: &[&str] = &[
    "/onvif/media_service",
    "/onvif/Media",
    "/onvif/media",
];

pub enum PtzCommand {
    Move {
        camera: usize,
        pan: f64,
        tilt: f64,
        zoom: f64,
    },
    Stop {
        camera: usize,
    },
}

/// Owns a worker thread that serializes ONVIF SOAP calls so the UI never
/// blocks on network latency.
pub struct PtzSystem {
    tx: Option<Sender<PtzCommand>>,
    join: Option<JoinHandle<()>>,
    /// Indices into the camera list that have PTZ configured, with names.
    pub targets: Vec<(usize, String)>,
}

impl PtzSystem {
    /// Spawn the PTZ worker for every camera that has a `[cameras.ptz]` table.
    pub fn spawn(cameras: &[Camera]) -> Option<Self> {
        let mut clients: Vec<Option<OnvifPtz>> = Vec::with_capacity(cameras.len());
        let mut targets = Vec::new();

        for (idx, cam) in cameras.iter().enumerate() {
            if let Some(ptz) = &cam.ptz {
                clients.push(Some(OnvifPtz::new(cam, ptz)));
                targets.push((idx, cam.name.clone()));
            } else {
                clients.push(None);
            }
        }

        if targets.is_empty() {
            return None;
        }

        let (tx, rx) = channel::<PtzCommand>();
        let join = std::thread::Builder::new()
            .name("ptz-worker".into())
            .spawn(move || {
                for cmd in rx {
                    match cmd {
                        PtzCommand::Move {
                            camera,
                            pan,
                            tilt,
                            zoom,
                        } => {
                            if let Some(Some(client)) = clients.get(camera) {
                                if let Err(err) = client.continuous_move(pan, tilt, zoom) {
                                    debug!("camera {}: PTZ move failed: {err:#}", client.name);
                                }
                            }
                        }
                        PtzCommand::Stop { camera } => {
                            if let Some(Some(client)) = clients.get(camera) {
                                if let Err(err) = client.stop() {
                                    debug!("camera {}: PTZ stop failed: {err:#}", client.name);
                                }
                            }
                        }
                    }
                }
            })
            .expect("failed to spawn PTZ worker");

        Some(PtzSystem {
            tx: Some(tx),
            join: Some(join),
            targets,
        })
    }

    pub fn send(&self, cmd: PtzCommand) {
        if let Some(tx) = &self.tx {
            // Worker only dies if the UI is shutting down; ignore send errors.
            let _ = tx.send(cmd);
        }
    }
}

impl Drop for PtzSystem {
    fn drop(&mut self) {
        // Close the channel first so the worker's for-loop ends, then join.
        self.tx.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// Minimal ONVIF PTZ client: ContinuousMove / Stop over SOAP with
/// WS-UsernameToken (PasswordText). Tolerates the stripped-down ONVIF
/// implementations found on JOOAN / CloudEdge cameras.
pub struct OnvifPtz {
    pub name: String,
    ptz_urls: Vec<String>,
    media_urls: Vec<String>,
    username: Option<String>,
    password: Option<String>,
    configured_token: Option<String>,
    resolved_token: Mutex<Option<String>>,
    agent: ureq::Agent,
}

impl OnvifPtz {
    pub fn new(cam: &Camera, ptz: &PtzConfig) -> Self {
        let port = ptz.onvif_port.unwrap_or_else(|| cam.default_onvif_port());
        let ptz_urls = PTZ_PATHS
            .iter()
            .map(|p| format!("http://{}:{port}{p}", cam.host))
            .collect();
        let media_urls = MEDIA_PATHS
            .iter()
            .map(|p| format!("http://{}:{port}{p}", cam.host))
            .collect();

        let agent = ureq::AgentBuilder::new()
            .timeout(HTTP_TIMEOUT)
            .user_agent("ezviz-view/0.1")
            .build();

        Self {
            name: cam.name.clone(),
            ptz_urls,
            media_urls,
            username: cam.username.clone(),
            password: cam.password.clone(),
            configured_token: ptz.profile_token.clone(),
            resolved_token: Mutex::new(None),
            agent,
        }
    }

    pub fn continuous_move(&self, pan: f64, tilt: f64, zoom: f64) -> Result<()> {
        let token = self.ensure_token()?;
        let body = soap_envelope(
            &self.security_header(),
            &format!(
                r#"<tptz:ContinuousMove>
  <tptz:ProfileToken>{token}</tptz:ProfileToken>
  <tptz:Velocity>
    <tt:PanTilt x="{pan}" y="{tilt}"/>
    <tt:Zoom x="{zoom}"/>
  </tptz:Velocity>
</tptz:ContinuousMove>"#
            ),
        );
        self.post_any(&self.ptz_urls, &body, "ContinuousMove")
    }

    pub fn stop(&self) -> Result<()> {
        let token = self.ensure_token()?;
        let body = soap_envelope(
            &self.security_header(),
            &format!(
                r#"<tptz:Stop>
  <tptz:ProfileToken>{token}</tptz:ProfileToken>
  <tptz:PanTilt>true</tptz:PanTilt>
  <tptz:Zoom>true</tptz:Zoom>
</tptz:Stop>"#
            ),
        );
        self.post_any(&self.ptz_urls, &body, "Stop")
    }

    fn ensure_token(&self) -> Result<String> {
        if let Some(token) = &*self.resolved_token.lock().unwrap() {
            return Ok(token.clone());
        }

        if let Some(token) = &self.configured_token {
            *self.resolved_token.lock().unwrap() = Some(token.clone());
            info!("camera {}: PTZ profile token {token:?} (from config)", self.name);
            return Ok(token.clone());
        }

        // Try GetProfiles on the media service (works on full ONVIF stacks).
        for url in &self.media_urls {
            let body = soap_envelope(
                &self.security_header(),
                "<trt:GetProfiles/>",
            );
            match self.post(url, &body) {
                Ok(resp) => {
                    if let Some(token) = parse_profile_token(&resp) {
                        info!("camera {}: PTZ profile token {token:?} (GetProfiles)", self.name);
                        *self.resolved_token.lock().unwrap() = Some(token.clone());
                        return Ok(token);
                    }
                }
                Err(err) => debug!("camera {}: GetProfiles {url} failed: {err:#}", self.name),
            }
        }

        // Stripped ONVIF (JOOAN et al.): probe tokens with a harmless Stop.
        for token in FALLBACK_TOKENS {
            let body = soap_envelope(
                &self.security_header(),
                &format!(
                    r#"<tptz:Stop>
  <tptz:ProfileToken>{token}</tptz:ProfileToken>
  <tptz:PanTilt>true</tptz:PanTilt>
  <tptz:Zoom>true</tptz:Zoom>
</tptz:Stop>"#
                ),
            );
            if self.post_any(&self.ptz_urls, &body, "Stop probe").is_ok() {
                info!("camera {}: PTZ profile token {token:?} (probed)", self.name);
                *self.resolved_token.lock().unwrap() = Some((*token).to_string());
                return Ok((*token).to_string());
            }
        }

        bail!(
            "camera {}: could not determine PTZ profile token; set profile_token in [cameras.ptz] (try \"Profile_0\")",
            self.name
        )
    }

    fn security_header(&self) -> String {
        let user = escape_xml(self.username.as_deref().unwrap_or("admin"));
        let pass = escape_xml(self.password.as_deref().unwrap_or(""));
        format!(
            r#"<wsse:Security s:mustUnderstand="1">
  <wsse:UsernameToken>
    <wsse:Username>{user}</wsse:Username>
    <wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordText">{pass}</wsse:Password>
  </wsse:UsernameToken>
</wsse:Security>"#
        )
    }

    /// POST the envelope to each candidate URL until one succeeds.
    fn post_any(&self, urls: &[String], body: &str, what: &str) -> Result<()> {
        let mut last_err = None;
        for url in urls {
            match self.post(url, body) {
                Ok(_) => return Ok(()),
                Err(err) => {
                    debug!("camera {}: {what} {url} failed: {err:#}", self.name);
                    last_err = Some(err);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("camera {}: no {what} URLs", self.name)))
    }

    fn post(&self, url: &str, body: &str) -> Result<String> {
        let resp = self
            .agent
            .post(url)
            .set(
                "Content-Type",
                "application/soap+xml; charset=utf-8",
            )
            .send_string(body);

        match resp {
            Ok(resp) => {
                let status = resp.status();
                let text = resp.into_string().unwrap_or_default();
                if text.contains("Fault") || text.contains("fault") {
                    bail!("SOAP fault ({status}): {}", truncate(&text, 200));
                }
                if !(200..300).contains(&status) {
                    bail!("HTTP {status}: {}", truncate(&text, 200));
                }
                Ok(text)
            }
            Err(ureq::Error::Status(status, resp)) => {
                let text = resp.into_string().unwrap_or_default();
                if text.contains("Fault") || text.contains("fault") {
                    bail!("SOAP fault ({status}): {}", truncate(&text, 200));
                }
                bail!("HTTP {status}: {}", truncate(&text, 200))
            }
            Err(err) => Err(err.into()),
        }
    }
}

/// Extract the first `token="..."` from a GetProfiles response.
fn parse_profile_token(xml: &str) -> Option<String> {
    // Profiles elements carry a token attribute, e.g.
    // <tr2:Profiles token="Profile_0" fixed="true"> or <tt:Profiles token="000">
    let mut rest = xml;
    while let Some(idx) = rest.find("token=\"") {
        let after = &rest[idx + "token=\"".len()..];
        if let Some(end) = after.find('"') {
            let token = &after[..end];
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
        rest = after;
    }
    None
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        let mut end = max;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn envelope_contains_move_body() {
        let env = soap_envelope("<wsse:Security/>", "<tptz:ContinuousMove>...</tptz:ContinuousMove>");
        assert!(env.contains("ContinuousMove"));
        assert!(env.contains("ver20/ptz/wsdl"));
        assert!(env.contains("wsse:Security"));
    }

    #[test]
    fn parse_token_from_get_profiles_response() {
        let xml = r#"<?xml version="1.0"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
<s:Body><tr2:GetProfilesResponse xmlns:tr2="http://www.onvif.org/ver20/media/wsdl">
<tr2:Profiles token="Profile_0" fixed="true">
<tt:Name>MainStream</tt:Name>
</tr2:Profiles>
</tr2:GetProfilesResponse></s:Body></s:Envelope>"#;
        assert_eq!(parse_profile_token(xml).as_deref(), Some("Profile_0"));
    }

    #[test]
    fn parse_token_handles_multiple_profiles() {
        let xml = r#"<tr2:Profiles token="AAA"/><tr2:Profiles token="BBB"/>"#;
        assert_eq!(parse_profile_token(xml).as_deref(), Some("AAA"));
    }

    #[test]
    fn parse_token_absent() {
        assert!(parse_profile_token("<html>404</html>").is_none());
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("héllo", 2), "h");
    }

    /// End-to-end: OnvifPtz against an in-process mock ONVIF camera.
    /// Validates token discovery (GetProfiles) + ContinuousMove SOAP flow.
    #[test]
    fn onvif_continuous_move_end_to_end() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let get_profiles = r#"<?xml version="1.0"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body>
<tr2:GetProfilesResponse xmlns:tr2="http://www.onvif.org/ver20/media/wsdl">
<tr2:Profiles token="Profile_0" fixed="true">
<tt:Name xmlns:tt="http://www.onvif.org/ver10/schema">MainStream</tt:Name>
</tr2:Profiles>
</tr2:GetProfilesResponse></s:Body></s:Envelope>"#;
        let ok = r#"<?xml version="1.0"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body>
<tptz:ContinuousMoveResponse xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"/>
</s:Body></s:Envelope>"#;

        let server = std::thread::spawn(move || {
            let mut seen = Vec::new();
            // GetProfiles + ContinuousMove
            for _ in 0..2 {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut req = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    match stream.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            req.extend_from_slice(&buf[..n]);
                            if let Some(header_end) = req.windows(4).position(|w| w == b"\r\n\r\n") {
                                let headers = String::from_utf8_lossy(&req[..header_end]).to_string();
                                let content_length: usize = headers
                                    .lines()
                                    .find_map(|l| {
                                        l.to_ascii_lowercase()
                                            .strip_prefix("content-length:")
                                            .map(|v| v.trim().parse().unwrap_or(0))
                                    })
                                    .unwrap_or(0);
                                if req.len() >= header_end + 4 + content_length {
                                    break;
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
                let body = String::from_utf8_lossy(&req).to_string();
                seen.push(body.clone());
                let resp_body = if body.contains("GetProfiles") {
                    get_profiles
                } else {
                    ok
                };
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/soap+xml; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{resp_body}",
                    resp_body.len()
                );
                let _ = stream.write_all(resp.as_bytes());
                let _ = stream.flush();
            }
            seen
        });

        let cam = Camera {
            name: "mock".into(),
            host: "127.0.0.1".into(),
            brand: crate::config::Brand::Jooan,
            port: 554,
            channel: 1,
            stream: 1,
            username: Some("admin".into()),
            password: Some("secret".into()),
            scramble_password: None,
            path: None,
            motion_alert: false,
            motion_threshold: None,
            max_fps: None,
            max_width: None,
            ptz: Some(PtzConfig {
                onvif_port: Some(port),
                profile_token: None,
                speed: 0.5,
            }),
        };
        let client = OnvifPtz::new(&cam, cam.ptz.as_ref().unwrap());
        client.continuous_move(0.5, -0.5, 0.0).expect("move");

        let seen = server.join().unwrap();
        assert!(
            seen.iter().any(|b| b.contains("GetProfiles")),
            "expected a GetProfiles call, got: {seen:?}"
        );
        let move_req = seen
            .iter()
            .find(|b| b.contains("ContinuousMove"))
            .expect("ContinuousMove request");
        assert!(move_req.contains("Profile_0"));
        assert!(move_req.contains(r#"PanTilt x="0.5" y="-0.5""#));
        assert!(move_req.contains("wsse:UsernameToken"));
        assert!(move_req.contains("admin"));
        assert!(move_req.contains("secret"));
    }

    /// When profile_token is configured, no GetProfiles probe is needed.
    #[test]
    fn onvif_uses_configured_token_without_getprofiles() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let server = std::thread::spawn(move || {
            let mut bodies = Vec::new();
            for _ in 0..1 {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut req = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    match stream.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            req.extend_from_slice(&buf[..n]);
                            if req.windows(4).any(|w| w == b"\r\n\r\n") {
                                // Enough for a small SOAP body in tests.
                                if req.len() > 400 {
                                    break;
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
                bodies.push(String::from_utf8_lossy(&req).to_string());
                let resp_body = r#"<?xml version="1.0"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body>
<tptz:StopResponse xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"/>
</s:Body></s:Envelope>"#;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/soap+xml\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{resp_body}",
                    resp_body.len()
                );
                let _ = stream.write_all(resp.as_bytes());
            }
            bodies
        });

        let cam = Camera {
            name: "mock2".into(),
            host: "127.0.0.1".into(),
            brand: crate::config::Brand::Jooan,
            port: 554,
            channel: 1,
            stream: 1,
            username: None,
            password: None,
            scramble_password: None,
            path: None,
            motion_alert: false,
            motion_threshold: None,
            max_fps: None,
            max_width: None,
            ptz: Some(PtzConfig {
                onvif_port: Some(port),
                profile_token: Some("MainProfile".into()),
                speed: 0.5,
            }),
        };
        let client = OnvifPtz::new(&cam, cam.ptz.as_ref().unwrap());
        client.stop().expect("stop");

        let bodies = server.join().unwrap();
        assert_eq!(bodies.len(), 1);
        assert!(!bodies[0].contains("GetProfiles"));
        assert!(bodies[0].contains("MainProfile"));
        assert!(bodies[0].contains("tptz:Stop"));
    }
}
