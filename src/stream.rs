use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use log::{error, info, warn};

use crate::config::{Camera, Transport};

/// A decoded RGB24 frame. `stride` is bytes per row (>= width*3; FFmpeg
/// may align rows), so the UI can upload without repacking.
#[derive(Clone)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pub stride: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamStatus {
    Connecting,
    Live,
    Reconnecting(String),
}

/// Single-slot handoff: the decoder publishes the newest frame, the UI takes
/// it. Older frames are dropped so the display never falls behind realtime.
pub struct FrameSlot {
    frame: Mutex<Option<VideoFrame>>,
}

impl FrameSlot {
    pub fn new() -> Self {
        Self {
            frame: Mutex::new(None),
        }
    }

    pub fn publish(&self, frame: VideoFrame) {
        *self.frame.lock().unwrap() = Some(frame);
    }

    pub fn take(&self) -> Option<VideoFrame> {
        self.frame.lock().unwrap().take()
    }
}

pub struct CameraHandle {
    pub camera: Camera,
    pub frames: Arc<FrameSlot>,
    pub status: Arc<Mutex<StreamStatus>>,
    pub join: JoinHandle<()>,
}

pub fn spawn_camera(
    camera: Camera,
    transport: Transport,
    shutdown: Arc<AtomicBool>,
) -> CameraHandle {
    let frames = Arc::new(FrameSlot::new());
    let status = Arc::new(Mutex::new(StreamStatus::Connecting));
    let thread_frames = Arc::clone(&frames);
    let thread_status = Arc::clone(&status);
    let name = camera.name.clone();
    let thread_camera = camera.clone();

    let join = std::thread::Builder::new()
        .name(format!("stream-{name}"))
        .spawn(move || {
            camera_loop(thread_camera, transport, shutdown, thread_frames, thread_status)
        })
        .expect("failed to spawn stream thread");

    CameraHandle {
        camera,
        frames,
        status,
        join,
    }
}

fn camera_loop(
    camera: Camera,
    transport: Transport,
    shutdown: Arc<AtomicBool>,
    frames: Arc<FrameSlot>,
    status: Arc<Mutex<StreamStatus>>,
) {
    let mut backoff = Duration::from_secs(1);
    let max_backoff = Duration::from_secs(15);
    let mut first_attempt = true;

    while !shutdown.load(Ordering::Relaxed) {
        if first_attempt {
            *status.lock().unwrap() = StreamStatus::Connecting;
            first_attempt = false;
        } else {
            *status.lock().unwrap() = StreamStatus::Reconnecting(format!("retry in {backoff:?}"));
        }

        match run_session(&camera, transport, &shutdown, &frames, &status) {
            Ok(()) if shutdown.load(Ordering::Relaxed) => break,
            Ok(()) => {
                warn!("camera {}: stream ended, reconnecting", camera.name);
                backoff = Duration::from_secs(1);
            }
            Err(err) => {
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
                error!("camera {}: {err:#}", camera.name);
                *status.lock().unwrap() = StreamStatus::Reconnecting(format!("{err:#}"));
            }
        }

        let deadline = Instant::now() + backoff;
        while Instant::now() < deadline && !shutdown.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));
        }
        backoff = (backoff * 2).min(max_backoff);
    }

    info!("camera {}: stream thread stopped", camera.name);
}

/// Decode-side CPU limits.
#[derive(Debug, Clone, Copy)]
pub struct DecodeLimits {
    /// Publish at most this many frames per second (conversion + upload is
    /// skipped for extra frames; the decoder still drains them).
    pub max_fps: u32,
    /// Downscale so width never exceeds this (0 = keep native size).
    pub max_width: u32,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_fps: 15,
            max_width: 1280,
        }
    }
}

fn run_session(
    camera: &Camera,
    transport: Transport,
    shutdown: &AtomicBool,
    frames: &FrameSlot,
    status: &Mutex<StreamStatus>,
) -> Result<()> {
    // EZVIZ/Hikvision firmware disagrees about whether the RTSP password is
    // the MD5-scrambled verification code or the plain app password. If the
    // configured mode gets a 401, retry once with the other mode.
    let mut scramble = camera.scramble_enabled();
    let mut ictx = match open_rtsp(camera, transport, scramble) {
        Ok(ctx) => ctx,
        Err(err)
            if is_auth_error(&err)
                && camera.password.as_deref().is_some_and(|p| !p.is_empty()) =>
        {
            let failed_mode = scramble;
            scramble = !scramble;
            info!(
                "camera {}: auth failed with scramble_password={failed_mode}, retrying with scramble_password={scramble}",
                camera.name
            );
            open_rtsp(camera, transport, scramble).map_err(|retry_err| {
                anyhow::anyhow!(
                    "cannot open {} (scramble_password={scramble}): {retry_err:#}",
                    redact_url(&camera.rtsp_url_with_scramble(transport, scramble))
                )
            })?
        }
        Err(err) => return Err(err),
    };
    if scramble != camera.scramble_enabled() {
        info!(
            "camera {}: using scramble_password={scramble} (differs from config)",
            camera.name
        );
    }

    decode_context(
        &mut ictx,
        &camera.name,
        frames,
        status,
        shutdown,
        camera.decode_limits(),
    )
}

fn decode_context(
    ictx: &mut ffmpeg_next::format::context::Input,
    label: &str,
    frames: &FrameSlot,
    status: &Mutex<StreamStatus>,
    shutdown: &AtomicBool,
    limits: DecodeLimits,
) -> Result<()> {
    let video_stream = ictx
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .context("no video stream in URL")?;
    let stream_index = video_stream.index();
    let parameters = video_stream.parameters();

    let mut decoder = ffmpeg_next::codec::context::Context::from_parameters(parameters)?
        .decoder()
        .video()
        .context("unsupported video codec")?;

    // One thread per decoder: N cameras x auto-threads thrashes the CPU.
    decoder.set_threading(ffmpeg_next::codec::threading::Config {
        kind: ffmpeg_next::codec::threading::Type::Frame,
        count: 1,
    });

    let codec_name = ffmpeg_next::codec::decoder::find(decoder.id())
        .map(|c| c.name().to_string())
        .unwrap_or_else(|| "unknown".into());
    info!(
        "camera {}: {} {}x{} stream opened (max_fps={}, max_width={})",
        label,
        codec_name,
        decoder.width(),
        decoder.height(),
        limits.max_fps,
        limits.max_width,
    );

    let min_publish = Duration::from_micros(1_000_000 / limits.max_fps.max(1) as u64);
    let mut last_publish = Instant::now() - min_publish;

    let mut scaler: Option<ffmpeg_next::software::scaling::Context> = None;
    let mut rgb: Option<ffmpeg_next::frame::Video> = None;
    let mut decoded = ffmpeg_next::frame::Video::empty();
    let mut live_reported = false;

    for (stream, packet) in ictx.packets() {
        if shutdown.load(Ordering::Relaxed) {
            return Ok(());
        }
        if stream.index() != stream_index {
            continue;
        }

        decoder.send_packet(&packet)?;
        while decoder.receive_frame(&mut decoded).is_ok() {
            if decoded.width() == 0 || decoded.height() == 0 {
                continue;
            }

            if !live_reported {
                *status.lock().unwrap() = StreamStatus::Live;
                live_reported = true;
            }

            // Drop-to-cap: still drain the decoder, but skip scale/upload.
            if last_publish.elapsed() < min_publish {
                continue;
            }

            let (src_w, src_h) = (decoded.width(), decoded.height());
            let (out_w, out_h) = if limits.max_width > 0 && src_w > limits.max_width {
                let h = (u64::from(src_h) * u64::from(limits.max_width) / u64::from(src_w)) as u32;
                (limits.max_width, h.max(1))
            } else {
                (src_w, src_h)
            };

            if scaler
                .as_ref()
                .map(|s| {
                    s.input().format != decoded.format()
                        || s.input().width != src_w
                        || s.input().height != src_h
                        || s.output().width != out_w
                        || s.output().height != out_h
                })
                .unwrap_or(true)
            {
                scaler = Some(
                    ffmpeg_next::software::scaling::Context::get(
                        decoded.format(),
                        src_w,
                        src_h,
                        ffmpeg_next::format::Pixel::RGB24,
                        out_w,
                        out_h,
                        ffmpeg_next::software::scaling::Flags::BILINEAR,
                    )
                    .context("failed to create scaler")?,
                );
            }

            let needs_new_rgb = rgb
                .as_ref()
                .map(|f| f.width() != out_w || f.height() != out_h)
                .unwrap_or(true);
            if needs_new_rgb {
                rgb = Some(ffmpeg_next::frame::Video::new(
                    ffmpeg_next::format::Pixel::RGB24,
                    out_w,
                    out_h,
                ));
            }
            let rgb_frame = rgb.as_mut().expect("rgb frame just allocated");
            scaler.as_mut().unwrap().run(&decoded, rgb_frame)?;

            // Keep FFmpeg's row padding and let SDL upload with pitch —
            // avoids a full-frame repack copy every frame.
            let stride = rgb_frame.stride(0);
            let need = stride * out_h as usize;
            let mut out = Vec::with_capacity(need);
            out.extend_from_slice(&rgb_frame.data(0)[..need]);

            frames.publish(VideoFrame {
                width: out_w,
                height: out_h,
                rgb: out,
                stride,
            });
            last_publish = Instant::now();
        }
    }

    Ok(())
}

fn open_rtsp(
    camera: &Camera,
    transport: Transport,
    scramble: bool,
) -> Result<ffmpeg_next::format::context::Input> {
    let url = camera.rtsp_url_with_scramble(transport, scramble);
    info!(
        "camera {}: connecting to {} (scramble_password={scramble})",
        camera.name,
        redact_url(&url)
    );

    let mut opts = ffmpeg_next::Dictionary::new();
    opts.set("rtsp_transport", transport.as_str());
    // Microseconds: fail connect/read attempts promptly so the shutdown flag
    // and reconnect backoff stay responsive.
    opts.set("timeout", "5000000");
    opts.set("rw_timeout", "5000000");
    opts.set("max_delay", "500000");

    ffmpeg_next::format::input_with_dictionary(&url, opts)
        .with_context(|| format!("cannot open {}", redact_url(&url)))
}

/// True when FFmpeg failed because the camera rejected our credentials.
fn is_auth_error(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}").to_ascii_lowercase();
    msg.contains("401")
        || msg.contains("unauthorized")
        || msg.contains("authorization failed")
}

fn redact_url(url: &str) -> String {
    let Some(rest) = url.strip_prefix("rtsp://") else {
        return url.to_string();
    };
    match rest.split_once('@') {
        Some((userinfo, tail)) => {
            let user = userinfo.split_once(':').map(|(u, _)| u).unwrap_or(userinfo);
            format!("rtsp://{user}:***@{tail}")
        }
        None => url.to_string(),
    }
}

pub fn spawn_all(
    cameras: Vec<Camera>,
    transport: Transport,
    shutdown: Arc<AtomicBool>,
) -> Vec<CameraHandle> {
    cameras
        .into_iter()
        .map(|c| spawn_camera(c, transport, Arc::clone(&shutdown)))
        .collect()
}

pub fn shutdown_all(handles: Vec<CameraHandle>) {
    for handle in handles {
        let name = handle.camera.name.clone();
        match handle.join.join() {
            Ok(()) => log::debug!("camera {name}: joined"),
            Err(_) => error!("camera {name}: stream thread panicked"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_keeps_user_and_hides_password() {
        assert_eq!(
            redact_url("rtsp://admin:secret@192.168.1.64:554/Streaming/Channels/101"),
            "rtsp://admin:***@192.168.1.64:554/Streaming/Channels/101"
        );
    }

    #[test]
    fn redact_url_without_credentials_unchanged() {
        assert_eq!(
            redact_url("rtsp://192.168.1.64:554/Streaming/Channels/101"),
            "rtsp://192.168.1.64:554/Streaming/Channels/101"
        );
    }

    #[test]
    fn auth_error_detection() {
        let err: anyhow::Error = anyhow::anyhow!(
            "cannot open rtsp://admin:***@192.168.0.236:554/x: Server returned 401 Unauthorized (authorization failed)"
        );
        assert!(is_auth_error(&err));
        let err = anyhow::anyhow!("Connection refused");
        assert!(!is_auth_error(&err));
    }
}
