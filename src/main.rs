mod auth;
mod config;
mod motion;
mod ptz;
mod soap;
mod stream;
mod ui;
mod window_state;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::Parser;
use log::info;

use config::{Camera, Config, Transport, SAMPLE_CONFIG};

#[derive(Parser, Debug)]
#[command(
    name = "ezviz-view",
    about = "Multi-camera RTSP viewer for EZVIZ / Hikvision / JOOAN cameras on Linux, with ONVIF PTZ",
    version
)]
struct Args {
    /// Path to the TOML config file.
    #[arg(short, long, default_value = "ezviz-view.toml")]
    config: PathBuf,

    /// View a camera from a full RTSP URL (repeatable). Useful for quick checks.
    #[arg(long = "url", value_name = "RTSP_URL")]
    urls: Vec<String>,

    /// RTSP transport: tcp (recommended) or udp (lower latency).
    /// Overrides `transport` from the config file.
    #[arg(long)]
    transport: Option<String>,

    /// Initial window width in pixels (defaults to the last used size).
    #[arg(long)]
    width: Option<u32>,

    /// Initial window height in pixels (defaults to the last used size).
    #[arg(long)]
    height: Option<u32>,

    /// Print a sample config file to stdout and exit.
    #[arg(long)]
    print_sample_config: bool,

    /// Enable motion alerts on every camera (same as motion_alert = true
    /// in the config; toggle at runtime with the m key).
    #[arg(long)]
    motion: bool,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    if args.print_sample_config {
        print!("{SAMPLE_CONFIG}");
        return Ok(());
    }

    let mut cameras: Vec<Camera> = Vec::new();
    let mut config_transport: Option<Transport> = None;
    if args.config.exists() {
        let cfg = Config::load(&args.config)?;
        info!(
            "loaded {} camera(s) from {}",
            cfg.cameras.len(),
            args.config.display()
        );
        config_transport = Some(cfg.transport);
        cameras.extend(cfg.cameras);
    } else if args.urls.is_empty() {
        std::fs::write(&args.config, SAMPLE_CONFIG)
            .with_context(|| format!("failed to write {}", args.config.display()))?;
        eprintln!(
            "No config found. Wrote a sample config to {} — edit it with your\ncamera IPs and verification codes, then re-run this app.\n",
            args.config.display()
        );
        eprintln!("Alternatively pass --url rtsp://user:pass@ip:554/Streaming/Channels/101");
        std::process::exit(1);
    }

    for url in &args.urls {
        cameras.push(Camera::from_url(url)?);
    }
    if cameras.is_empty() {
        bail!("no cameras configured");
    }

    if args.motion {
        for cam in &mut cameras {
            cam.motion_alert = true;
        }
    }

    let transport = match &args.transport {
        Some(t) => match t.as_str() {
            "tcp" => Transport::Tcp,
            "udp" => Transport::Udp,
            other => bail!("invalid --transport {other:?}: expected tcp or udp"),
        },
        None => config_transport.unwrap_or_default(),
    };

    ffmpeg_next::init().context("ffmpeg initialization failed")?;
    ffmpeg_next::util::log::set_level(ffmpeg_next::util::log::Level::Warning);

    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&shutdown);
        ctrlc::set_handler(move || {
            flag.store(true, Ordering::SeqCst);
        })
        .context("failed to install Ctrl-C handler")?;
    }

    info!("starting viewer with {} camera(s)", cameras.len());
    let ptz_system = ptz::PtzSystem::spawn(&cameras);
    if let Some(system) = &ptz_system {
        let names: Vec<&str> = system.targets.iter().map(|(_, n)| n.as_str()).collect();
        info!(
            "PTZ enabled for {} camera(s): {} (arrows pan/tilt, +/- zoom, p selects target)",
            names.len(),
            names.join(", ")
        );
    }

    let handles = stream::spawn_all(cameras, transport, Arc::clone(&shutdown));

    // Window size: CLI flag > last used (persisted on exit) > 1280x720.
    let saved = window_state::WindowState::load();
    let width = args.width.unwrap_or(saved.width).clamp(320, 7680);
    let height = args.height.unwrap_or(saved.height).clamp(200, 4320);
    info!("window {width}x{height}");

    let ui_result = ui::run(&handles, &shutdown, width, height, ptz_system.as_ref());

    shutdown.store(true, Ordering::SeqCst);
    drop(ptz_system); // joins the PTZ worker
    stream::shutdown_all(handles);
    info!("goodbye");
    ui_result
}
