use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use log::{info, warn};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::Texture;
use sdl2::video::FullscreenType;

use crate::motion::{
    load_thresholds, load_zones, next_sensitivity, save_thresholds, save_zones,
    sensitivity_label, MotionAlertState, MotionZone, DEFAULT_MOTION_THRESHOLD,
    OUTLINE_SIMPLIFY_EPSILON,
};
use crate::ptz::{PtzCommand, PtzSystem};
use crate::stream::{CameraHandle, StreamStatus};

const LAYOUTS: [usize; 3] = [1, 4, 9];
const BORDER: i32 = 2;

/// Minimum spacing between freehand samples, in output pixels, so a slow
/// drag does not record hundreds of nearly identical points.
const STROKE_MIN_SAMPLE_PX: i32 = 4;

/// Below this total movement a right-drag is treated as a plain click
/// (which clears the zone).
const STROKE_CLICK_PX: i32 = 10;

/// In-progress right-drag freehand stroke used to draw a motion zone.
struct ZoneStroke {
    cam_idx: usize,
    cell: Rect,
    /// Outline sampled so far, in video-normalized coordinates.
    points: Vec<[f32; 2]>,
    /// Last sampled output-pixel position.
    last_px: (i32, i32),
    /// Output-pixel position where the stroke began.
    start_px: (i32, i32),
    /// Largest distance moved from the start (output pixels).
    max_move: i32,
}

struct CamTexture<'a> {
    texture: Texture<'a>,
    width: u32,
    height: u32,
}

/// Keyboard state for hold-to-move PTZ control.
struct PtzUiState {
    /// Index into `PtzSystem::targets` of the camera the arrows control.
    selected: usize,
    up: bool,
    down: bool,
    left: bool,
    right: bool,
    zoom_in: bool,
    zoom_out: bool,
}

impl PtzUiState {
    fn new() -> Self {
        Self {
            selected: 0,
            up: false,
            down: false,
            left: false,
            right: false,
            zoom_in: false,
            zoom_out: false,
        }
    }

    fn cycle(&mut self, target_count: usize) {
        if target_count > 0 {
            self.selected = (self.selected + 1) % target_count;
        }
    }

    fn key_down(&mut self, key: Keycode) -> bool {
        match key {
            Keycode::Up => self.up = true,
            Keycode::Down => self.down = true,
            Keycode::Left => self.left = true,
            Keycode::Right => self.right = true,
            Keycode::Plus | Keycode::KpPlus | Keycode::Equals => self.zoom_in = true,
            Keycode::Minus | Keycode::KpMinus => self.zoom_out = true,
            _ => return false,
        }
        true
    }

    fn key_up(&mut self, key: Keycode) -> bool {
        match key {
            Keycode::Up => self.up = false,
            Keycode::Down => self.down = false,
            Keycode::Left => self.left = false,
            Keycode::Right => self.right = false,
            Keycode::Plus | Keycode::KpPlus | Keycode::Equals => self.zoom_in = false,
            Keycode::Minus | Keycode::KpMinus => self.zoom_out = false,
            _ => return false,
        }
        true
    }

    /// Send Move/Stop to the worker based on the currently held keys.
    fn sync(&self, ptz: &PtzSystem, cameras: &[CameraHandle]) {
        let Some(&(cam_idx, _)) = ptz.targets.get(self.selected) else {
            return;
        };
        let speed = cameras
            .get(cam_idx)
            .and_then(|h| h.camera.ptz.as_ref())
            .map(|p| p.speed)
            .unwrap_or(0.5);

        let pan = (self.right as i32 - self.left as i32) as f64 * speed;
        let tilt = (self.up as i32 - self.down as i32) as f64 * speed;
        let zoom = (self.zoom_in as i32 - self.zoom_out as i32) as f64 * speed;

        if pan == 0.0 && tilt == 0.0 && zoom == 0.0 {
            ptz.send(PtzCommand::Stop { camera: cam_idx });
        } else {
            ptz.send(PtzCommand::Move {
                camera: cam_idx,
                pan,
                tilt,
                zoom,
            });
        }
    }
}

pub fn run(
    cameras: &[CameraHandle],
    shutdown: &Arc<AtomicBool>,
    width: u32,
    height: u32,
    ptz: Option<&PtzSystem>,
) -> Result<()> {
    let sdl = sdl2::init().map_err(|e| anyhow::anyhow!("SDL init failed (is DISPLAY set?): {e}"))?;
    let video = sdl
        .video()
        .map_err(|e| anyhow::anyhow!("SDL video subsystem unavailable: {e}"))?;
    let window = video
        .window("EZVIZ View", width, height)
        .position_centered()
        .resizable()
        .build()
        .context("failed to create window")?;
    let mut canvas = window
        .into_canvas()
        .accelerated()
        .present_vsync()
        .build()
        .context("failed to create renderer")?;
    let texture_creator = canvas.texture_creator();
    let mut textures: HashMap<usize, CamTexture<'_>> = HashMap::new();
    let mut event_pump = sdl
        .event_pump()
        .map_err(|e| anyhow::anyhow!("failed to create SDL event pump: {e}"))?;

    let mut layout_idx = if cameras.len() > 1 { 1 } else { 0 };
    let mut page = 0usize;
    let mut fullscreen = false;
    let mut ptz_state = PtzUiState::new();
    // Camera index shown full-window after a double-click; None = grid.
    let mut single_view: Option<usize> = None;
    let mut saved_layout_idx = layout_idx;
    let mut saved_page = page;
    // Motion alerts: per-camera enable from config; `m` is a global master
    // switch (starts armed so the per-tile icons work immediately).
    // Per-camera sensitivity: runtime adjustment (shift+click badge) wins,
    // then the config's motion_threshold, then the default.
    let mut runtime_thresholds = load_thresholds();
    let mut motion: Vec<MotionAlertState> = cameras
        .iter()
        .map(|h| {
            let thr = runtime_thresholds
                .get(&h.camera.name)
                .copied()
                .or(h.camera.motion_threshold.map(|t| t as f32))
                .unwrap_or(DEFAULT_MOTION_THRESHOLD);
            MotionAlertState::new(h.camera.motion_alert, thr)
        })
        .collect();
    for (cam, state) in cameras.iter().zip(&motion) {
        if state.enabled {
            info!(
                "camera {}: motion alerts armed, sensitivity {} (threshold {:.3})",
                cam.camera.name,
                sensitivity_label(state.threshold()),
                state.threshold()
            );
        }
    }
    if !motion.iter().any(|m| m.enabled) {
        info!(
            "motion alerts: press m to arm all cameras, or click a tile's badge for one \
             (shift+click a badge cycles sensitivity, right-drag draws a freeform zone)"
        );
    }
    let mut motion_armed = true;
    // Per-camera motion zones (right-drag draws freeform, right-click or x
    // clears), persisted by camera name across runs.
    let saved_zones = load_zones();
    let mut zones: Vec<Option<MotionZone>> = cameras
        .iter()
        .map(|h| saved_zones.get(&h.camera.name).cloned().map(|z| z.clamped()))
        .collect();
    for (cam, zone) in cameras.iter().zip(&zones) {
        match zone {
            Some(z) if z.is_polygon() => info!(
                "camera {}: freeform detection zone active ({} points) — motion elsewhere is ignored",
                cam.camera.name,
                z.points.len()
            ),
            Some(_) => info!(
                "camera {}: rectangle detection zone active — motion elsewhere is ignored",
                cam.camera.name
            ),
            None => {}
        }
    }
    let mut zone_drag: Option<ZoneStroke> = None;
    // Audible beep on motion: SDL2 audio, with external player and
    // terminal-bell fallbacks (see MotionBeep).
    let mut beep = MotionBeep::new(&sdl);

    'main: while !shutdown.load(Ordering::Relaxed) {
        // Snapshot modifier state before polling (mouse events do not
        // carry it): used for shift+click on the motion badge.
        let shift_held = sdl
            .keyboard()
            .mod_state()
            .intersects(sdl2::keyboard::Mod::LSHIFTMOD | sdl2::keyboard::Mod::RSHIFTMOD);
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'main,
                Event::MouseButtonDown {
                    mouse_btn: MouseButton::Right,
                    x,
                    y,
                    ..
                } => {
                    let layout = if single_view.is_some() {
                        1
                    } else {
                        LAYOUTS[layout_idx]
                    };
                    let (out_w, out_h) = canvas.output_size().unwrap_or((0, 0));
                    let (win_w, win_h) = canvas.window().size();
                    let cells = visible_cells(single_view, layout, cameras.len(), out_w, out_h);
                    let Some((px, py)) = window_to_output(x, y, win_w, win_h, out_w, out_h) else {
                        continue;
                    };
                    if let Some(tile) = tile_at(px, py, &cells) {
                        let cam_idx = if single_view.is_some() {
                            single_view.unwrap_or(0)
                        } else {
                            page * layout + tile
                        };
                        if cameras.get(cam_idx).is_some() {
                            let cell = cells[tile];
                            let dest = video_dest(&textures, cam_idx, cell);
                            let start = point_to_norm(px, py, dest);
                            zone_drag = Some(ZoneStroke {
                                cam_idx,
                                cell,
                                points: start.map(|(x, y)| [x, y]).into_iter().collect(),
                                last_px: (px, py),
                                start_px: (px, py),
                                max_move: 0,
                            });
                        }
                    }
                }
                Event::MouseMotion { x, y, .. } => {
                    if let Some(stroke) = zone_drag.as_mut() {
                        let (out_w, out_h) = canvas.output_size().unwrap_or((0, 0));
                        let (win_w, win_h) = canvas.window().size();
                        if let Some((px, py)) = window_to_output(x, y, win_w, win_h, out_w, out_h) {
                            let (lx, ly) = stroke.last_px;
                            if (px - lx).abs() >= STROKE_MIN_SAMPLE_PX
                                || (py - ly).abs() >= STROKE_MIN_SAMPLE_PX
                            {
                                let dest = video_dest(&textures, stroke.cam_idx, stroke.cell);
                                if let Some((nx, ny)) = point_to_norm(px, py, dest) {
                                    stroke.points.push([nx, ny]);
                                }
                                stroke.last_px = (px, py);
                                let (sx, sy) = stroke.start_px;
                                stroke.max_move = stroke.max_move.max((px - sx).abs().max((py - sy).abs()));
                            }
                        }
                    }
                }
                Event::MouseButtonUp {
                    mouse_btn: MouseButton::Right,
                    ..
                } => {
                    if let Some(stroke) = zone_drag.take() {
                        let name = cameras[stroke.cam_idx].camera.name.clone();
                        match stroke_to_zone(&stroke) {
                            None => {
                                // Plain right-click (or a degenerate stroke)
                                // clears the zone.
                                zones[stroke.cam_idx] = None;
                                info!("camera {name}: motion zone cleared");
                            }
                            Some(zone) => {
                                if zone.is_polygon() {
                                    info!(
                                        "camera {name}: freeform motion zone set ({} points, bbox x={:.2} y={:.2} w={:.2} h={:.2})",
                                        zone.points.len(),
                                        zone.x,
                                        zone.y,
                                        zone.w,
                                        zone.h
                                    );
                                } else {
                                    info!(
                                        "camera {name}: motion zone set to x={:.2} y={:.2} w={:.2} h={:.2}",
                                        zone.x, zone.y, zone.w, zone.h
                                    );
                                }
                                zones[stroke.cam_idx] = Some(zone);
                            }
                        }
                        persist_zones(cameras, &zones);
                    }
                }
                Event::MouseButtonDown {
                    mouse_btn: MouseButton::Left,
                    x,
                    y,
                    clicks,
                    ..
                } => {
                    let layout = if single_view.is_some() {
                        1
                    } else {
                        LAYOUTS[layout_idx]
                    };
                    let (out_w, out_h) = canvas.output_size().unwrap_or((0, 0));
                    let (win_w, win_h) = canvas.window().size();
                    let cells =
                        visible_cells(single_view, layout, cameras.len(), out_w, out_h);
                    let output_pos = window_to_output(x, y, win_w, win_h, out_w, out_h);
                    let hit = output_pos.and_then(|(px, py)| tile_at(px, py, &cells));
                    let on_icon = hit.zip(output_pos).and_then(|(tile, (px, py))| {
                        let icon = motion_icon_rect(cells[tile]);
                        tile_contains(icon, px, py).then_some(tile)
                    });
                    let cam_idx = hit.map(|tile| {
                        if single_view.is_some() {
                            single_view.unwrap_or(0)
                        } else {
                            page * layout + tile
                        }
                    });

                    if on_icon.is_some() {
                        // Click on the badge toggles this camera's alerts;
                        // shift+click cycles its sensitivity instead.
                        if let Some(idx) = cam_idx {
                            // Modifiers are not part of mouse events; use the
                            // shift state captured at the top of the frame.
                            if shift_held {
                                if let Some(state) = motion.get_mut(idx) {
                                    let next = next_sensitivity(state.threshold());
                                    state.set_threshold(next);
                                    runtime_thresholds
                                        .insert(cameras[idx].camera.name.clone(), next);
                                    save_thresholds(&runtime_thresholds);
                                    info!(
                                        "camera {}: motion sensitivity {} (threshold {:.3})",
                                        cameras[idx].camera.name,
                                        sensitivity_label(next),
                                        next
                                    );
                                }
                            } else if let Some(state) = motion.get_mut(idx) {
                                state.enabled = !state.enabled;
                                info!(
                                    "camera {}: motion alerts {}",
                                    cameras[idx].camera.name,
                                    if state.enabled { "ENABLED" } else { "DISABLED" }
                                );
                            }
                        }
                    } else if clicks >= 2 {
                        if single_view.is_some() {
                            // Any double-click while maximized restores the grid.
                            single_view = None;
                            layout_idx = saved_layout_idx;
                            page = saved_page;
                            info!("restored grid layout {}", LAYOUTS[layout_idx]);
                        } else if let Some(cam_idx) = cam_idx.filter(|i| cameras.get(*i).is_some())
                        {
                            saved_layout_idx = layout_idx;
                            saved_page = page;
                            single_view = Some(cam_idx);
                            info!("single view: {}", cameras[cam_idx].camera.name);
                        }
                    }
                }
                Event::KeyDown {
                    keycode: Some(key),
                    repeat: false,
                    ..
                } => {
                    if let Some(ptz) = ptz {
                        match key {
                            Keycode::P => {
                                ptz_state.cycle(ptz.targets.len());
                                ptz_state.sync(ptz, cameras);
                                continue;
                            }
                            Keycode::Num1 | Keycode::Kp1
                            | Keycode::Num4 | Keycode::Kp4
                            | Keycode::Num9 | Keycode::Kp9 => {
                                // Layout keys must not also pan the camera.
                            }
                            _ => {}
                        }
                        if ptz_state.key_down(key) {
                            ptz_state.sync(ptz, cameras);
                            continue;
                        }
                    }
                    match key {
                        Keycode::Escape | Keycode::Q => break 'main,
                        Keycode::Num1 | Keycode::Kp1 => {
                            single_view = None;
                            layout_idx = 0;
                        }
                        Keycode::Num4 | Keycode::Kp4 => {
                            single_view = None;
                            layout_idx = 1;
                        }
                        Keycode::Num9 | Keycode::Kp9 => {
                            single_view = None;
                            layout_idx = 2;
                        }
                        Keycode::LeftBracket => {
                            single_view = None;
                            page = page.saturating_sub(1);
                        }
                        Keycode::RightBracket => {
                            single_view = None;
                            page += 1;
                        }
                        Keycode::F | Keycode::F11 => {
                            fullscreen = !fullscreen;
                            let mode = if fullscreen {
                                FullscreenType::Desktop
                            } else {
                                FullscreenType::Off
                            };
                            if let Err(err) = canvas.window_mut().set_fullscreen(mode) {
                                log::warn!("fullscreen toggle failed: {err}");
                                fullscreen = !fullscreen;
                            }
                        }
                        Keycode::Space => {
                            single_view = None;
                            layout_idx = (layout_idx + 1) % LAYOUTS.len();
                        }
                        Keycode::M => {
                            // Master switch: toggles motion alerts on every
                            // camera (matches the --motion flag), not just an
                            // internal gate the user cannot see.
                            motion_armed = !motion_armed;
                            for state in &mut motion {
                                state.enabled = motion_armed;
                            }
                            info!(
                                "motion alerts {} on all cameras",
                                if motion_armed { "ARMED" } else { "DISARMED" }
                            );
                        }
                        Keycode::X => {
                            // Clear zone on the PTZ-selected camera, else camera 0.
                            let cam_idx = ptz
                                .and_then(|p| p.targets.get(ptz_state.selected))
                                .map(|(i, _)| *i)
                                .unwrap_or(0);
                            if zones.get(cam_idx).is_some() {
                                let name = cameras[cam_idx].camera.name.clone();
                                zones[cam_idx] = None;
                                info!("camera {name}: motion zone cleared");
                                persist_zones(cameras, &zones);
                            }
                        }
                        _ => {}
                    }
                }
                Event::KeyUp {
                    keycode: Some(key),
                    repeat: false,
                    ..
                } => {
                    if let Some(ptz) = ptz {
                        if ptz_state.key_up(key) {
                            ptz_state.sync(ptz, cameras);
                        }
                    }
                }
                _ => {}
            }
        }

        if let Err(err) = refresh_textures(
            cameras,
            &texture_creator,
            &mut textures,
            &mut motion,
            motion_armed,
            &zones,
            |cam| beep.play(cam),
        ) {
            log::error!("texture update failed: {err:#}");
        }

        let ptz_selected = ptz.map(|p| p.targets.get(ptz_state.selected).map(|(i, _)| *i));
        let now = Instant::now();
        let flashing: Vec<usize> = motion
            .iter()
            .enumerate()
            .filter(|(_, m)| m.is_flashing(now))
            .map(|(i, _)| i)
            .collect();
        let motion_enabled: Vec<bool> = motion.iter().map(|m| m.enabled).collect();
        let drag_overlay = zone_drag
            .as_ref()
            .filter(|d| d.points.len() >= 2)
            .map(|d| (d.cam_idx, d.points.clone()));
        render(
            &mut canvas,
            RenderState {
                cameras,
                textures: &textures,
                layout: LAYOUTS[layout_idx],
                page,
                ptz_selected: ptz_selected.flatten(),
                single_view,
                motion_flashing: &flashing,
                motion_enabled: &motion_enabled,
                motion_armed,
                zones: &zones,
                drag_overlay,
            },
        )?;
        canvas.present();
    }

    // Persist window size for the next run.
    let (win_w, win_h) = canvas.window().size();
    crate::window_state::WindowState::default().save(win_w, win_h);

    // Release any camera still moving before exit.
    if let Some(ptz) = ptz {
        ptz.send(PtzCommand::Stop {
            camera: ptz
                .targets
                .get(ptz_state.selected)
                .map(|(i, _)| *i)
                .unwrap_or(0),
        });
    }

    info!("UI loop exited");
    Ok(())
}

fn refresh_textures<'a>(
    cameras: &[CameraHandle],
    texture_creator: &'a sdl2::render::TextureCreator<sdl2::video::WindowContext>,
    textures: &mut HashMap<usize, CamTexture<'a>>,
    motion: &mut [MotionAlertState],
    motion_armed: bool,
    zones: &[Option<MotionZone>],
    mut on_motion: impl FnMut(usize),
) -> Result<()> {
    for (idx, handle) in cameras.iter().enumerate() {
        let Some(frame) = handle.frames.take() else {
            continue;
        };
        if frame.width == 0 || frame.height == 0 {
            continue;
        }

        if let Some(state) = motion.get_mut(idx) {
            let now = Instant::now();
            let zone = zones.get(idx).and_then(|z| z.as_ref());
            if state.observe(&frame, motion_armed, zone, now) {
                warn!(
                    "camera {}: MOTION detected (score {:.2} >= {:.2})",
                    handle.camera.name,
                    state.last_score(),
                    state.threshold()
                );
                on_motion(idx);
            }
        }

        let stride = frame.stride;
        match textures.get_mut(&idx) {
            Some(cam_tex) if cam_tex.width == frame.width && cam_tex.height == frame.height => {
                cam_tex
                    .texture
                    .update(None, &frame.rgb, stride)
                    .context("texture update failed")?;
            }
            _ => {
                let mut texture = texture_creator
                    .create_texture_streaming(PixelFormatEnum::RGB24, frame.width, frame.height)
                    .context("texture create failed")?;
                texture
                    .update(None, &frame.rgb, stride)
                    .context("texture update failed")?;
                textures.insert(
                    idx,
                    CamTexture {
                        texture,
                        width: frame.width,
                        height: frame.height,
                    },
                );
            }
        }
    }
    Ok(())
}

struct RenderState<'a> {
    cameras: &'a [CameraHandle],
    textures: &'a HashMap<usize, CamTexture<'a>>,
    layout: usize,
    page: usize,
    ptz_selected: Option<usize>,
    single_view: Option<usize>,
    motion_flashing: &'a [usize],
    /// Per-camera motion-alert enable (drives the corner badge).
    motion_enabled: &'a [bool],
    /// Global motion master switch (`m`).
    motion_armed: bool,
    /// Per-camera detection zone (video-normalized); None = whole frame.
    zones: &'a [Option<MotionZone>],
    /// In-progress freehand stroke: (camera index, normalized points).
    drag_overlay: Option<(usize, Vec<[f32; 2]>)>,
}

fn render(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    state: RenderState<'_>,
) -> Result<()> {
    let RenderState {
        cameras,
        textures,
        layout,
        page,
        ptz_selected,
        single_view,
        motion_flashing,
        motion_enabled,
        motion_armed,
        zones,
        drag_overlay,
    } = state;
    let (out_w, out_h) = canvas
        .output_size()
        .map_err(|e| anyhow::anyhow!("output_size failed: {e}"))?;
    canvas.set_draw_color(sdl2::pixels::Color::RGB(20, 20, 20));
    canvas.clear();

    if cameras.is_empty() {
        return Ok(());
    }

    // Double-clicked tile: show only that camera, full window.
    let (cells, stride, mut title) = if let Some(idx) = single_view
        .filter(|i| cameras.get(*i).is_some())
    {
        let name = &cameras[idx].camera.name;
        let cells = vec![Rect::new(0, 0, out_w, out_h)];
        let title = format!("EZVIZ View — {name} (double-click to restore grid)");
        (cells, idx, title)
    } else {
        let tiles = layout.min(cameras.len());
        let cells = grid_cells(tiles, out_w, out_h);
        let mut title = format!(
            "EZVIZ View — {} camera(s), grid {layout}, page {}",
            cameras.len(),
            page + 1
        );
        if let Some(idx) = ptz_selected {
            if let Some(handle) = cameras.get(idx) {
                title.push_str(&format!("  |  PTZ: {}", handle.camera.name));
            }
        }
        (cells, page * layout, title)
    };
    let motion_status = if !motion_armed {
        "motion: OFF (m)"
    } else if motion_enabled.iter().any(|e| *e) {
        "motion: ON (m off, right-drag draws zone)"
    } else {
        "motion: OFF (m enables all)"
    };
    title.push_str("  |  ");
    title.push_str(motion_status);
    let _ = canvas.window_mut().set_title(&title);

    for (tile, cell) in cells.iter().enumerate() {
        let cam_idx = stride + tile;
        let Some(handle) = cameras.get(cam_idx) else {
            continue;
        };

        canvas
            .set_draw_color(sdl2::pixels::Color::RGB(0, 0, 0));
        let _ = canvas.fill_rect(*cell);

        if let Some(cam_tex) = textures.get(&cam_idx) {
            let dest = aspect_fit(
                cam_tex.width,
                cam_tex.height,
                cell.x(),
                cell.y(),
                cell.width(),
                cell.height(),
            );
            let _ = canvas.copy(&cam_tex.texture, None, Some(dest));
        }

        let border_color = if motion_flashing.contains(&cam_idx) {
            // Orange: motion just detected on this camera.
            sdl2::pixels::Color::RGB(255, 140, 0)
        } else if ptz_selected == Some(cam_idx) {
            // Cyan: this camera is the current PTZ keyboard target.
            sdl2::pixels::Color::RGB(0, 210, 230)
        } else {
            let status = handle.status.lock().unwrap().clone();
            match (&status, textures.contains_key(&cam_idx)) {
                (StreamStatus::Live, true) => sdl2::pixels::Color::RGB(40, 200, 40),
                (StreamStatus::Live, false) => sdl2::pixels::Color::RGB(40, 160, 40),
                (StreamStatus::Connecting, _) => sdl2::pixels::Color::RGB(220, 200, 40),
                (StreamStatus::Reconnecting(_), _) => sdl2::pixels::Color::RGB(220, 40, 40),
            }
        };
        draw_border(canvas, *cell, border_color)?;

        // Saved detection zone outline (yellow) when one is set: polygon
        // for freeform zones, rectangle otherwise.
        if let Some(zone) = zones.get(cam_idx).cloned().flatten() {
            let dest = video_dest(textures, cam_idx, *cell);
            let color = sdl2::pixels::Color::RGB(255, 220, 40);
            if zone.is_polygon() {
                draw_outline(canvas, &poly_norm_to_screen(&zone.points, dest), true, color)
                    .map_err(|e| anyhow::anyhow!("zone outline: {e}"))?;
            } else if let Some(zr) = zone_screen_rect(&zone, dest) {
                canvas.set_draw_color(color);
                canvas
                    .draw_rect(zr)
                    .map_err(|e| anyhow::anyhow!("zone outline: {e}"))?;
            }
        }

        // Live freehand stroke while right-dragging a new zone on this tile.
        if let Some((drag_cam, points)) = &drag_overlay {
            if *drag_cam == cam_idx {
                let dest = video_dest(textures, cam_idx, *cell);
                draw_outline(
                    canvas,
                    &poly_norm_to_screen(points, dest),
                    false,
                    sdl2::pixels::Color::RGB(0, 200, 255),
                )
                .map_err(|e| anyhow::anyhow!("zone drag: {e}"))?;
            }
        }

        // Per-camera motion badge (click to toggle this camera's alerts).
        let enabled = motion_enabled.get(cam_idx).copied().unwrap_or(false);
        draw_motion_icon(canvas, motion_icon_rect(*cell), enabled)?;
    }

    Ok(())
}

/// Cell rects for the cameras currently on screen.
fn visible_cells(
    single_view: Option<usize>,
    layout: usize,
    camera_count: usize,
    out_w: u32,
    out_h: u32,
) -> Vec<Rect> {
    if single_view.is_some() {
        vec![Rect::new(0, 0, out_w, out_h)]
    } else {
        grid_cells(layout.min(camera_count), out_w, out_h)
    }
}

/// The aspect-fitted video rect inside a tile (falls back to the cell when
/// no frame has arrived yet).
fn video_dest(textures: &HashMap<usize, CamTexture<'_>>, cam_idx: usize, cell: Rect) -> Rect {
    textures
        .get(&cam_idx)
        .map(|t| aspect_fit(t.width, t.height, cell.x(), cell.y(), cell.width(), cell.height()))
        .unwrap_or(cell)
}

/// Turn a finished freehand stroke into a motion zone. Straight drags
/// become rectangles; real freehand loops become polygons; clicks and
/// degenerate scribbles return None (clear the zone).
fn stroke_to_zone(stroke: &ZoneStroke) -> Option<MotionZone> {
    if stroke.max_move <= STROKE_CLICK_PX || stroke.points.len() < 2 {
        return None;
    }
    let simple =
        crate::motion::simplify_outline(&stroke.points, OUTLINE_SIMPLIFY_EPSILON);
    match simple.len() {
        0..=1 => None,
        // A drag with no meaningful width collapsed to a line: keep the
        // old rectangle behaviour for corner-to-corner drags.
        2 => MotionZone::from_corners(simple[0][0], simple[0][1], simple[1][0], simple[1][1]),
        _ => MotionZone::from_outline(&simple),
    }
}

fn point_to_norm(px: i32, py: i32, dest: Rect) -> Option<(f32, f32)> {
    if dest.width() == 0 || dest.height() == 0 {
        return None;
    }
    let nx = (px - dest.x) as f32 / dest.width() as f32;
    let ny = (py - dest.y) as f32 / dest.height() as f32;
    Some((nx.clamp(0.0, 1.0), ny.clamp(0.0, 1.0)))
}

/// Map a normalized zone onto the aspect-fitted video rect for drawing.
fn zone_screen_rect(zone: &MotionZone, dest: Rect) -> Option<Rect> {
    if dest.width() == 0 || dest.height() == 0 {
        return None;
    }
    let x = dest.x() + (zone.x * dest.width() as f32) as i32;
    let y = dest.y() + (zone.y * dest.height() as f32) as i32;
    let w = ((zone.w * dest.width() as f32) as u32).max(1);
    let h = ((zone.h * dest.height() as f32) as u32).max(1);
    Some(Rect::new(x, y, w, h))
}

/// Map normalized outline points onto the aspect-fitted video rect.
fn poly_norm_to_screen(points: &[[f32; 2]], dest: Rect) -> Vec<(i32, i32)> {
    if dest.width() == 0 || dest.height() == 0 {
        return Vec::new();
    }
    points
        .iter()
        .map(|p| {
            let x = dest.x() + (p[0].clamp(0.0, 1.0) * dest.width() as f32) as i32;
            let y = dest.y() + (p[1].clamp(0.0, 1.0) * dest.height() as f32) as i32;
            (x, y)
        })
        .collect()
}

/// Polyline through `points` (optionally closed back to the first point).
fn draw_outline(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    points: &[(i32, i32)],
    close: bool,
    color: sdl2::pixels::Color,
) -> std::result::Result<(), String> {
    if points.len() < 2 {
        return Ok(());
    }
    canvas.set_draw_color(color);
    let mut path: Vec<sdl2::rect::Point> = points
        .iter()
        .map(|(x, y)| sdl2::rect::Point::new(*x, *y))
        .collect();
    if close {
        path.push(path[0]);
    }
    canvas.draw_lines(&path[..])
}

fn persist_zones(cameras: &[CameraHandle], zones: &[Option<MotionZone>]) {
    let map: HashMap<String, MotionZone> = cameras
        .iter()
        .zip(zones)
        .filter_map(|(cam, zone)| zone.clone().map(|z| (cam.camera.name.clone(), z)))
        .collect();
    save_zones(&map);
}

/// Motion alert sound: SDL audio queue when available, else an external
/// player (paplay/pw-play/aplay/ffplay), else the terminal bell. The SDL
/// device is retried lazily so a missing audio server at startup is not
/// permanent.
struct MotionBeep {
    audio: Option<sdl2::AudioSubsystem>,
    device: Option<sdl2::audio::AudioQueue<f32>>,
    external_logged: bool,
    bell_only_logged: bool,
}

/// Base beep pitch per camera, so you can hear WHICH camera alerted.
/// Cameras beyond the list reuse the last pitch.
const CAMERA_BEEP_FREQS: [f32; 8] = [523.0, 659.0, 784.0, 880.0, 587.0, 698.0, 831.0, 988.0];

impl MotionBeep {
    fn new(sdl: &sdl2::Sdl) -> Self {
        let audio = sdl.audio().ok();
        let device = audio.as_ref().and_then(open_beep_queue);
        if device.is_some() {
            info!("motion beep: SDL audio");
        } else {
            info!("motion beep: SDL audio unavailable; will try an external player, then the terminal bell");
        }
        Self {
            audio,
            device,
            external_logged: false,
            bell_only_logged: false,
        }
    }

    fn play(&mut self, camera: usize) {
        let freq = CAMERA_BEEP_FREQS
            .get(camera)
            .copied()
            .unwrap_or(*CAMERA_BEEP_FREQS.last().unwrap());
        if self.device.is_none() {
            // Retry: the audio server may have come up after startup.
            self.device = self.audio.as_ref().and_then(open_beep_queue);
            if self.device.is_some() {
                info!("motion beep: SDL audio device ready");
            }
        }
        if let Some(dev) = self.device.as_ref() {
            let samples = beep_samples(freq);
            // Drop backlog so repeated alerts do not queue seconds of tone.
            if dev.size() > 44_100 * 4 {
                dev.clear();
            }
            if dev.queue_audio(&samples).is_ok() {
                dev.resume();
                crate::motion::beep(); // terminal bell too
                return;
            }
            // Queueing failed (device likely lost): drop it and fall through.
            self.device = None;
        }
        if play_external_beep(freq) && !self.external_logged {
            info!("motion beep: playing via external audio player");
            self.external_logged = true;
        } else if !self.external_logged && !self.bell_only_logged {
            info!("motion beep: no audio player found; motion alerts use the terminal bell only");
            self.bell_only_logged = true;
        }
        crate::motion::beep(); // terminal bell too
    }
}

fn open_beep_queue(audio: &sdl2::AudioSubsystem) -> Option<sdl2::audio::AudioQueue<f32>> {
    audio
        .open_queue(
            None,
            &sdl2::audio::AudioSpecDesired {
                freq: Some(44_100),
                channels: Some(1),
                samples: Some(1024),
            },
        )
        .ok()
}

/// Short beep tone with a soft envelope (not a harsh click).
fn beep_samples(freq_hz: f32) -> Vec<f32> {
    const SAMPLE_RATE: f32 = 44_100.0;
    const DURATION: f32 = 0.12;
    let n = (SAMPLE_RATE * DURATION) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE;
            let env = (1.0 - i as f32 / n as f32).min(t / 0.005);
            (t * freq_hz * std::f32::consts::TAU).sin() * 0.25 * env.max(0.0)
        })
        .collect()
}

/// PCM16 mono WAV container around the given samples.
fn wav_pcm16_mono(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    use std::io::Write as _;
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    let _ = wav.write_all(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    let _ = wav.write_all(&16u32.to_le_bytes()); // fmt chunk size
    let _ = wav.write_all(&1u16.to_le_bytes()); // PCM
    let _ = wav.write_all(&1u16.to_le_bytes()); // mono
    let _ = wav.write_all(&sample_rate.to_le_bytes());
    let _ = wav.write_all(&(sample_rate * 2).to_le_bytes()); // byte rate
    let _ = wav.write_all(&2u16.to_le_bytes()); // block align
    let _ = wav.write_all(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    let _ = wav.write_all(&data_len.to_le_bytes());
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        let _ = wav.write_all(&v.to_le_bytes());
    }
    wav
}

/// Play the beep through the first external player found. Returns true
/// when a player was started (best-effort; play finishes on its own).
fn play_external_beep(freq_hz: f32) -> bool {
    use std::io::Write as _;
    use std::process::{Command, Stdio};

    const PLAYERS: &[&[&str]] = &[
        &["paplay"],
        &["pw-play", "-"],
        &["aplay", "-q", "-t", "wav"],
        &["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"],
    ];
    let wav = wav_pcm16_mono(&beep_samples(freq_hz), 44_100);
    for args in PLAYERS {
        let Some((cmd, extra)) = args.split_first() else {
            continue;
        };
        let child = Command::new(cmd)
            .args(extra)
            .stdin(Stdio::piped())
            .stderr(Stdio::null())
            .spawn();
        let Ok(mut child) = child else {
            continue;
        };
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(&wav);
        }
        // Reap the player without blocking the UI thread.
        std::thread::spawn(move || {
            let _ = child.wait();
        });
        return true;
    }
    false
}

/// Top-right badge on each tile; click toggles motion alerts for that camera.
const MOTION_ICON_SIZE: i32 = 22;
const MOTION_ICON_MARGIN: i32 = 8;

fn motion_icon_rect(cell: Rect) -> Rect {
    let x = cell.x() + cell.width() as i32 - MOTION_ICON_SIZE - MOTION_ICON_MARGIN;
    let y = cell.y() + MOTION_ICON_MARGIN;
    Rect::new(x, y, MOTION_ICON_SIZE as u32, MOTION_ICON_SIZE as u32)
}

fn tile_contains(rect: Rect, x: i32, y: i32) -> bool {
    x >= rect.x()
        && y >= rect.y()
        && x < rect.x() + rect.width() as i32
        && y < rect.y() + rect.height() as i32
}

fn draw_motion_icon(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    rect: Rect,
    enabled: bool,
) -> Result<()> {
    let color = |r: u8, g: u8, b: u8| sdl2::pixels::Color::RGB(r, g, b);

    // Badge background + border.
    let (bg, border, dot) = if enabled {
        (color(180, 90, 0), color(255, 170, 40), color(255, 235, 180))
    } else {
        (color(50, 50, 50), color(95, 95, 95), color(25, 25, 25))
    };
    canvas.set_draw_color(bg);
    canvas
        .fill_rect(rect)
        .map_err(|e| anyhow::anyhow!("icon fill: {e}"))?;
    canvas.set_draw_color(border);
    canvas
        .draw_rect(rect)
        .map_err(|e| anyhow::anyhow!("icon border: {e}"))?;

    // Center "record" dot: bright when armed for this camera, dark when off.
    let inset = 8;
    let dw = (rect.width() as i32 - inset * 2).max(1) as u32;
    let dh = (rect.height() as i32 - inset * 2).max(1) as u32;
    let dot_rect = Rect::new(rect.x() + inset, rect.y() + inset, dw, dh);
    canvas.set_draw_color(dot);
    canvas
        .fill_rect(dot_rect)
        .map_err(|e| anyhow::anyhow!("icon dot: {e}"))?;
    Ok(())
}

/// Map mouse coordinates from window points to output pixels (HiDPI-safe).
fn window_to_output(
    win_x: i32,
    win_y: i32,
    win_w: u32,
    win_h: u32,
    out_w: u32,
    out_h: u32,
) -> Option<(i32, i32)> {
    if win_w == 0 || win_h == 0 {
        return None;
    }
    let sx = f64::from(out_w) / f64::from(win_w);
    let sy = f64::from(out_h) / f64::from(win_h);
    Some(((win_x as f64 * sx) as i32, (win_y as f64 * sy) as i32))
}

/// Which tile contains the given output-pixel point, if any.
fn tile_at(x: i32, y: i32, cells: &[Rect]) -> Option<usize> {
    cells.iter().position(|c| {
        x >= c.x()
            && y >= c.y()
            && x < c.x() + c.width() as i32
            && y < c.y() + c.height() as i32
    })
}

fn grid_cells(tiles: usize, out_w: u32, out_h: u32) -> Vec<Rect> {
    if tiles == 0 || out_w == 0 || out_h == 0 {
        return Vec::new();
    }
    let cols: u32 = match tiles {
        1 => 1,
        2..=4 => 2,
        5..=9 => 3,
        n => (n as f64).sqrt().ceil() as u32,
    };
    let rows = (tiles as u32).div_ceil(cols);
    let cell_w = out_w / cols;
    let cell_h = out_h / rows;
    (0..tiles as u32)
        .map(|i| {
            let row = i / cols;
            let col = i % cols;
            Rect::new(
                (col * cell_w) as i32,
                (row * cell_h) as i32,
                cell_w,
                cell_h,
            )
        })
        .collect()
}

fn aspect_fit(src_w: u32, src_h: u32, cell_x: i32, cell_y: i32, cell_w: u32, cell_h: u32) -> Rect {
    if src_w == 0 || src_h == 0 || cell_w == 0 || cell_h == 0 {
        return Rect::new(cell_x, cell_y, cell_w, cell_h);
    }
    let scale = (f64::from(cell_w) / f64::from(src_w)).min(f64::from(cell_h) / f64::from(src_h));
    let w = ((f64::from(src_w) * scale) as u32).max(1);
    let h = ((f64::from(src_h) * scale) as u32).max(1);
    let x = cell_x + ((cell_w.saturating_sub(w)) / 2) as i32;
    let y = cell_y + ((cell_h.saturating_sub(h)) / 2) as i32;
    Rect::new(x, y, w, h)
}

fn draw_border(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    rect: Rect,
    color: sdl2::pixels::Color,
) -> Result<()> {
    canvas.set_draw_color(color);
    let x = rect.x();
    let y = rect.y();
    let w = rect.width() as i32;
    let h = rect.height() as i32;
    let b = BORDER.min(w / 2).max(1);
    let edges = [
        Rect::new(x, y, w as u32, b as u32),
        Rect::new(x, y + h - b, w as u32, b as u32),
        Rect::new(x, y, b as u32, h as u32),
        Rect::new(x + w - b, y, b as u32, h as u32),
    ];
    for edge in edges {
        canvas
            .fill_rect(edge)
            .map_err(|e| anyhow::anyhow!("failed to draw border: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_for_one_tile() {
        let cells = grid_cells(1, 1920, 1080);
        assert_eq!(cells, vec![Rect::new(0, 0, 1920, 1080)]);
    }

    #[test]
    fn grid_for_four_tiles() {
        let cells = grid_cells(4, 1920, 1080);
        assert_eq!(cells.len(), 4);
        assert_eq!(cells[0], Rect::new(0, 0, 960, 540));
        assert_eq!(cells[3], Rect::new(960, 540, 960, 540));
    }

    #[test]
    fn grid_for_nine_tiles() {
        let cells = grid_cells(9, 1920, 1080);
        assert_eq!(cells.len(), 9);
        assert_eq!(cells[0].width(), 640);
        assert_eq!(cells[8], Rect::new(1280, 720, 640, 360));
    }

    #[test]
    fn aspect_fit_letterboxes_wide_frame() {
        let r = aspect_fit(1920, 1080, 0, 0, 640, 640);
        assert_eq!(r.width(), 640);
        assert_eq!(r.height(), 360);
        assert_eq!(r.y(), 140);
    }

    #[test]
    fn aspect_fit_pillarboxes_tall_frame() {
        let r = aspect_fit(480, 1920, 0, 0, 640, 640);
        assert_eq!(r.width(), 160);
        assert_eq!(r.height(), 640);
    }

    #[test]
    fn tile_at_finds_cell_under_point() {
        let cells = grid_cells(4, 1920, 1080);
        assert_eq!(tile_at(10, 10, &cells), Some(0));
        assert_eq!(tile_at(959, 539, &cells), Some(0));
        assert_eq!(tile_at(960, 540, &cells), Some(3));
        assert_eq!(tile_at(1919, 1079, &cells), Some(3));
        assert_eq!(tile_at(2000, 10, &cells), None);
    }

    #[test]
    fn window_to_output_scales_on_hidpi() {
        // 2x scale: window point (100, 50) -> output pixel (200, 100)
        assert_eq!(window_to_output(100, 50, 640, 360, 1280, 720), Some((200, 100)));
        // 1:1
        assert_eq!(window_to_output(3, 4, 800, 600, 800, 600), Some((3, 4)));
        assert_eq!(window_to_output(0, 0, 0, 600, 800, 600), None);
    }

    #[test]
    fn motion_icon_sits_in_top_right_of_cell() {
        let cell = Rect::new(0, 0, 640, 360);
        let icon = motion_icon_rect(cell);
        assert_eq!(icon.width(), MOTION_ICON_SIZE as u32);
        assert_eq!(icon.x() + icon.width() as i32 + MOTION_ICON_MARGIN, 640);
        assert_eq!(icon.y(), MOTION_ICON_MARGIN);
    }

    #[test]
    fn tile_contains_hit_testing() {
        let r = Rect::new(10, 20, 30, 40);
        assert!(tile_contains(r, 10, 20));
        assert!(tile_contains(r, 39, 59));
        assert!(!tile_contains(r, 40, 20));
        assert!(!tile_contains(r, 10, 60));
        assert!(!tile_contains(r, 9, 20));
    }

    #[test]
    fn zone_screen_rect_maps_normalized_to_dest() {
        let dest = Rect::new(100, 50, 640, 360);
        let zone = MotionZone::rect(0.25, 0.5, 0.5, 0.25);
        let r = zone_screen_rect(&zone, dest).unwrap();
        assert_eq!(r.x(), 100 + 160);
        assert_eq!(r.y(), 50 + 180);
        assert_eq!(r.width(), 320);
        assert_eq!(r.height(), 90);
    }

    #[test]
    fn poly_norm_to_screen_maps_and_clamps() {
        let dest = Rect::new(100, 50, 640, 360);
        let pts = poly_norm_to_screen(&[[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]], dest);
        assert_eq!(pts, vec![(100, 50), (740, 230), (420, 410)]);
        // Out-of-range values are clamped into the video rect.
        let pts = poly_norm_to_screen(&[[-1.0, 2.0]], dest);
        assert_eq!(pts, vec![(100, 410)]);
    }

    fn stroke(points_px: &[(i32, i32)], max_move: i32) -> ZoneStroke {
        ZoneStroke {
            cam_idx: 0,
            cell: Rect::new(0, 0, 640, 360),
            points: points_px
                .iter()
                .map(|(x, y)| [*x as f32 / 640.0, *y as f32 / 360.0])
                .collect(),
            last_px: *points_px.last().unwrap(),
            start_px: points_px[0],
            max_move,
        }
    }

    #[test]
    fn stroke_to_zone_click_is_none() {
        // A right-click (no movement) clears the zone.
        let s = stroke(&[(100, 100), (100, 100)], 0);
        assert!(stroke_to_zone(&s).is_none());
    }

    #[test]
    fn stroke_to_zone_straight_drag_becomes_rect() {
        // Exactly collinear corner-to-corner drag collapses to 2 points
        // after simplification, which becomes a rectangle zone.
        let s = stroke(&[(64, 36), (352, 198), (640, 360)], 400);
        let zone = stroke_to_zone(&s).unwrap();
        assert!(!zone.is_polygon());
        assert!(zone.w > 0.8);
        assert!(zone.h > 0.8);
    }

    #[test]
    fn stroke_to_zone_freeform_loop_becomes_polygon() {
        // Rough square drawn freehand.
        let s = stroke(
            &[
                (128, 72),
                (320, 72),
                (512, 72),
                (512, 180),
                (512, 288),
                (320, 288),
                (128, 288),
                (128, 180),
                (128, 72),
            ],
            400,
        );
        let zone = stroke_to_zone(&s).unwrap();
        assert!(zone.is_polygon());
        assert!(zone.points.len() >= 3);
        // Bounding box should cover the drawn square.
        assert!(zone.x <= 0.21);
        assert!(zone.x + zone.w >= 0.79);
    }

    #[test]
    fn video_dest_falls_back_to_cell_without_texture() {
        let cell = Rect::new(10, 20, 300, 200);
        assert_eq!(video_dest(&HashMap::new(), 0, cell), cell);
    }

    #[test]
    fn beep_pitches_are_distinct_per_camera() {
        for (i, f) in CAMERA_BEEP_FREQS.iter().enumerate() {
            assert!(*f > 400.0 && *f < 1200.0, "pitch {i} out of audible mid range");
            assert!(
                !CAMERA_BEEP_FREQS[..i].contains(f),
                "pitch {i} duplicates an earlier camera"
            );
        }
    }
}
