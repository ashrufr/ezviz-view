use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::stream::VideoFrame;

/// Resolution the frames are downscaled to before comparing. Small enough
/// that per-frame cost is negligible, big enough to catch a person walking.
pub const ANALYSIS_W: u32 = 80;
pub const ANALYSIS_H: u32 = 45;

/// Minimum time between audible/visual alerts for one camera, so a scene
/// with continuous motion does not spam the terminal bell.
const ALERT_COOLDOWN: Duration = Duration::from_secs(2);

/// Default sensitivity: mean per-pixel change (0..=1) needed to alert.
/// Calibrated against real cameras at the 15 fps analysis cadence, where a
/// quiet scene scores ~0.002 and a person in frame ~0.005..0.03.
pub const DEFAULT_MOTION_THRESHOLD: f32 = 0.012;

/// How long the tile border stays highlighted after motion was detected.
const FLASH_DURATION: Duration = Duration::from_millis(1500);

/// Minimum enclosed area (fraction of the frame) for a freeform zone, so
/// accidental scribbles do not become near-empty detection areas.
const MIN_OUTLINE_AREA: f32 = 0.002;

/// Ramer-Douglas-Peucker tolerance (normalized units) when simplifying a
/// freehand stroke into a polygon.
pub const OUTLINE_SIMPLIFY_EPSILON: f32 = 0.015;

/// Detection area in video-normalized coordinates (0.0..=1.0 of the frame).
/// When absent, the whole frame is analyzed. A zone is either a plain
/// rectangle (`points` empty) or a freeform polygon (`points` with the
/// outline; `x/y/w/h` is then just its bounding box).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotionZone {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Freeform outline vertices (normalized). When it has >= 3 points only
    /// motion inside the polygon is analyzed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub points: Vec<[f32; 2]>,
}

impl MotionZone {
    /// A plain rectangular zone.
    pub fn rect(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            x,
            y,
            w,
            h,
            points: Vec::new(),
        }
    }

    /// Build a zone from two corner points in normalized space (order-free).
    pub fn from_corners(x0: f32, y0: f32, x1: f32, y1: f32) -> Option<Self> {
        let x = x0.min(x1);
        let y = y0.min(y1);
        let w = (x0 - x1).abs();
        let h = (y0 - y1).abs();
        // Ignore accidental micro-drags.
        if w < 0.02 || h < 0.02 {
            return None;
        }
        Some(Self::rect(
            x.clamp(0.0, 1.0),
            y.clamp(0.0, 1.0),
            w.min(1.0 - x.clamp(0.0, 1.0)).max(0.01),
            h.min(1.0 - y.clamp(0.0, 1.0)).max(0.01),
        ))
    }

    /// Build a zone from a freehand outline (normalized, need not be closed).
    /// The stroke is simplified; shapes that collapse or enclose too little
    /// area are rejected (`None`).
    pub fn from_outline(raw: &[[f32; 2]]) -> Option<Self> {
        let points: Vec<[f32; 2]> = raw
            .iter()
            .map(|p| [p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0)])
            .collect();
        let points = simplify_outline(&points, OUTLINE_SIMPLIFY_EPSILON);
        if points.len() < 3 || polygon_area(&points) < MIN_OUTLINE_AREA {
            return None;
        }
        let min_x = points.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let max_x = points.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        let min_y = points.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let max_y = points.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
        Some(Self {
            x: min_x,
            y: min_y,
            w: (max_x - min_x).max(0.01),
            h: (max_y - min_y).max(0.01),
            points,
        })
    }

    /// Clamp into the unit square (e.g. after loading a hand-edited file).
    pub fn clamped(self) -> Self {
        let x = self.x.clamp(0.0, 1.0);
        let y = self.y.clamp(0.0, 1.0);
        Self {
            x,
            y,
            w: self.w.clamp(0.01, 1.0 - x),
            h: self.h.clamp(0.01, 1.0 - y),
            points: self
                .points
                .into_iter()
                .map(|p| [p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0)])
                .collect(),
        }
    }

    /// The polygon used for detection/drawing: the freeform outline when
    /// present, else the rectangle as four corners.
    pub fn outline(&self) -> Vec<[f32; 2]> {
        if self.points.len() >= 3 {
            self.points.clone()
        } else {
            let z = self.clone().clamped();
            vec![
                [z.x, z.y],
                [z.x + z.w, z.y],
                [z.x + z.w, z.y + z.h],
                [z.x, z.y + z.h],
            ]
        }
    }

    pub fn is_polygon(&self) -> bool {
        self.points.len() >= 3
    }
}

fn zones_path() -> PathBuf {
    config_dir().join("zones.toml")
}

fn thresholds_path() -> PathBuf {
    config_dir().join("thresholds.toml")
}

fn config_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("ezviz-view")
}

/// Load persisted per-camera zones, keyed by camera name.
pub fn load_zones() -> HashMap<String, MotionZone> {
    load_zones_from(&zones_path())
}

pub fn save_zones(zones: &HashMap<String, MotionZone>) {
    save_zones_to(zones, &zones_path());
}

/// Runtime-adjusted per-camera motion thresholds (shift+click on the
/// badge), keyed by camera name; takes precedence over the config file.
pub fn load_thresholds() -> HashMap<String, f32> {
    load_thresholds_from(&thresholds_path())
}

pub fn save_thresholds(thresholds: &HashMap<String, f32>) {
    save_thresholds_to(thresholds, &thresholds_path());
}

fn load_thresholds_from(path: &Path) -> HashMap<String, f32> {
    let Ok(raw) = fs::read_to_string(path) else {
        return HashMap::new();
    };
    toml::from_str::<HashMap<String, f32>>(&raw)
        .map(|m| {
            m.into_iter()
                .filter(|(_, t)| (0.001..=1.0).contains(t))
                .collect()
        })
        .unwrap_or_default()
}

fn save_thresholds_to(thresholds: &HashMap<String, f32>, path: &Path) {
    let Ok(raw) = toml::to_string(thresholds) else {
        return;
    };
    if let Some(dir) = path.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let _ = fs::write(path, raw);
}

/// Sensitivity presets cycled with shift+click on a camera badge, from
/// most sensitive (alerts on small/far motion) to least.
pub const SENSITIVITY_PRESETS: [f32; 3] = [0.004, 0.012, 0.03];

pub fn sensitivity_label(threshold: f32) -> &'static str {
    match SENSITIVITY_PRESETS
        .iter()
        .position(|p| (*p - threshold).abs() < 1e-4)
    {
        Some(0) => "HIGH",
        Some(1) => "MEDIUM",
        Some(2) => "LOW",
        _ => "CUSTOM",
    }
}

/// Next preset after the current threshold (wraps to most sensitive).
pub fn next_sensitivity(threshold: f32) -> f32 {
    SENSITIVITY_PRESETS
        .iter()
        .copied()
        .find(|p| *p > threshold + 1e-6)
        .unwrap_or(SENSITIVITY_PRESETS[0])
}

fn load_zones_from(path: &Path) -> HashMap<String, MotionZone> {
    let Ok(raw) = fs::read_to_string(path) else {
        return HashMap::new();
    };
    toml::from_str::<HashMap<String, MotionZone>>(&raw)
        .map(|m| m.into_iter().map(|(k, z)| (k, z.clamped())).collect())
        .unwrap_or_default()
}

fn save_zones_to(zones: &HashMap<String, MotionZone>, path: &Path) {
    let Ok(raw) = toml::to_string(zones) else {
        return;
    };
    if let Some(dir) = path.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let _ = fs::write(path, raw);
}

pub struct MotionDetector {
    prev_gray: Option<Vec<u8>>,
    threshold: f32,
    /// Cached analysis-grid mask for the current zone (rebuilt when the
    /// zone changes; polygons can have many vertices so rasterizing once
    /// per change beats re-testing 3600 cell centers every frame).
    mask_cache: Option<(MotionZone, Vec<u8>)>,
}

impl MotionDetector {
    pub fn new(threshold: f32) -> Self {
        Self {
            prev_gray: None,
            threshold,
            mask_cache: None,
        }
    }

    /// Feed a frame; returns the motion score (0.0..=1.0 mean absolute
    /// difference of downscaled grayscale) when enough history exists.
    /// Only analysis cells inside `zone` (rectangle or freeform polygon)
    /// are compared when a zone is set.
    pub fn score(&mut self, frame: &VideoFrame, zone: Option<&MotionZone>) -> Option<f32> {
        if frame.width == 0 || frame.height == 0 || frame.rgb.is_empty() {
            return None;
        }
        let gray = downscale_gray(&frame.rgb, frame.stride, frame.width, frame.height);
        let score = match (&self.prev_gray, zone) {
            (Some(prev), None) => Some(mean_abs_diff(prev, &gray)),
            (Some(prev), Some(zone)) => {
                let zone = zone.clone().clamped();
                if !self
                    .mask_cache
                    .as_ref()
                    .is_some_and(|(cached, _)| *cached == zone)
                {
                    self.mask_cache = Some((zone.clone(), build_zone_mask(&zone)));
                }
                let mask = &self.mask_cache.as_ref().unwrap().1;
                Some(masked_mean_abs_diff(prev, &gray, mask))
            }
            _ => None,
        };
        self.prev_gray = Some(gray);
        score
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.001, 1.0);
    }

    /// Forget history (e.g. after alerts were disabled).
    pub fn reset(&mut self) {
        self.prev_gray = None;
    }
}

/// Per-camera alert state: detection, cooldown, and border flash timing.
pub struct MotionAlertState {
    detector: MotionDetector,
    /// Per-camera enable (from config or the tile badge); `m` is a global gate.
    pub enabled: bool,
    flash_until: Instant,
    last_alert: Instant,
    last_score: f32,
}

impl MotionAlertState {
    pub fn new(enabled: bool, threshold: f32) -> Self {
        Self {
            detector: MotionDetector::new(threshold),
            enabled,
            flash_until: Instant::now(),
            last_alert: Instant::now() - ALERT_COOLDOWN,
            last_score: 0.0,
        }
    }

    /// Feed a frame. Returns true when an alert should be raised (motion
    /// over threshold inside the zone, cooldown elapsed, alerts armed).
    pub fn observe(
        &mut self,
        frame: &VideoFrame,
        global_armed: bool,
        zone: Option<&MotionZone>,
        now: Instant,
    ) -> bool {
        if !self.enabled || !global_armed {
            // Skip analysis entirely (CPU saver); drop history so re-arming
            // does not fire on a stale difference.
            self.detector.reset();
            return false;
        }
        let Some(score) = self.detector.score(frame, zone) else {
            return false;
        };
        self.last_score = score;
        if score >= self.detector.threshold() && now.duration_since(self.last_alert) >= ALERT_COOLDOWN
        {
            self.last_alert = now;
            self.flash_until = now + FLASH_DURATION;
            true
        } else {
            false
        }
    }

    /// True while the border should flash after a detection.
    pub fn is_flashing(&self, now: Instant) -> bool {
        now < self.flash_until
    }

    pub fn last_score(&self) -> f32 {
        self.last_score
    }

    pub fn threshold(&self) -> f32 {
        self.detector.threshold()
    }

    /// Change this camera's motion sensitivity at runtime.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.detector.set_threshold(threshold);
    }
}

/// Nearest-neighbour downscale to fixed-size BT.601-ish luma.
fn downscale_gray(rgb: &[u8], stride: usize, width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((ANALYSIS_W * ANALYSIS_H) as usize);
    for y in 0..ANALYSIS_H {
        let sy = (y * height / ANALYSIS_H).min(height - 1);
        for x in 0..ANALYSIS_W {
            let sx = (x * width / ANALYSIS_W).min(width - 1);
            let i = sy as usize * stride + sx as usize * 3;
            if i + 2 >= rgb.len() {
                out.push(0);
                continue;
            }
            let luma = (u32::from(rgb[i]) * 77
                + u32::from(rgb[i + 1]) * 150
                + u32::from(rgb[i + 2]) * 29)
                >> 8;
            out.push(luma.min(255) as u8);
        }
    }
    out
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let sum: u32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| u32::from(x.abs_diff(*y)))
        .sum();
    sum as f32 / (a.len() as f32 * 255.0)
}

/// Mean abs diff over the analysis-grid cells marked in `mask` only.
fn masked_mean_abs_diff(a: &[u8], b: &[u8], mask: &[u8]) -> f32 {
    if a.len() != b.len() || a.len() != mask.len() || a.is_empty() {
        return 0.0;
    }
    let mut sum = 0u32;
    let mut count = 0u32;
    for ((x, y), m) in a.iter().zip(b).zip(mask) {
        if *m != 0 {
            sum += u32::from(x.abs_diff(*y));
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    sum as f32 / (count as f32 * 255.0)
}

/// Rasterize a zone (freeform polygon or rectangle) onto the analysis grid:
/// 1 = the cell center is inside the outline.
fn build_zone_mask(zone: &MotionZone) -> Vec<u8> {
    let outline = zone.outline();
    let mut mask = vec![0u8; (ANALYSIS_W * ANALYSIS_H) as usize];
    for cy in 0..ANALYSIS_H {
        let ny = (cy as f32 + 0.5) / ANALYSIS_H as f32;
        for cx in 0..ANALYSIS_W {
            let nx = (cx as f32 + 0.5) / ANALYSIS_W as f32;
            if point_in_polygon(nx, ny, &outline) {
                mask[(cy * ANALYSIS_W + cx) as usize] = 1;
            }
        }
    }
    mask
}

/// Even-odd ray casting test against a closed polygon (last -> first edge
/// implied). Points on the boundary may go either way; irrelevant here.
pub fn point_in_polygon(px: f32, py: f32, polygon: &[[f32; 2]]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (polygon[i][0], polygon[i][1]);
        let (xj, yj) = (polygon[j][0], polygon[j][1]);
        if (yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Absolute enclosed area of a polygon (shoelace formula), in normalized
/// units squared.
pub fn polygon_area(polygon: &[[f32; 2]]) -> f32 {
    let n = polygon.len();
    if n < 3 {
        return 0.0;
    }
    let mut acc = 0.0f32;
    for i in 0..n {
        let (x0, y0) = (polygon[i][0], polygon[i][1]);
        let (x1, y1) = (polygon[(i + 1) % n][0], polygon[(i + 1) % n][1]);
        acc += x0 * y1 - x1 * y0;
    }
    (acc * 0.5).abs()
}

/// Ramer-Douglas-Peucker simplification of a freehand stroke, so hand
/// tremor does not turn into hundreds of polygon vertices.
pub fn simplify_outline(points: &[[f32; 2]], epsilon: f32) -> Vec<[f32; 2]> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    let mut stack = vec![(0usize, points.len() - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (ax, ay) = (points[a][0], points[a][1]);
        let (bx, by) = (points[b][0], points[b][1]);
        let seg_len = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
        let mut best_dist = 0.0f32;
        let mut best_idx = usize::MAX;
        for (i, p) in points.iter().enumerate().take(b).skip(a + 1) {
            let (px, py) = (p[0], p[1]);
            let dist = if seg_len < f32::EPSILON {
                ((px - ax).powi(2) + (py - ay).powi(2)).sqrt()
            } else {
                (((px - ax) * (by - ay) - (py - ay) * (bx - ax)).abs()) / seg_len
            };
            if dist > best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        if best_dist > epsilon {
            keep[best_idx] = true;
            stack.push((a, best_idx));
            stack.push((best_idx, b));
        }
    }
    points
        .iter()
        .zip(&keep)
        .filter(|(_, keep)| **keep)
        .map(|(p, _)| *p)
        .collect()
}

/// Ring the terminal bell (best-effort; shown by most terminals).
pub fn beep() {
    use std::io::Write as _;
    let mut err = std::io::stderr();
    let _ = err.write_all(b"\x07");
    let _ = err.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(w: u32, h: u32, rgb: [u8; 3]) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..w * h {
            data.extend_from_slice(&rgb);
        }
        VideoFrame {
            width: w,
            height: h,
            rgb: data,
            stride: w as usize * 3,
        }
    }

    /// Frame that is black everywhere except a rectangle of white at
    /// normalized (x, y, w, h).
    fn partial_flash(w: u32, h: u32, zx: f32, zy: f32, zw: f32, zh: f32) -> VideoFrame {
        let mut data = vec![0u8; (w * h * 3) as usize];
        let x0 = (zx * w as f32) as u32;
        let y0 = (zy * h as f32) as u32;
        let x1 = ((zx + zw) * w as f32) as u32;
        let y1 = ((zy + zh) * h as f32) as u32;
        for y in y0..y1.min(h) {
            for x in x0..x1.min(w) {
                let i = ((y * w + x) * 3) as usize;
                data[i] = 255;
                data[i + 1] = 255;
                data[i + 2] = 255;
            }
        }
        VideoFrame {
            width: w,
            height: h,
            rgb: data,
            stride: w as usize * 3,
        }
    }

    #[test]
    fn identical_frames_score_near_zero() {
        let mut det = MotionDetector::new(0.05);
        let f = solid_frame(320, 240, [120, 120, 120]);
        assert!(det.score(&f, None).is_none()); // first frame: no history
        let s = det.score(&f, None).unwrap();
        assert!(s < 0.01, "expected ~0, got {s}");
    }

    #[test]
    fn changed_frames_score_high() {
        let mut det = MotionDetector::new(0.05);
        let a = solid_frame(320, 240, [0, 0, 0]);
        let b = solid_frame(320, 240, [255, 255, 255]);
        assert!(det.score(&a, None).is_none());
        let s = det.score(&b, None).unwrap();
        assert!(s > 0.9, "expected high score, got {s}");
    }

    #[test]
    fn zone_ignores_motion_outside_it() {
        // Zone covers left half; flash happens on the right half only.
        let zone = MotionZone::rect(0.0, 0.0, 0.5, 1.0);
        let mut det = MotionDetector::new(0.05);
        let a = solid_frame(160, 90, [0, 0, 0]);
        let b = partial_flash(160, 90, 0.7, 0.2, 0.2, 0.5);
        assert!(det.score(&a, Some(&zone)).is_none());
        let s = det.score(&b, Some(&zone)).unwrap();
        assert!(s < 0.02, "motion outside zone should be ignored, got {s}");
    }

    #[test]
    fn zone_detects_motion_inside_it() {
        let zone = MotionZone::rect(0.0, 0.0, 0.5, 1.0);
        let mut det = MotionDetector::new(0.05);
        let a = solid_frame(160, 90, [0, 0, 0]);
        let b = partial_flash(160, 90, 0.1, 0.2, 0.3, 0.5);
        assert!(det.score(&a, Some(&zone)).is_none());
        let s = det.score(&b, Some(&zone)).unwrap();
        assert!(s > 0.1, "motion inside zone should score high, got {s}");
    }

    #[test]
    fn polygon_zone_ignores_motion_outside_it() {
        // Freeform zone: triangle over the left half of the frame.
        let zone = MotionZone {
            x: 0.0,
            y: 0.0,
            w: 0.5,
            h: 1.0,
            points: vec![[0.0, 0.0], [0.5, 0.0], [0.25, 1.0]],
        };
        let mut det = MotionDetector::new(0.05);
        let a = solid_frame(160, 90, [0, 0, 0]);
        // Flash in the top-right corner: outside the triangle.
        let b = partial_flash(160, 90, 0.7, 0.05, 0.2, 0.2);
        assert!(det.score(&a, Some(&zone)).is_none());
        let s = det.score(&b, Some(&zone)).unwrap();
        assert!(s < 0.02, "motion outside polygon should be ignored, got {s}");
    }

    #[test]
    fn polygon_zone_detects_motion_inside_it() {
        let zone = MotionZone {
            x: 0.0,
            y: 0.0,
            w: 0.5,
            h: 1.0,
            points: vec![[0.0, 0.0], [0.5, 0.0], [0.25, 1.0]],
        };
        let mut det = MotionDetector::new(0.05);
        let a = solid_frame(160, 90, [0, 0, 0]);
        // Flash near the triangle's top edge, well inside it.
        let b = partial_flash(160, 90, 0.1, 0.05, 0.3, 0.15);
        assert!(det.score(&a, Some(&zone)).is_none());
        let s = det.score(&b, Some(&zone)).unwrap();
        assert!(s > 0.05, "motion inside polygon should score high, got {s}");
    }

    #[test]
    fn point_in_polygon_triangle() {
        let tri = [[0.0f32, 0.0], [1.0, 0.0], [0.5, 1.0]];
        assert!(point_in_polygon(0.5, 0.1, &tri));
        assert!(!point_in_polygon(0.05, 0.9, &tri));
        assert!(!point_in_polygon(0.95, 0.9, &tri));
        // Degenerate input never contains anything.
        assert!(!point_in_polygon(0.5, 0.5, &[[0.0, 0.0], [1.0, 1.0]]));
    }

    #[test]
    fn polygon_area_of_unit_square() {
        let square = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!((polygon_area(&square) - 1.0).abs() < 1e-6);
        assert_eq!(polygon_area(&[[0.0, 0.0], [1.0, 1.0]]), 0.0);
    }

    #[test]
    fn simplify_outline_collapses_wobbly_line_to_endpoints() {
        // Dense samples along a straight line with tiny tremor.
        let line: Vec<[f32; 2]> = (0..=50)
            .map(|i| {
                let t = i as f32 / 50.0;
                [t, 0.5 + 0.001 * (i % 2) as f32]
            })
            .collect();
        let simple = simplify_outline(&line, OUTLINE_SIMPLIFY_EPSILON);
        assert!(simple.len() <= 2, "expected collapse, got {} points", simple.len());
    }

    #[test]
    fn simplify_outline_keeps_corner_vertices() {
        let corner = vec![[0.0f32, 0.0], [0.5, 0.0], [1.0, 0.001], [1.0, 0.5], [1.0, 1.0]];
        let simple = simplify_outline(&corner, OUTLINE_SIMPLIFY_EPSILON);
        assert!(simple.contains(&[0.5, 0.0]) || simple.len() >= 3);
        assert!(simple.len() >= 3, "corner must survive, got {simple:?}");
    }

    #[test]
    fn from_outline_builds_polygon_zone_with_bbox() {
        // A big lasso around the middle of the frame.
        let lasso = vec![[0.2, 0.2], [0.8, 0.25], [0.75, 0.8], [0.25, 0.75], [0.2, 0.2]];
        let z = MotionZone::from_outline(&lasso).unwrap();
        assert!(z.is_polygon());
        assert!(z.x <= 0.2 && z.y <= 0.2);
        assert!(z.x + z.w >= 0.75 && z.y + z.h >= 0.75);
        assert!(!z.points.is_empty());
    }

    #[test]
    fn from_outline_rejects_tiny_or_degenerate_shapes() {
        // Two points: a line has no interior.
        assert!(MotionZone::from_outline(&[[0.1, 0.1], [0.9, 0.1]]).is_none());
        // Triangle covering ~0.1% of the frame: below the area floor.
        assert!(MotionZone::from_outline(&[[0.0, 0.0], [0.05, 0.0], [0.025, 0.05]]).is_none());
    }

    #[test]
    fn alert_fires_once_per_cooldown() {
        let mut state = MotionAlertState::new(true, 0.05);
        let now = Instant::now();
        let dark = solid_frame(64, 48, [0, 0, 0]);
        let bright = solid_frame(64, 48, [255, 255, 255]);

        assert!(!state.observe(&dark, true, None, now));
        assert!(state.observe(&bright, true, None, now), "first motion alerts");
        assert!(
            !state.observe(&dark, true, None, now),
            "second change within cooldown is suppressed"
        );
    }

    #[test]
    fn disabled_camera_never_alerts() {
        let mut state = MotionAlertState::new(false, 0.05);
        let now = Instant::now();
        let a = solid_frame(64, 48, [0, 0, 0]);
        let b = solid_frame(64, 48, [255, 255, 255]);
        assert!(!state.observe(&a, true, None, now));
        assert!(!state.observe(&b, true, None, now));
    }

    #[test]
    fn global_disarm_suppresses_alerts() {
        let mut state = MotionAlertState::new(true, 0.05);
        let now = Instant::now();
        let a = solid_frame(64, 48, [0, 0, 0]);
        let b = solid_frame(64, 48, [255, 255, 255]);
        assert!(!state.observe(&a, false, None, now));
        assert!(!state.observe(&b, false, None, now));
    }

    #[test]
    fn downscale_is_fixed_size() {
        let f = solid_frame(1920, 1080, [7, 7, 7]);
        let g = downscale_gray(&f.rgb, f.stride, f.width, f.height);
        assert_eq!(g.len(), (ANALYSIS_W * ANALYSIS_H) as usize);
    }

    #[test]
    fn mean_abs_diff_bounds() {
        assert_eq!(mean_abs_diff(&[0, 0], &[0, 0]), 0.0);
        assert!((mean_abs_diff(&[0], &[255]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zone_from_corners_normalizes_and_rejects_tiny_drags() {
        let z = MotionZone::from_corners(0.8, 0.7, 0.2, 0.3).unwrap();
        assert_eq!(z.x, 0.2);
        assert_eq!(z.y, 0.3);
        assert!((z.w - 0.6).abs() < 1e-6);
        assert!((z.h - 0.4).abs() < 1e-6);
        assert!(MotionZone::from_corners(0.5, 0.5, 0.51, 0.51).is_none());
    }

    #[test]
    fn zone_roundtrip_persistence() {
        let path = std::env::temp_dir().join(format!("ezviz-zones-{}.toml", std::process::id()));
        let mut zones = HashMap::new();
        zones.insert("Driveway".to_string(), MotionZone::rect(0.1, 0.2, 0.5, 0.4));
        save_zones_to(&zones, &path);
        let loaded = load_zones_from(&path);
        assert_eq!(loaded.len(), 1);
        let z = &loaded["Driveway"];
        assert!((z.x - 0.1).abs() < 1e-5);
        assert!((z.h - 0.4).abs() < 1e-5);
        assert!(z.points.is_empty());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn polygon_zone_roundtrip_persistence() {
        let path =
            std::env::temp_dir().join(format!("ezviz-zones-poly-{}.toml", std::process::id()));
        let mut zones = HashMap::new();
        zones.insert(
            "Driveway".to_string(),
            MotionZone {
                x: 0.1,
                y: 0.2,
                w: 0.5,
                h: 0.4,
                points: vec![[0.1, 0.2], [0.6, 0.25], [0.55, 0.6]],
            },
        );
        save_zones_to(&zones, &path);
        let loaded = load_zones_from(&path);
        let z = &loaded["Driveway"];
        assert!(z.is_polygon());
        assert_eq!(z.points.len(), 3);
        assert!((z.points[1][0] - 0.6).abs() < 1e-5);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn legacy_zones_file_without_points_still_loads() {
        let path =
            std::env::temp_dir().join(format!("ezviz-zones-legacy-{}.toml", std::process::id()));
        fs::write(&path, "[Driveway]\nx = 0.1\ny = 0.2\nw = 0.5\nh = 0.4\n").unwrap();
        let loaded = load_zones_from(&path);
        let z = &loaded["Driveway"];
        assert!(!z.is_polygon());
        assert!((z.w - 0.5).abs() < 1e-5);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn thresholds_roundtrip_and_filter_bad_values() {
        let path =
            std::env::temp_dir().join(format!("ezviz-thresh-{}.toml", std::process::id()));
        let mut map = HashMap::new();
        map.insert("Pool".to_string(), 0.004f32);
        map.insert("Front".to_string(), 0.03f32);
        save_thresholds_to(&map, &path);
        let loaded = load_thresholds_from(&path);
        assert_eq!(loaded.len(), 2);
        assert!((loaded["Pool"] - 0.004).abs() < 1e-6);

        // Out-of-range values in a hand-edited file are dropped.
        fs::write(&path, "Pool = 0.004\nBad = 9.0\nAlsoBad = 0.0\n").unwrap();
        let loaded = load_thresholds_from(&path);
        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains_key("Pool"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn sensitivity_cycle_wraps_through_presets() {
        assert!((next_sensitivity(0.004) - 0.012).abs() < 1e-6);
        assert!((next_sensitivity(0.012) - 0.03).abs() < 1e-6);
        assert!((next_sensitivity(0.03) - 0.004).abs() < 1e-6);
        // Custom thresholds step to the next preset above them.
        assert!((next_sensitivity(0.02) - 0.03).abs() < 1e-6);
        assert!((next_sensitivity(0.5) - 0.004).abs() < 1e-6);
    }

    #[test]
    fn sensitivity_labels() {
        assert_eq!(sensitivity_label(0.004), "HIGH");
        assert_eq!(sensitivity_label(0.012), "MEDIUM");
        assert_eq!(sensitivity_label(0.03), "LOW");
        assert_eq!(sensitivity_label(0.07), "CUSTOM");
    }

    #[test]
    fn set_threshold_updates_alert_state() {
        let mut state = MotionAlertState::new(true, 0.05);
        state.set_threshold(0.004);
        assert!((state.threshold() - 0.004).abs() < 1e-6);
        // Clamped into the valid range.
        state.set_threshold(99.0);
        assert!((state.threshold() - 1.0).abs() < 1e-6);
    }
}
