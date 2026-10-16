use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Last window size, persisted between runs in
/// `$XDG_CONFIG_HOME/ezviz-view/window.toml` (default `~/.config/...`).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct WindowState {
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
}

impl WindowState {
    fn path() -> PathBuf {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .unwrap_or_else(|| PathBuf::from("."));
        base.join("ezviz-view").join("window.toml")
    }

    /// Load saved size, or defaults (1280x720) when missing/unreadable.
    pub fn load() -> Self {
        Self::load_from(&Self::path())
    }

    pub fn load_from(path: &PathBuf) -> Self {
        let Ok(raw) = fs::read_to_string(path) else {
            return Self::default_size();
        };
        toml::from_str::<WindowState>(&raw)
            .map(|s| s.clamped())
            .unwrap_or_else(|_| Self::default_size())
    }

    /// Save the current size (no-op on zero/absurd dimensions).
    pub fn save(&self, width: u32, height: u32) {
        self.save_to(&Self::path(), width, height);
    }

    pub fn save_to(&self, path: &PathBuf, width: u32, height: u32) {
        let state = WindowState { width, height }.clamped();
        if state.width == 0 || state.height == 0 {
            return;
        }
        let Ok(raw) = toml::to_string(&state) else {
            return;
        };
        if let Some(dir) = path.parent() {
            let _ = fs::create_dir_all(dir);
        }
        let _ = fs::write(path, raw);
    }

    fn default_size() -> Self {
        Self {
            width: 1280,
            height: 720,
        }
    }

    fn clamped(self) -> Self {
        Self {
            width: self.width.clamp(320, 7680),
            height: self.height.clamp(200, 4320),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_in_temp_file() {
        let dir = std::env::temp_dir().join(format!("ezviz-ws-{}", std::process::id()));
        let path = dir.join("window.toml");
        let _ = fs::remove_file(&path);

        // Missing file -> defaults.
        assert_eq!(WindowState::load_from(&path), WindowState::default_size());

        WindowState::default().save_to(&path, 1920, 1080);
        assert_eq!(
            WindowState::load_from(&path),
            WindowState {
                width: 1920,
                height: 1080
            }
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn clamps_absurd_sizes() {
        let dir = std::env::temp_dir().join(format!("ezviz-ws2-{}", std::process::id()));
        let path = dir.join("window.toml");
        WindowState::default().save_to(&path, 10, 10);
        let loaded = WindowState::load_from(&path);
        assert_eq!(loaded.width, 320);
        assert_eq!(loaded.height, 200);

        WindowState::default().save_to(&path, 99999, 99999);
        let loaded = WindowState::load_from(&path);
        assert_eq!(loaded.width, 7680);
        assert_eq!(loaded.height, 4320);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_file_falls_back_to_defaults() {
        let dir = std::env::temp_dir().join(format!("ezviz-ws3-{}", std::process::id()));
        let path = dir.join("window.toml");
        fs::create_dir_all(&dir).unwrap();
        fs::write(&path, "not = valid toml [").unwrap();
        assert_eq!(WindowState::load_from(&path), WindowState::default_size());
        let _ = fs::remove_dir_all(&dir);
    }
}
