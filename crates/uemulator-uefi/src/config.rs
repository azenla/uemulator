use alloc::string::{String, ToString};
use alloc::vec::Vec;

pub const DEFAULT_KERNEL: &str = "\\vmlinuz";
pub const DEFAULT_INITRD: &str = "\\initramfs";
pub const DEFAULT_MEMORY_MIB: usize = 1024;
pub const CONFIG_FILE: &str = "\\uemulator.cfg";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accel {
    /// Hardware virtualization when the CPU offers it, the emulator otherwise.
    Auto,
    Hardware,
    Emulator,
}

#[derive(Debug)]
pub struct Config {
    pub kernel: String,
    pub initrd: Option<String>,
    pub memory_mib: usize,
    pub accel: Accel,
    pub cmdline_extra: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            kernel: DEFAULT_KERNEL.to_string(),
            initrd: Some(DEFAULT_INITRD.to_string()),
            memory_mib: DEFAULT_MEMORY_MIB,
            accel: Accel::Auto,
            cmdline_extra: Vec::new(),
        }
    }
}

impl Config {
    /// Applies whitespace-separated `key=value` words: `kernel=`, `initrd=` (`none` boots
    /// without one), `mem=` (MiB) and `accel=auto|hv|emu`. Everything after a bare `--` is appended to
    /// the kernel command line. Later calls override earlier ones, so the config file can be
    /// overridden by the load options.
    pub fn apply(&mut self, text: &str) -> Result<(), String> {
        let mut rest = false;
        let words = text
            .lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .flat_map(str::split_whitespace);
        for word in words {
            if rest {
                self.cmdline_extra.push(word.to_string());
            } else if word == "--" {
                rest = true;
            } else if let Some((key, value)) = word.split_once('=') {
                match key {
                    "kernel" => self.kernel = value.to_string(),
                    "initrd" if value == "none" => self.initrd = None,
                    "initrd" => self.initrd = Some(value.to_string()),
                    "accel" => {
                        self.accel = match value {
                            "auto" => Accel::Auto,
                            "hv" => Accel::Hardware,
                            "emu" => Accel::Emulator,
                            _ => return Err(alloc::format!("bad accel value {value:?}")),
                        }
                    }
                    "mem" => {
                        self.memory_mib = value
                            .parse()
                            .map_err(|_| alloc::format!("bad mem value {value:?}"))?
                    }
                    _ => return Err(alloc::format!("unknown option {key:?}")),
                }
            } else {
                return Err(alloc::format!("unexpected word {word:?}"));
            }
        }
        Ok(())
    }
}
