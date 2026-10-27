use alloc::string::{String, ToString};
use alloc::vec::Vec;

pub const DEFAULT_KERNEL: &str = "\\vmlinuz";
pub const DEFAULT_INITRD: &str = "\\initramfs";
pub const DEFAULT_MEMORY_MIB: usize = 1024;
/// Read from the filesystem the application was loaded from.
pub const CONFIG_FILE: &str = "\\uemulator.cfg";

/// `initrd=none` boots without an initramfs.
const INITRD_NONE: &str = "none";
/// Everything after this word goes to the kernel command line.
const CMDLINE_SEPARATOR: &str = "--";
const COMMENT_PREFIX: char = '#';

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accel {
    /// Hardware virtualization when the CPU offers it, the emulator otherwise.
    Auto,
    /// Hardware virtualization only; an error where unavailable.
    Hardware,
    /// The interpreter, even where hardware virtualization is available.
    Emulator,
}

#[derive(Debug)]
pub struct Config {
    /// Path on the boot filesystem, with `\` separators.
    pub kernel: String,
    pub initrd: Option<String>,
    pub memory_mib: usize,
    pub accel: Accel,
    /// Words appended to the library's default kernel command line.
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
    /// without one), `mem=` (MiB) and `accel=auto|hv|emu`. Everything after a bare `--` is
    /// appended to the kernel command line. Later calls override earlier ones, so the config file
    /// can be overridden by the load options. Only whole lines starting with `#` are comments.
    pub fn apply(&mut self, text: &str) -> Result<(), String> {
        let mut rest = false;
        let words = text
            .lines()
            .filter(|line| !line.trim_start().starts_with(COMMENT_PREFIX))
            .flat_map(str::split_whitespace);
        for word in words {
            if rest {
                self.cmdline_extra.push(word.to_string());
            } else if word == CMDLINE_SEPARATOR {
                rest = true;
            } else if let Some((key, value)) = word.split_once('=') {
                match key {
                    "kernel" => self.kernel = value.to_string(),
                    "initrd" if value == INITRD_NONE => self.initrd = None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn applied(text: &str) -> Config {
        let mut config = Config::default();
        config.apply(text).unwrap();
        config
    }

    fn error_of(text: &str) -> String {
        Config::default().apply(text).unwrap_err()
    }

    #[test]
    fn defaults_boot_the_conventional_files_with_a_gigabyte() {
        let config = Config::default();
        assert_eq!(config.kernel, "\\vmlinuz");
        assert_eq!(config.initrd.as_deref(), Some("\\initramfs"));
        assert_eq!(config.memory_mib, 1024);
        assert_eq!(config.accel, Accel::Auto);
        assert!(config.cmdline_extra.is_empty());
        assert_eq!(CONFIG_FILE, "\\uemulator.cfg");
    }

    #[test]
    fn empty_text_changes_nothing() {
        let config = applied("");
        assert_eq!(config.kernel, DEFAULT_KERNEL);
        assert_eq!(config.memory_mib, DEFAULT_MEMORY_MIB);
        let config = applied("   \n\t\n");
        assert_eq!(config.kernel, DEFAULT_KERNEL);
    }

    #[test]
    fn kernel_and_initrd_paths_are_taken_verbatim() {
        let config = applied("kernel=\\boot\\bzImage initrd=\\boot\\rd.img");
        assert_eq!(config.kernel, "\\boot\\bzImage");
        assert_eq!(config.initrd.as_deref(), Some("\\boot\\rd.img"));
    }

    #[test]
    fn value_may_contain_equals_signs() {
        assert_eq!(applied("kernel=a=b").kernel, "a=b");
    }

    #[test]
    fn initrd_none_boots_without_one() {
        assert_eq!(applied("initrd=none").initrd, None);
    }

    #[test]
    fn initrd_none_is_case_sensitive() {
        assert_eq!(applied("initrd=None").initrd.as_deref(), Some("None"));
    }

    #[test]
    fn a_later_initrd_path_replaces_none() {
        assert_eq!(
            applied("initrd=none initrd=\\x").initrd.as_deref(),
            Some("\\x")
        );
    }

    #[test]
    fn memory_is_in_mib() {
        assert_eq!(applied("mem=512").memory_mib, 512);
        assert_eq!(applied("mem=0").memory_mib, 0);
    }

    #[test]
    fn memory_rejects_anything_but_a_number() {
        for bad in ["mem=", "mem=abc", "mem=-1", "mem=1.5", "mem=1G", "mem=0x10"] {
            let message = error_of(bad);
            assert!(message.starts_with("bad mem value"), "{bad}: {message}");
        }
    }

    #[test]
    fn memory_rejects_numbers_too_large_for_usize() {
        assert!(error_of("mem=99999999999999999999999").contains("bad mem value"));
    }

    #[test]
    fn accel_values_map_to_their_modes() {
        assert_eq!(applied("accel=auto").accel, Accel::Auto);
        assert_eq!(applied("accel=hv").accel, Accel::Hardware);
        assert_eq!(applied("accel=emu").accel, Accel::Emulator);
    }

    #[test]
    fn accel_rejects_other_values() {
        for bad in ["accel=", "accel=kvm", "accel=HV", "accel=hardware"] {
            let message = error_of(bad);
            assert!(message.starts_with("bad accel value"), "{bad}: {message}");
        }
    }

    #[test]
    fn unknown_option_is_an_error_naming_it() {
        let message = error_of("colour=red");
        assert!(message.contains("unknown option"));
        assert!(message.contains("colour"));
    }

    #[test]
    fn option_without_a_key_is_unknown() {
        assert!(error_of("=value").contains("unknown option"));
    }

    #[test]
    fn bare_word_before_the_separator_is_an_error() {
        let message = error_of("quiet");
        assert!(message.contains("unexpected word"));
        assert!(message.contains("quiet"));
    }

    #[test]
    fn words_after_the_separator_go_to_the_kernel_command_line() {
        let config = applied("mem=256 -- console=ttyS0 quiet root=/dev/ram0");
        assert_eq!(config.memory_mib, 256);
        assert_eq!(
            config.cmdline_extra,
            vec!["console=ttyS0", "quiet", "root=/dev/ram0"]
        );
    }

    #[test]
    fn options_after_the_separator_are_not_interpreted() {
        let config = applied("-- mem=1 kernel=x accel=hv bogus");
        assert_eq!(config.memory_mib, DEFAULT_MEMORY_MIB);
        assert_eq!(config.kernel, DEFAULT_KERNEL);
        assert_eq!(config.accel, Accel::Auto);
        assert_eq!(
            config.cmdline_extra,
            vec!["mem=1", "kernel=x", "accel=hv", "bogus"]
        );
    }

    #[test]
    fn a_second_separator_is_an_ordinary_command_line_word() {
        assert_eq!(applied("-- a -- b").cmdline_extra, vec!["a", "--", "b"]);
    }

    #[test]
    fn separator_with_nothing_after_it_adds_nothing() {
        assert!(applied("mem=64 --").cmdline_extra.is_empty());
    }

    #[test]
    fn command_line_continues_over_lines() {
        let config = applied("--\nconsole=ttyS0\nquiet\n");
        assert_eq!(config.cmdline_extra, vec!["console=ttyS0", "quiet"]);
    }

    #[test]
    fn options_on_separate_lines_all_apply() {
        let config = applied("kernel=\\k\ninitrd=none\nmem=128\naccel=emu\n");
        assert_eq!(config.kernel, "\\k");
        assert_eq!(config.initrd, None);
        assert_eq!(config.memory_mib, 128);
        assert_eq!(config.accel, Accel::Emulator);
    }

    #[test]
    fn lines_starting_with_a_hash_are_comments() {
        let config = applied("# mem=1\nmem=2\n   # accel=bogus\n\t#kernel=nope\n");
        assert_eq!(config.memory_mib, 2);
        assert_eq!(config.kernel, DEFAULT_KERNEL);
    }

    #[test]
    fn a_hash_in_the_middle_of_a_line_is_not_a_comment() {
        assert_eq!(applied("kernel=a#b").kernel, "a#b");
        assert!(error_of("mem=1 # note").contains("unexpected word"));
    }

    #[test]
    fn comment_lines_inside_the_command_line_are_skipped() {
        let config = applied("-- a\n# b\nc");
        assert_eq!(config.cmdline_extra, vec!["a", "c"]);
    }

    #[test]
    fn whitespace_of_any_kind_separates_words() {
        let config = applied("mem=8\t\tkernel=\\k   accel=emu");
        assert_eq!(config.memory_mib, 8);
        assert_eq!(config.kernel, "\\k");
        assert_eq!(config.accel, Accel::Emulator);
    }

    #[test]
    fn crlf_line_endings_are_fine() {
        let config = applied("mem=32\r\nkernel=\\k\r\n");
        assert_eq!(config.memory_mib, 32);
        assert_eq!(config.kernel, "\\k");
    }

    #[test]
    fn a_later_apply_overrides_an_earlier_one() {
        let mut config = Config::default();
        config
            .apply("mem=64 kernel=\\a initrd=none accel=hv")
            .unwrap();
        config
            .apply("mem=128 kernel=\\b initrd=\\c accel=emu")
            .unwrap();
        assert_eq!(config.memory_mib, 128);
        assert_eq!(config.kernel, "\\b");
        assert_eq!(config.initrd.as_deref(), Some("\\c"));
        assert_eq!(config.accel, Accel::Emulator);
    }

    #[test]
    fn an_apply_keeps_what_it_does_not_mention() {
        let mut config = Config::default();
        config.apply("mem=64").unwrap();
        config.apply("kernel=\\b").unwrap();
        assert_eq!(config.memory_mib, 64);
        assert_eq!(config.kernel, "\\b");
    }

    #[test]
    fn command_line_words_accumulate_over_applies() {
        let mut config = Config::default();
        config.apply("-- one").unwrap();
        config.apply("-- two three").unwrap();
        assert_eq!(config.cmdline_extra, vec!["one", "two", "three"]);
    }

    #[test]
    fn the_separator_state_does_not_carry_into_the_next_apply() {
        let mut config = Config::default();
        config.apply("--").unwrap();
        config.apply("mem=2").unwrap();
        assert_eq!(config.memory_mib, 2);
        assert!(config.cmdline_extra.is_empty());
    }

    #[test]
    fn an_error_stops_before_later_words_but_keeps_earlier_ones() {
        let mut config = Config::default();
        assert!(config.apply("mem=7 bogus=1 mem=9").is_err());
        assert_eq!(config.memory_mib, 7);
    }

    #[test]
    fn empty_value_sets_an_empty_kernel_path() {
        // Nothing rejects it here; reading the file then fails with a clear message.
        assert_eq!(applied("kernel=").kernel, "");
    }
}
