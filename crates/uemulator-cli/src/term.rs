#[cfg(unix)]
mod imp {
    use std::process::{Command, Stdio};

    /// Runs `stty` with our stdin inherited, so it acts on the terminal we read from, and returns
    /// its trimmed stdout. `None` if it could not run or failed, e.g. stdin is not a terminal.
    /// Shelling out avoids declaring termios by hand for every platform.
    pub fn stty(args: &[&str]) -> Option<String> {
        let out = Command::new("stty")
            .args(args)
            .stdin(Stdio::inherit())
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    /// `struct winsize` from <sys/ioctl.h>; the field order is fixed by the kernel ABI.
    #[repr(C)]
    #[derive(Default)]
    pub(super) struct WinSize {
        pub rows: u16,
        pub cols: u16,
        pub xpix: u16,
        pub ypix: u16,
    }

    impl WinSize {
        /// `(cols, rows)`. A zero in either dimension counts as unknown: terminals that do not
        /// know their size (serial lines, some CI pseudo-terminals) report 0x0 with success.
        pub fn dimensions(&self) -> Option<(u16, u16)> {
            (self.cols > 0 && self.rows > 0).then_some((self.cols, self.rows))
        }
    }

    const STDOUT_FD: i32 = 1;

    unsafe extern "C" {
        fn ioctl(fd: i32, req: u64, ...) -> i32;
    }

    /// Request code of the `TIOCGWINSZ` ioctl. It is `_IOR('t', 104, struct winsize)` on the BSD
    /// family (Darwin included) and the legacy constant 0x5413 on Linux. `None` on other unixes,
    /// where guessing would issue an unrelated ioctl.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub(super) const TIOCGWINSZ: Option<u64> = Some(0x5413);
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "dragonfly"
    ))]
    pub(super) const TIOCGWINSZ: Option<u64> = Some(0x4008_7468);
    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "dragonfly"
    )))]
    pub(super) const TIOCGWINSZ: Option<u64> = None;

    /// `(cols, rows)` of the terminal on stdout, or `None` if stdout is not a terminal that knows
    /// its size.
    pub fn term_size() -> Option<(u16, u16)> {
        let request = TIOCGWINSZ?;
        let mut size = WinSize::default();
        let result = unsafe { ioctl(STDOUT_FD, request, &mut size as *mut WinSize) };
        if result == 0 { size.dimensions() } else { None }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn winsize_matches_the_c_layout() {
            assert_eq!(std::mem::size_of::<WinSize>(), 8);
            assert_eq!(std::mem::align_of::<WinSize>(), 2);
            let size = WinSize {
                rows: 1,
                cols: 2,
                xpix: 3,
                ypix: 4,
            };
            let words: [u16; 4] = unsafe { std::mem::transmute(size) };
            assert_eq!(words, [1, 2, 3, 4], "rows, cols, xpixel, ypixel in order");
        }

        #[test]
        fn dimensions_are_cols_then_rows() {
            let size = WinSize {
                rows: 24,
                cols: 80,
                xpix: 0,
                ypix: 0,
            };
            assert_eq!(size.dimensions(), Some((80, 24)));
        }

        #[test]
        fn zero_columns_or_rows_mean_unknown() {
            let unknown = |rows, cols| {
                WinSize {
                    rows,
                    cols,
                    xpix: 640,
                    ypix: 480,
                }
                .dimensions()
            };
            assert_eq!(unknown(0, 0), None);
            assert_eq!(unknown(24, 0), None);
            assert_eq!(unknown(0, 80), None);
            assert_eq!(unknown(1, 1), Some((1, 1)));
        }

        #[test]
        fn dimensions_accept_the_largest_size() {
            let size = WinSize {
                rows: u16::MAX,
                cols: u16::MAX,
                ..Default::default()
            };
            assert_eq!(size.dimensions(), Some((u16::MAX, u16::MAX)));
        }

        #[test]
        fn default_winsize_is_unknown() {
            assert_eq!(WinSize::default().dimensions(), None);
        }

        #[cfg(any(target_os = "linux", target_os = "android"))]
        #[test]
        fn tiocgwinsz_is_the_linux_value() {
            assert_eq!(TIOCGWINSZ, Some(0x5413));
        }

        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "freebsd",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "dragonfly"
        ))]
        #[test]
        fn tiocgwinsz_is_the_bsd_value() {
            assert_eq!(TIOCGWINSZ, Some(0x4008_7468));
        }

        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "freebsd",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "dragonfly"
        ))]
        #[test]
        fn bsd_tiocgwinsz_is_ior_t_104_of_winsize() {
            // _IOR: the "out" direction bit, parameter size in bits 16..29, group character,
            // command number.
            const BSD_IOC_OUT: u64 = 0x4000_0000;
            let expected = BSD_IOC_OUT
                | ((std::mem::size_of::<WinSize>() as u64) << 16)
                | ((b't' as u64) << 8)
                | 104;
            assert_eq!(TIOCGWINSZ, Some(expected));
        }

        #[test]
        fn term_size_never_reports_a_zero_dimension() {
            // Whether stdout is a terminal depends on how the tests are run; only the contract
            // holds either way.
            if let Some((cols, rows)) = term_size() {
                assert!(cols > 0 && rows > 0);
            }
        }

        #[test]
        fn stty_with_an_invalid_argument_fails() {
            assert_eq!(stty(&["--uemulator-no-such-option"]), None);
        }
    }
}

/// No raw mode and no size forwarding off unix: stdin stays line-buffered and the guest keeps its
/// default console size.
#[cfg(not(unix))]
mod imp {
    pub fn stty(_args: &[&str]) -> Option<String> {
        None
    }

    pub fn term_size() -> Option<(u16, u16)> {
        None
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn stty_is_unavailable() {
            assert_eq!(stty(&["-g"]), None);
            assert_eq!(stty(&["raw", "-echo"]), None);
        }

        #[test]
        fn term_size_is_unavailable() {
            assert_eq!(term_size(), None);
        }
    }
}

pub use imp::{stty, term_size};
