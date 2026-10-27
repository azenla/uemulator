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
    struct WinSize {
        rows: u16,
        cols: u16,
        xpix: u16,
        ypix: u16,
    }

    const STDOUT_FD: i32 = 1;

    unsafe extern "C" {
        fn ioctl(fd: i32, req: u64, ...) -> i32;
    }

    /// Request code of the `TIOCGWINSZ` ioctl; the value differs between Darwin and Linux.
    #[cfg(target_os = "macos")]
    const TIOCGWINSZ: u64 = 0x4008_7468;
    #[cfg(not(target_os = "macos"))]
    const TIOCGWINSZ: u64 = 0x5413;

    /// `(cols, rows)` of the terminal on stdout. A zero size counts as unknown.
    pub fn term_size() -> Option<(u16, u16)> {
        let mut size = WinSize::default();
        let result = unsafe { ioctl(STDOUT_FD, TIOCGWINSZ, &mut size as *mut WinSize) };
        (result == 0 && size.cols > 0 && size.rows > 0).then_some((size.cols, size.rows))
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
}

pub use imp::{stty, term_size};
