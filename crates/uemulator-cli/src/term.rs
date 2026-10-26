use std::process::{Command, Stdio};

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

#[repr(C)]
#[derive(Default)]
struct WinSize {
    rows: u16,
    cols: u16,
    xpix: u16,
    ypix: u16,
}

unsafe extern "C" {
    fn ioctl(fd: i32, req: u64, ...) -> i32;
}

#[cfg(target_os = "macos")]
const TIOCGWINSZ: u64 = 0x4008_7468;
#[cfg(not(target_os = "macos"))]
const TIOCGWINSZ: u64 = 0x5413;

pub fn term_size() -> Option<(u16, u16)> {
    let mut ws = WinSize::default();
    let r = unsafe { ioctl(1, TIOCGWINSZ, &mut ws as *mut WinSize) };
    (r == 0 && ws.cols > 0 && ws.rows > 0).then_some((ws.cols, ws.rows))
}
