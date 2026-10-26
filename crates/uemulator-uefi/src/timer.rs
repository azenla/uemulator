use core::time::Duration;

use uefi::proto::unsafe_protocol;
use uefi::{Status, boot};

#[repr(C)]
#[unsafe_protocol("26baccb3-6f42-11d4-bce7-0080c73c8881")]
struct TimerArch {
    register_handler: usize,
    set_timer_period: unsafe extern "efiapi" fn(*mut TimerArch, u64) -> Status,
    get_timer_period: usize,
    generate_interrupt: usize,
}

/// Changes the firmware's periodic tick, which is what makes its timer interrupt fire.
pub fn set_period(period: Duration) {
    let Ok(handle) = boot::get_handle_for_protocol::<TimerArch>() else {
        return;
    };
    let Ok(mut timer) = boot::open_protocol_exclusive::<TimerArch>(handle) else {
        return;
    };
    let ptr: *mut TimerArch = &mut *timer;
    let set = timer.set_timer_period;
    let _ = unsafe { set(ptr, (period.as_nanos() / 100) as u64) };
}
