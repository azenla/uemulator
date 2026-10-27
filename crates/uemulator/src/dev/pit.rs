#[cfg(target_os = "uefi")]
use crate::prelude::*;

/// Input clock of the 8254 (14.31818 MHz / 12, rounded). The guest calibrates the TSC against
/// this, so it must be derived from the same emulated nanosecond clock as the TSC.
const PIT_HZ: u128 = 1_193_182;
const NANOS_PER_SECOND: u128 = 1_000_000_000;

const CHANNEL0_PORT: u16 = 0x40;
const CONTROL_PORT: u16 = 0x43;
const CHANNEL_COUNT: usize = 3;
/// Channel 2 is the only one with a software-controlled gate (port 0x61 bit 0).
const CHANNEL_SPEAKER: usize = 2;

/// Control word fields: bits 7:6 channel, 5:4 access, 3:1 mode. Bit 0 (BCD) is ignored;
/// counts are always binary.
const CONTROL_CHANNEL_SHIFT: u8 = 6;
const CONTROL_ACCESS_SHIFT: u8 = 4;
const CONTROL_ACCESS_MASK: u8 = 3;
const CONTROL_MODE_SHIFT: u8 = 1;
const CONTROL_MODE_MASK: u8 = 7;
/// "Channel" value 3 in a control word selects the read-back command instead.
const CONTROL_READ_BACK: usize = 3;
/// Read-back command: bits 1..=3 select channels 0..=2. The count/status latch-inhibit bits
/// (5:4) are not honoured; the count is always latched.
const READ_BACK_CHANNEL0: u8 = 1 << 1;

/// Access field 0 is the counter latch command; it never becomes a channel's access mode.
const ACCESS_LATCH: u8 = 0;
const ACCESS_LOW: u8 = 1;
const ACCESS_HIGH: u8 = 2;
const ACCESS_LOW_HIGH: u8 = 3;

const MODE_TERMINAL_COUNT: u8 = 0;
const MODE_RATE_GENERATOR: u8 = 2;
const MODE_SQUARE_WAVE: u8 = 3;
const MODE_SOFTWARE_STROBE: u8 = 4;
/// Mode field values 6 and 7 are aliases of 2 and 3.
const MODE_RATE_GENERATOR_ALIAS: u8 = 6;
const MODE_SQUARE_WAVE_ALIAS: u8 = 7;

/// A reload value of 0 means 65536.
const FULL_COUNT: u32 = 0x10000;
const COUNT_MASK: u32 = 0xffff;

#[derive(Default, Clone, Copy)]
struct Channel {
    /// Programmed count; 0 is normalized to `FULL_COUNT` when loaded.
    reload: u32,
    mode: u8,
    access: u8,
    /// Low byte already written in a low/high access sequence; next write is the high byte.
    write_high_pending: bool,
    /// Next unlatched read returns the high byte.
    read_high_next: bool,
    write_low_byte: u8,
    /// Count captured by a latch command, consumed by the following read(s).
    latched_count: Option<u16>,
    latch_high_next: bool,
    gate: bool,
    running: bool,
    /// Emulated time at which the counter held `start_count`; the current count is derived from
    /// elapsed time rather than ticked.
    start_ns: u64,
    start_count: u32,
    /// A count has been written since the last control word; until then OUT reads as high.
    loaded: bool,
}

#[derive(Default)]
pub(crate) struct Pit {
    channels: [Channel; CHANNEL_COUNT],
    /// Emulated time of the next IRQ 0, if channel 0 is armed. Consumed by `tick`.
    next_irq: Option<u64>,
}

fn ns_to_ticks(ns: u64) -> u64 {
    (ns as u128 * PIT_HZ / NANOS_PER_SECOND) as u64
}

/// Rounds up so a deadline derived from a tick count is never earlier than that tick.
fn ticks_to_ns(ticks: u64) -> u64 {
    (ticks as u128 * NANOS_PER_SECOND).div_ceil(PIT_HZ) as u64
}

fn is_periodic(mode: u8) -> bool {
    matches!(
        mode,
        MODE_RATE_GENERATOR | MODE_SQUARE_WAVE | MODE_RATE_GENERATOR_ALIAS | MODE_SQUARE_WAVE_ALIAS
    )
}

impl Channel {
    /// Current counter value. Square wave is counted like a rate generator (r down to 1 once per
    /// period) rather than decrementing by two twice per period.
    fn count(&self, now: u64) -> u32 {
        if !self.running {
            return self.start_count;
        }
        let ticks = ns_to_ticks(now.saturating_sub(self.start_ns));
        if is_periodic(self.mode) {
            let reload = self.reload as u64;
            (reload - (ticks % reload)) as u32
        } else {
            (self.start_count as u64).wrapping_sub(ticks) as u32 & COUNT_MASK
        }
    }

    /// OUT pin level. Only channel 2's is observable (port 0x61 bit 5); modes 1, 4 and 5 are
    /// not modelled and read high.
    fn out(&self, now: u64) -> bool {
        if !self.loaded {
            return true;
        }
        let ticks = ns_to_ticks(now.saturating_sub(self.start_ns));
        match self.mode {
            MODE_TERMINAL_COUNT => self.running && ticks >= self.start_count as u64,
            MODE_RATE_GENERATOR | MODE_RATE_GENERATOR_ALIAS => {
                let reload = self.reload as u64;
                !(self.running && ticks % reload == reload - 1)
            }
            MODE_SQUARE_WAVE | MODE_SQUARE_WAVE_ALIAS => {
                let reload = self.reload as u64;
                !self.running || (ticks % reload) < reload.div_ceil(2)
            }
            _ => true,
        }
    }
}

impl Pit {
    pub fn new() -> Self {
        Pit::default()
    }

    /// Drives channel 2's gate from port 0x61 bit 0. Gate low freezes the count; gate high
    /// restarts periodic modes from the reload value and resumes mode 0 from the held count.
    pub fn set_gate2(&mut self, on: bool, now: u64) {
        let channel = &mut self.channels[CHANNEL_SPEAKER];
        if channel.gate == on {
            return;
        }
        channel.gate = on;
        if !channel.loaded {
            return;
        }
        if on {
            if is_periodic(channel.mode) {
                channel.start_count = channel.reload;
            }
            channel.start_ns = now;
            channel.running = true;
        } else {
            channel.start_count = channel.count(now);
            channel.running = false;
        }
    }

    pub fn out2(&self, now: u64) -> bool {
        self.channels[CHANNEL_SPEAKER].out(now)
    }

    /// Called once the full count has been written. Channels 0 and 1 have no software gate and
    /// start counting immediately; channel 2 waits for its gate.
    fn load(&mut self, index: usize, now: u64) {
        let channel = &mut self.channels[index];
        let count = if channel.reload == 0 {
            FULL_COUNT
        } else {
            channel.reload
        };
        channel.reload = count;
        channel.start_count = count;
        channel.start_ns = now;
        channel.loaded = true;
        channel.running = index != CHANNEL_SPEAKER || channel.gate;
        if index == 0 {
            // Mode 4 must be armed: Linux programs its one-shot clock event as mode 4
            // (control word 0x38) and the guest hangs if IRQ 0 never fires. Modes 1 and 5
            // wait for a gate trigger that is not modelled.
            self.next_irq = match channel.mode {
                MODE_TERMINAL_COUNT
                | MODE_RATE_GENERATOR
                | MODE_SQUARE_WAVE
                | MODE_SOFTWARE_STROBE
                | MODE_RATE_GENERATOR_ALIAS
                | MODE_SQUARE_WAVE_ALIAS => Some(now + ticks_to_ns(count as u64)),
                _ => None,
            };
        }
    }

    pub fn write(&mut self, port: u16, value: u8, now: u64) {
        if port == CONTROL_PORT {
            let selected = (value >> CONTROL_CHANNEL_SHIFT) as usize;
            if selected == CONTROL_READ_BACK {
                for index in 0..CHANNEL_COUNT {
                    if value & (READ_BACK_CHANNEL0 << index) != 0 {
                        let channel = &mut self.channels[index];
                        // Read-back does not overwrite a count that is still waiting to be read.
                        if channel.latched_count.is_none() {
                            channel.latched_count = Some(channel.count(now) as u16);
                            channel.latch_high_next = false;
                        }
                    }
                }
                return;
            }
            let access = (value >> CONTROL_ACCESS_SHIFT) & CONTROL_ACCESS_MASK;
            let channel = &mut self.channels[selected];
            if access == ACCESS_LATCH {
                channel.latched_count = Some(channel.count(now) as u16);
                channel.latch_high_next = false;
                return;
            }
            channel.access = access;
            channel.mode = (value >> CONTROL_MODE_SHIFT) & CONTROL_MODE_MASK;
            channel.write_high_pending = false;
            channel.read_high_next = false;
            channel.loaded = false;
            channel.running = false;
            if selected == 0 {
                self.next_irq = None;
            }
            return;
        }
        let index = (port - CHANNEL0_PORT) as usize;
        let channel = &mut self.channels[index];
        match channel.access {
            ACCESS_LOW => {
                channel.reload = value as u32;
                self.load(index, now);
            }
            ACCESS_HIGH => {
                channel.reload = (value as u32) << 8;
                self.load(index, now);
            }
            // Includes the power-on access value 0 (no control word written yet).
            _ => {
                if !channel.write_high_pending {
                    channel.write_low_byte = value;
                    channel.write_high_pending = true;
                } else {
                    channel.reload = channel.write_low_byte as u32 | (value as u32) << 8;
                    channel.write_high_pending = false;
                    self.load(index, now);
                }
            }
        }
    }

    pub fn read(&mut self, port: u16, now: u64) -> u8 {
        let index = (port - CHANNEL0_PORT) as usize;
        let channel = &mut self.channels[index];
        if let Some(latched) = channel.latched_count {
            let high = channel.access == ACCESS_HIGH
                || (channel.access == ACCESS_LOW_HIGH && channel.latch_high_next);
            if channel.access == ACCESS_LOW_HIGH && !channel.latch_high_next {
                channel.latch_high_next = true;
                return latched as u8;
            }
            channel.latched_count = None;
            channel.latch_high_next = false;
            return if high {
                (latched >> 8) as u8
            } else {
                latched as u8
            };
        }
        let count = channel.count(now) as u16;
        match channel.access {
            ACCESS_LOW => count as u8,
            ACCESS_HIGH => (count >> 8) as u8,
            _ => {
                channel.read_high_next = !channel.read_high_next;
                if channel.read_high_next {
                    count as u8
                } else {
                    (count >> 8) as u8
                }
            }
        }
    }

    pub fn state(&self, now: u64) -> String {
        let channel = &self.channels[0];
        format!(
            "pit0 mode {} reload {} next_irq {:?} now {now}",
            channel.mode, channel.reload, self.next_irq
        )
    }

    pub fn next_deadline(&self) -> Option<u64> {
        self.next_irq
    }

    /// Returns true if IRQ 0 should be raised. Periodic modes skip over periods that were
    /// missed while the guest was not being run, so a late tick yields one IRQ rather than a
    /// burst; one-shot modes disarm.
    pub fn tick(&mut self, now: u64) -> bool {
        let Some(deadline) = self.next_irq else {
            return false;
        };
        if now < deadline {
            return false;
        }
        let channel = &self.channels[0];
        let period = ticks_to_ns(channel.reload as u64).max(1);
        self.next_irq = if is_periodic(channel.mode) {
            let periods_elapsed = (now - deadline) / period + 1;
            Some(deadline + periods_elapsed * period)
        } else {
            None
        };
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHANNEL2_PORT: u16 = 0x42;

    #[test]
    fn ch2_mode0() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 0);
        pit.write(CONTROL_PORT, 0xb0, 0);
        pit.write(CHANNEL2_PORT, 0x10, 0);
        pit.write(CHANNEL2_PORT, 0x00, 0);
        assert!(!pit.out2(1000));
        assert!(pit.out2(ticks_to_ns(16) + 10));
    }

    #[test]
    fn ch0_periodic() {
        let mut pit = Pit::new();
        pit.write(CONTROL_PORT, 0x34, 0);
        pit.write(CHANNEL0_PORT, 0xa9, 0);
        pit.write(CHANNEL0_PORT, 0x04, 0);
        let deadline = pit.next_deadline().unwrap();
        assert!(deadline > 0);
        assert!(pit.tick(deadline));
        assert!(pit.next_deadline().unwrap() > deadline);
    }

    #[test]
    fn ch0_mode4_oneshot_fires() {
        let mut pit = Pit::new();
        pit.write(CONTROL_PORT, 0x38, 0);
        pit.write(CHANNEL0_PORT, 0x10, 0);
        pit.write(CHANNEL0_PORT, 0x00, 0);
        let deadline = pit.next_deadline().expect("mode 4 must schedule an irq");
        assert!(pit.tick(deadline));
        assert!(pit.next_deadline().is_none());
    }

    #[test]
    fn lohi_read_toggles() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 0);
        pit.write(CONTROL_PORT, 0xb0, 0);
        pit.write(CHANNEL2_PORT, 0xff, 0);
        pit.write(CHANNEL2_PORT, 0xff, 0);
        assert_eq!(pit.read(CHANNEL2_PORT, 10), 0xff);
        assert_eq!(pit.read(CHANNEL2_PORT, 10), 0xff);
    }
}
