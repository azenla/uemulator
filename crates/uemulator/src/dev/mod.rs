#[cfg(target_os = "uefi")]
use crate::prelude::*;
pub(crate) mod guestmem;
mod pci;
mod pic;
mod pit;
mod serial;
#[cfg(not(target_os = "uefi"))]
mod vblk;
mod vcons;
mod virtio;
mod vnet;

use crate::console::ConsoleHandle;
#[cfg(not(target_os = "uefi"))]
use crate::disk::Disk;
use crate::netdev::NetHandle;
use guestmem::GuestMem;
use pci::PciHost;
use pic::Pics;
use pit::Pit;
use serial::Uart;
#[cfg(not(target_os = "uefi"))]
use vblk::VirtioBlk;
use vcons::VirtioConsole;
use virtio::Transport;
use vnet::VirtioNet;

/// PIC wiring: IRQ lines used by the legacy devices we model.
const IRQ_PIT: u8 = 0;
const IRQ_UART: u8 = 4;

/// I/O ports. Anything not listed reads as all ones and ignores writes.
const PIC_MASTER_COMMAND: u16 = 0x20;
const PIC_MASTER_DATA: u16 = 0x21;
const PIC_SLAVE_COMMAND: u16 = 0xa0;
const PIC_SLAVE_DATA: u16 = 0xa1;
/// Edge/level control registers of the two PICs.
const PIC_ELCR_MASTER: u16 = 0x4d0;
const PIC_ELCR_SLAVE: u16 = 0x4d1;
const PIT_CHANNEL0: u16 = 0x40;
const PIT_CHANNEL2: u16 = 0x42;
const PIT_COMMAND: u16 = 0x43;
const SYSTEM_CONTROL_PORT_B: u16 = 0x61;
const CMOS_INDEX_PORT: u16 = 0x70;
const CMOS_DATA_PORT: u16 = 0x71;
/// i8042 command port; only the reset command is emulated.
const KBC_COMMAND_PORT: u16 = 0x64;
const FAST_A20_PORT: u16 = 0x92;
const RESET_CONTROL_PORT: u16 = 0xcf9;
/// PCI config mechanism #1 address (0xCF8..=0xCFB) and data (0xCFC..=0xCFF) windows. 0xCF9 sits
/// inside the address window but is the reset control register, so its match arm must come first.
const PCI_CONFIG_PORT_FIRST: u16 = 0xcf8;
const PCI_CONFIG_PORT_LAST: u16 = 0xcff;
const UART_COM1_BASE: u16 = 0x3f8;
const UART_COM1_LAST: u16 = 0x3ff;

/// i8042 command 0xFE pulses the CPU reset line.
const KBC_COMMAND_PULSE_RESET: u8 = 0xfe;
/// Port 0x92: bit 1 is the A20 gate (stored, not acted on), bit 0 resets the CPU.
const FAST_A20_GATE: u8 = 1 << 1;
const FAST_A20_RESET: u8 = 1 << 0;
/// Port 0xCF9 bit 2 (RST_CPU) triggers the reset; bit 1 selects a hard (1) or soft (0) reset
/// and is the only bit that reads back, as in QEMU's PIIX. Both kinds reset the machine here.
const RESET_CONTROL_RESET_CPU: u8 = 1 << 2;
const RESET_CONTROL_SYSTEM_RESET: u8 = 1 << 1;

/// Port 0x61 bits. Only the low two are writable (timer 2 gate, speaker data enable); the
/// others are synthesized on read.
const PORT_B_WRITABLE_MASK: u8 = 0x03;
const PORT_B_TIMER2_GATE: u8 = 1 << 0;
const PORT_B_REFRESH_TOGGLE: u8 = 1 << 4;
const PORT_B_TIMER2_OUT: u8 = 1 << 5;
/// The refresh toggle flips every 15.085 us (PIT channel 1 divisor 18 at 1.193182 MHz), so
/// guests busy-waiting on it see time pass. Derived from virtual time to stay deterministic.
const REFRESH_TOGGLE_PERIOD_NS: u64 = 15085;

const NS_PER_SECOND: u64 = 1_000_000_000;
const SECONDS_PER_MINUTE: u64 = 60;
const SECONDS_PER_HOUR: u64 = 3600;
const SECONDS_PER_DAY: u64 = 86400;
/// 1970-01-01 was a Thursday; CMOS day-of-week counts Sunday as 1.
const EPOCH_DAYS_SINCE_SUNDAY: u64 = 4;
const DAYS_PER_WEEK: u64 = 7;

/// CMOS/RTC register indices. Alarm registers (1, 3, 5) and unlisted ones are plain RAM.
const CMOS_SIZE: usize = 128;
/// Bit 7 of the index port is the NMI mask, not part of the index.
const CMOS_INDEX_MASK: u8 = 0x7f;
const CMOS_SECONDS: usize = 0x00;
const CMOS_MINUTES: usize = 0x02;
const CMOS_HOURS: usize = 0x04;
const CMOS_DAY_OF_WEEK: usize = 0x06;
const CMOS_DAY_OF_MONTH: usize = 0x07;
const CMOS_MONTH: usize = 0x08;
const CMOS_YEAR: usize = 0x09;
const CMOS_STATUS_A: usize = 0x0a;
const CMOS_STATUS_B: usize = 0x0b;
const CMOS_STATUS_C: usize = 0x0c;
const CMOS_STATUS_D: usize = 0x0d;
const CMOS_CENTURY: usize = 0x32;
/// Status A: update-in-progress clear, 32.768 kHz time base, 1024 Hz periodic rate.
const CMOS_STATUS_A_VALUE: u8 = 0x26;
/// Status B: 24-hour mode, BCD, no interrupts enabled.
const CMOS_STATUS_B_VALUE: u8 = 0x02;
/// Status D: battery good (VRT), so the kernel trusts the time.
const CMOS_STATUS_D_VALUE: u8 = 0x80;

/// What to attach to the PCI bus; every device is optional.
#[derive(Default)]
pub(crate) struct DevConfig {
    #[cfg(not(target_os = "uefi"))]
    pub disk: Option<Disk>,
    /// MAC address and the host side of the frame channel.
    pub net: Option<([u8; 6], NetHandle)>,
    pub virtio_console: bool,
}

/// The whole device bus: legacy port-I/O devices plus the virtio-pci host. Port I/O goes
/// through `io_in`/`io_out`, physical accesses above RAM through `mmio_read`/`mmio_write`.
/// Devices raise IRQs directly on `pic`; the CPU polls `pic.has_pending()`.
pub(crate) struct Devices {
    pub pic: Pics,
    pit: Pit,
    uart: Uart,
    cmos_index: u8,
    cmos: [u8; CMOS_SIZE],
    boot_unix: u64,
    system_control_port_b: u8,
    system_control_port_a: u8,
    reset_control: u8,
    pci: PciHost,
    /// Set by the reset ports (i8042, 0x92, 0xCF9); the CPU turns it into `ExitReason::Reset`.
    pub reset_requested: bool,
}

/// Mask of the low `size` bytes of a port value; sizes of 4 or more cover the whole dword.
fn size_mask(size: u8) -> u32 {
    match size {
        0 => 0,
        1..=3 => (1u32 << (8 * size as u32)) - 1,
        _ => u32::MAX,
    }
}

fn bcd(value: u64) -> u8 {
    (((value / 10) << 4) | (value % 10)) as u8
}

/// Days since 1970-01-01 to (year, month, day), proleptic Gregorian. Howard Hinnant's
/// civil_from_days: shifts the epoch to 0000-03-01 so the leap day is the last day of the
/// year, then works in 400-year eras.
fn civil(days_since_unix_epoch: i64) -> (i64, u64, u64) {
    const DAYS_FROM_0000_03_01_TO_UNIX_EPOCH: i64 = 719468;
    const DAYS_PER_ERA: i64 = 146097;
    const DAYS_PER_4_YEARS: i64 = 1460;
    const DAYS_PER_100_YEARS: i64 = 36524;
    const DAYS_PER_YEAR: i64 = 365;
    const YEARS_PER_ERA: i64 = 400;
    let shifted = days_since_unix_epoch + DAYS_FROM_0000_03_01_TO_UNIX_EPOCH;
    let era = shifted.div_euclid(DAYS_PER_ERA);
    let day_of_era = shifted.rem_euclid(DAYS_PER_ERA);
    let year_of_era = (day_of_era - day_of_era / DAYS_PER_4_YEARS
        + day_of_era / DAYS_PER_100_YEARS
        - day_of_era / (DAYS_PER_ERA - 1))
        / DAYS_PER_YEAR;
    let day_of_year =
        day_of_era - (DAYS_PER_YEAR * year_of_era + year_of_era / 4 - year_of_era / 100);
    // Month counted from March (0 = March); 153 is the day count of five consecutive months.
    let month_from_march = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_from_march + 2) / 5 + 1) as u64;
    let month = if month_from_march < 10 {
        month_from_march + 3
    } else {
        month_from_march - 9
    } as u64;
    let year = year_of_era + era * YEARS_PER_ERA + (month <= 2) as i64;
    (year, month, day)
}

impl Devices {
    pub fn new(console: ConsoleHandle, cfg: DevConfig) -> Self {
        // Order decides PCI slot and legacy IRQ: console, disk, net.
        let mut virtio_devices = Vec::new();
        if cfg.virtio_console {
            virtio_devices.push(Transport::new(Box::new(VirtioConsole::new(
                console.clone(),
            ))));
        }
        #[cfg(not(target_os = "uefi"))]
        if let Some(disk) = cfg.disk {
            virtio_devices.push(Transport::new(Box::new(VirtioBlk::new(disk))));
        }
        if let Some((mac, handle)) = cfg.net {
            virtio_devices.push(Transport::new(Box::new(VirtioNet::new(mac, handle))));
        }
        #[cfg(not(target_os = "uefi"))]
        let boot_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        #[cfg(target_os = "uefi")]
        let boot_unix = 0;
        Devices {
            pic: Pics::default(),
            pit: Pit::new(),
            uart: Uart::new(console),
            cmos_index: 0,
            cmos: [0; CMOS_SIZE],
            boot_unix,
            system_control_port_b: 0,
            system_control_port_a: 0,
            reset_control: 0,
            pci: PciHost::new(virtio_devices),
            reset_requested: false,
        }
    }

    pub fn state(&self, now: u64) -> String {
        format!("{} | {}", self.pic.state(), self.pit.state(now))
    }

    /// Pins the RTC to `unix` seconds at virtual time 0 (`--time`); guests must not depend on it.
    pub fn set_boot_time(&mut self, unix: u64) {
        self.boot_unix = unix;
    }

    /// Only the PIT schedules deadlines. The UART and virtio devices are polled from `tick`
    /// instead, which the CPU calls every `EVENT_INTERVAL` instructions and while idle, so
    /// host input and received frames wait at most one such interval.
    pub fn next_deadline(&self, _now: u64) -> Option<u64> {
        self.pit.next_deadline()
    }

    /// The devices pulse `Pics::raise` on an interrupt event; a line the guest configured as
    /// level triggered (ELCR) also has to be released when the device stops asserting it, or it
    /// would re-request forever. Edge-triggered lines ignore `lower`.
    fn update_uart_irq(&mut self) {
        if self.uart.irq_edge() {
            self.pic.raise(IRQ_UART);
        }
        if !self.uart.irq_level() {
            self.pic.lower(IRQ_UART);
        }
    }

    fn update_pci_irqs(&mut self) {
        for irq in PciHost::interrupt_lines() {
            if !self.pci.line_asserted(irq) {
                self.pic.lower(irq);
            }
        }
    }

    /// `mem` collects code pages the devices wrote by DMA; the caller must pass them to
    /// `Cpu::code_written` afterwards (see `GuestMem`).
    pub fn tick(&mut self, now: u64, mem: &mut GuestMem) {
        if self.pit.tick(now) {
            self.pic.raise(IRQ_PIT);
        }
        let mut raise = Vec::new();
        self.pci.poll(mem, &mut raise);
        for irq in raise {
            self.pic.raise(irq);
        }
        self.update_pci_irqs();
        // Once the virtio console driver is up, host input belongs to hvc0 and the UART must not also take it.
        self.uart.set_rx_enabled(!self.pci.console_active());
        self.uart.poll();
        self.update_uart_irq();
    }

    /// None means no device decodes `phys_addr` and the caller treats it as unmapped.
    pub fn mmio_read(&mut self, phys_addr: u64, size: u8) -> Option<u64> {
        let value = self.pci.mmio_read(phys_addr, size);
        // Reading the ISR register acknowledges the interrupt.
        self.update_pci_irqs();
        value.map(|value| value as u64)
    }

    /// Returns false if no device decodes `phys_addr`.
    pub fn mmio_write(&mut self, phys_addr: u64, size: u8, val: u64, mem: &mut GuestMem) -> bool {
        let result = self.pci.mmio_write(phys_addr, size, val as u32, mem);
        match result {
            Some(irq) => {
                // IRQ 0 is the PIT and never a PCI line: it means "no interrupt".
                if irq != 0 {
                    self.pic.raise(irq);
                }
                self.update_pci_irqs();
                true
            }
            None => false,
        }
    }

    /// `now` is virtual nanoseconds. Time registers are computed on read from `boot_unix`
    /// plus elapsed virtual time; writes to them are stored but never read back.
    fn cmos_read(&self, now: u64) -> u8 {
        let index = (self.cmos_index & CMOS_INDEX_MASK) as usize;
        let unix_seconds = self.boot_unix.saturating_add(now / NS_PER_SECOND);
        let days = unix_seconds / SECONDS_PER_DAY;
        let (year, month, day) = civil(days as i64);
        let seconds_of_day = unix_seconds % SECONDS_PER_DAY;
        let day_of_week = (days + EPOCH_DAYS_SINCE_SUNDAY) % DAYS_PER_WEEK + 1;
        match index {
            CMOS_SECONDS => bcd(seconds_of_day % SECONDS_PER_MINUTE),
            CMOS_MINUTES => bcd((seconds_of_day / SECONDS_PER_MINUTE) % SECONDS_PER_MINUTE),
            CMOS_HOURS => bcd(seconds_of_day / SECONDS_PER_HOUR),
            CMOS_DAY_OF_WEEK => day_of_week as u8,
            CMOS_DAY_OF_MONTH => bcd(day),
            CMOS_MONTH => bcd(month),
            CMOS_YEAR => bcd(year as u64 % 100),
            CMOS_CENTURY => bcd(year as u64 / 100),
            CMOS_STATUS_A => CMOS_STATUS_A_VALUE,
            CMOS_STATUS_B => CMOS_STATUS_B_VALUE,
            CMOS_STATUS_C => 0,
            CMOS_STATUS_D => CMOS_STATUS_D_VALUE,
            _ => self.cmos[index],
        }
    }

    pub fn io_in(&mut self, port: u16, size: u8, now: u64) -> u32 {
        let ones = size_mask(size);
        let value: u8 = match port {
            PIC_MASTER_COMMAND | PIC_MASTER_DATA | PIC_SLAVE_COMMAND | PIC_SLAVE_DATA
            | PIC_ELCR_MASTER | PIC_ELCR_SLAVE => self.pic.io_read(port),
            PIT_CHANNEL0..=PIT_CHANNEL2 => self.pit.read(port, now),
            SYSTEM_CONTROL_PORT_B => {
                let mut value = self.system_control_port_b & PORT_B_WRITABLE_MASK;
                if (now / REFRESH_TOGGLE_PERIOD_NS) & 1 != 0 {
                    value |= PORT_B_REFRESH_TOGGLE;
                }
                if self.pit.out2(now) {
                    value |= PORT_B_TIMER2_OUT;
                }
                value
            }
            CMOS_DATA_PORT => self.cmos_read(now),
            FAST_A20_PORT => self.system_control_port_a,
            RESET_CONTROL_PORT => self.reset_control,
            UART_COM1_BASE..=UART_COM1_LAST => self.uart.read(port - UART_COM1_BASE),
            PCI_CONFIG_PORT_FIRST..=PCI_CONFIG_PORT_LAST if !self.pci.is_empty() => {
                return self.pci.io_in(port, size);
            }
            _ => return ones,
        };
        if matches!(port, UART_COM1_BASE..=UART_COM1_LAST) {
            self.update_uart_irq();
        }
        value as u32 & ones
    }

    /// Port devices are byte-wide: only the low byte of `value` is used except for PCI config.
    pub fn io_out(&mut self, port: u16, size: u8, value: u32, now: u64) {
        let byte = value as u8;
        match port {
            PIC_MASTER_COMMAND | PIC_MASTER_DATA | PIC_SLAVE_COMMAND | PIC_SLAVE_DATA
            | PIC_ELCR_MASTER | PIC_ELCR_SLAVE => self.pic.io_write(port, byte),
            PIT_CHANNEL0..=PIT_COMMAND => self.pit.write(port, byte, now),
            SYSTEM_CONTROL_PORT_B => {
                self.system_control_port_b = byte & PORT_B_WRITABLE_MASK;
                self.pit.set_gate2(byte & PORT_B_TIMER2_GATE != 0, now);
            }
            CMOS_INDEX_PORT => self.cmos_index = byte,
            CMOS_DATA_PORT => self.cmos[(self.cmos_index & CMOS_INDEX_MASK) as usize] = byte,
            KBC_COMMAND_PORT if byte == KBC_COMMAND_PULSE_RESET => self.reset_requested = true,
            FAST_A20_PORT => {
                self.system_control_port_a = byte & FAST_A20_GATE;
                if byte & FAST_A20_RESET != 0 {
                    self.reset_requested = true;
                }
            }
            RESET_CONTROL_PORT => {
                if byte & RESET_CONTROL_RESET_CPU != 0 {
                    self.reset_requested = true;
                } else {
                    self.reset_control = byte & RESET_CONTROL_SYSTEM_RESET;
                }
            }
            PCI_CONFIG_PORT_FIRST..=PCI_CONFIG_PORT_LAST if !self.pci.is_empty() => {
                self.pci.io_out(port, size, value)
            }
            UART_COM1_BASE..=UART_COM1_LAST => {
                self.uart.write(port - UART_COM1_BASE, byte);
                self.update_uart_irq();
            }
            _ => {}
        }
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
