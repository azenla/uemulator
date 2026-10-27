#[cfg(target_os = "uefi")]
use crate::prelude::*;

const MASTER_COMMAND_PORT: u16 = 0x20;
const MASTER_DATA_PORT: u16 = 0x21;
const SLAVE_COMMAND_PORT: u16 = 0xa0;
const SLAVE_DATA_PORT: u16 = 0xa1;
/// Edge/level control registers (one bit per IRQ). Stored and read back only;
/// every interrupt is treated as edge triggered regardless of ELCR.
const MASTER_ELCR_PORT: u16 = 0x4d0;

/// The slave's INT output is wired to master IR2; ICW3 is ignored and this wiring is fixed.
const CASCADE_IRQ: u8 = 2;
const IRQS_PER_PIC: u8 = 8;

/// Command-port write with this bit set is ICW1 and starts initialization.
const ICW1_INIT: u8 = 0x10;
/// ICW1: an ICW4 follows ICW3.
const ICW1_IC4: u8 = 0x01;
/// ICW2 carries the vector base in bits 7:3; the low bits are the IRQ number.
const ICW2_VECTOR_MASK: u8 = 0xf8;

/// Command-port write with bit 3 set and bit 4 clear is OCW3.
const OCW3_MARKER: u8 = 0x08;
/// OCW3: bit 0 (RIS) is valid only when this "read register command" bit is set.
const OCW3_READ_REGISTER: u8 = 0x02;
/// OCW3: selects ISR (set) or IRR (clear) for subsequent command-port reads.
const OCW3_READ_ISR: u8 = 0x01;

/// OCW2 is a command-port write with bits 4:3 clear; bits 7:5 select the EOI form.
const OCW2_OPERATION_SHIFT: u8 = 5;
const OCW2_EOI_NONSPECIFIC: u8 = 1;
const OCW2_EOI_SPECIFIC: u8 = 3;
const OCW2_EOI_NONSPECIFIC_ROTATE: u8 = 5;
const OCW2_EOI_SPECIFIC_ROTATE: u8 = 7;
/// OCW2 specific EOI: bits 2:0 name the IRQ to clear.
const OCW2_IRQ_MASK: u8 = 7;

#[derive(Default)]
struct Pic {
    /// Requested lines, set by `Pics::raise` and cleared on acknowledge (edge triggered).
    irr: u8,
    /// In-service lines, cleared by EOI. A set bit blocks that IRQ and every lower priority one.
    isr: u8,
    imr: u8,
    /// Vector of IRQ 0, from ICW2.
    base: u8,
    /// Which ICW the next data-port write is: 1 = ICW2, 2 = ICW3, 3 = ICW4,
    /// 0 = initialization done and the write is OCW1 (the mask).
    icw_step: u8,
    needs_icw4: bool,
    /// OCW3 selection for command-port reads: ISR when set, IRR otherwise.
    read_isr: bool,
}

#[derive(Default)]
pub(crate) struct Pics {
    master: Pic,
    slave: Pic,
    elcr: [u8; 2],
}

/// Highest-priority unmasked request of one 8259, or `None` if nothing is requested or a
/// higher-or-equal priority IRQ is already in service. Fixed priority 0..7: no rotation, and no
/// special fully nested mode, so the cascade IRQ blocks lower lines of the master while the slave
/// is in service.
///
/// `forced_irr` is ORed into the request lines (the master's IR2 comes from the slave).
fn highest_requested(pic: &Pic, forced_irr: u8) -> Option<u8> {
    let requested = (pic.irr | forced_irr) & !pic.imr;
    for irq in 0..IRQS_PER_PIC {
        if pic.isr & (1 << irq) != 0 {
            return None;
        }
        if requested & (1 << irq) != 0 {
            return Some(irq);
        }
    }
    None
}

impl Pic {
    fn command(&mut self, value: u8) {
        if value & ICW1_INIT != 0 {
            // Initialization also clears the mask, as on real hardware.
            self.irr = 0;
            self.isr = 0;
            self.imr = 0;
            self.needs_icw4 = value & ICW1_IC4 != 0;
            self.icw_step = 1;
            self.read_isr = false;
        } else if value & OCW3_MARKER != 0 {
            if value & OCW3_READ_REGISTER != 0 {
                self.read_isr = value & OCW3_READ_ISR != 0;
            }
        } else {
            match value >> OCW2_OPERATION_SHIFT {
                // The rotate variants are accepted but priorities are never rotated.
                OCW2_EOI_NONSPECIFIC | OCW2_EOI_NONSPECIFIC_ROTATE => {
                    if self.isr != 0 {
                        // Lowest bit number is the highest priority, i.e. the one being serviced.
                        self.isr &= !(1 << self.isr.trailing_zeros());
                    }
                }
                OCW2_EOI_SPECIFIC | OCW2_EOI_SPECIFIC_ROTATE => {
                    self.isr &= !(1 << (value & OCW2_IRQ_MASK))
                }
                _ => {}
            }
        }
    }

    /// ICW3 and ICW4 contents are ignored: the cascade wiring is fixed and the mode is always
    /// 8086 with manual EOI.
    fn data(&mut self, value: u8) {
        match self.icw_step {
            1 => {
                self.base = value & ICW2_VECTOR_MASK;
                self.icw_step = 2;
            }
            2 => self.icw_step = if self.needs_icw4 { 3 } else { 0 },
            3 => self.icw_step = 0,
            _ => self.imr = value,
        }
    }
}

impl Pics {
    pub fn state(&self) -> String {
        format!(
            "m(irr {:02x} isr {:02x} imr {:02x} base {:02x}) s(irr {:02x} isr {:02x} imr {:02x} base {:02x})",
            self.master.irr,
            self.master.isr,
            self.master.imr,
            self.master.base,
            self.slave.irr,
            self.slave.isr,
            self.slave.imr,
            self.slave.base
        )
    }

    /// Latches a rising edge on `irq` (0..15; 8..15 are the slave).
    pub fn raise(&mut self, irq: u8) {
        if irq < IRQS_PER_PIC {
            self.master.irr |= 1 << irq;
        } else {
            self.slave.irr |= 1 << (irq - IRQS_PER_PIC);
        }
    }

    /// Returns `(from_slave, irq_within_that_pic)` for the interrupt that would be delivered now.
    fn pick(&self) -> Option<(bool, u8)> {
        let slave_irq = highest_requested(&self.slave, 0);
        // The master has no IRR bit for the cascade line; it is asserted iff the slave has a
        // deliverable request.
        let cascade_request = if slave_irq.is_some() {
            1 << CASCADE_IRQ
        } else {
            0
        };
        match highest_requested(&self.master, cascade_request) {
            Some(CASCADE_IRQ) => slave_irq.map(|irq| (true, irq)),
            Some(irq) => Some((false, irq)),
            None => None,
        }
    }

    /// Polled by the CPU between instructions while IF=1, so the first check must stay cheap:
    /// it rejects the common case (nothing requested, or everything requested is masked) without
    /// running `pick`. It is only an early-out and must never reject what `pick` would accept.
    pub fn has_pending(&self) -> bool {
        ((self.master.irr & !self.master.imr) | (self.slave.irr & !self.slave.imr)) != 0
            && self.pick().is_some()
    }

    /// Interrupt acknowledge cycle: moves the request to in-service and returns the vector.
    /// A slave interrupt also marks the cascade IRQ in service on the master, so the guest
    /// issues two EOIs (slave, then master).
    pub fn ack(&mut self) -> Option<u8> {
        match self.pick()? {
            (true, irq) => {
                self.slave.irr &= !(1 << irq);
                self.slave.isr |= 1 << irq;
                self.master.isr |= 1 << CASCADE_IRQ;
                Some(self.slave.base + irq)
            }
            (false, irq) => {
                self.master.irr &= !(1 << irq);
                self.master.isr |= 1 << irq;
                Some(self.master.base + irq)
            }
        }
    }

    pub fn io_read(&mut self, port: u16) -> u8 {
        match port {
            MASTER_COMMAND_PORT | SLAVE_COMMAND_PORT => {
                let pic = if port == MASTER_COMMAND_PORT {
                    &self.master
                } else {
                    &self.slave
                };
                if pic.read_isr { pic.isr } else { pic.irr }
            }
            MASTER_DATA_PORT => self.master.imr,
            SLAVE_DATA_PORT => self.slave.imr,
            MASTER_ELCR_PORT => self.elcr[0],
            _ => self.elcr[1],
        }
    }

    pub fn io_write(&mut self, port: u16, value: u8) {
        match port {
            MASTER_COMMAND_PORT => self.master.command(value),
            SLAVE_COMMAND_PORT => self.slave.command(value),
            MASTER_DATA_PORT => self.master.data(value),
            SLAVE_DATA_PORT => self.slave.data(value),
            MASTER_ELCR_PORT => self.elcr[0] = value,
            _ => self.elcr[1] = value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOI_NONSPECIFIC: u8 = OCW2_EOI_NONSPECIFIC << OCW2_OPERATION_SHIFT;

    /// Programs both PICs the way Linux does: vectors 0x20 and 0x28, cascade, 8086 mode.
    fn init(pics: &mut Pics) {
        let master_icw3 = 1 << CASCADE_IRQ;
        let slave_icw3 = CASCADE_IRQ;
        for (command, data, base, icw3) in [
            (MASTER_COMMAND_PORT, MASTER_DATA_PORT, 0x20, master_icw3),
            (SLAVE_COMMAND_PORT, SLAVE_DATA_PORT, 0x28, slave_icw3),
        ] {
            pics.io_write(command, ICW1_INIT | ICW1_IC4);
            pics.io_write(data, base);
            pics.io_write(data, icw3);
            pics.io_write(data, 1);
            pics.io_write(data, 0);
        }
    }

    #[test]
    fn irq_flow() {
        let mut pics = Pics::default();
        init(&mut pics);
        pics.raise(4);
        assert_eq!(pics.ack(), Some(0x24));
        pics.raise(3);
        assert_eq!(pics.ack(), Some(0x23));
        pics.io_write(MASTER_COMMAND_PORT, EOI_NONSPECIFIC);
        pics.io_write(MASTER_COMMAND_PORT, EOI_NONSPECIFIC);
        pics.raise(9);
        assert_eq!(pics.ack(), Some(0x29));
        assert_eq!(pics.ack(), None);
        pics.io_write(SLAVE_COMMAND_PORT, EOI_NONSPECIFIC);
        pics.io_write(MASTER_COMMAND_PORT, EOI_NONSPECIFIC);
        pics.io_write(MASTER_DATA_PORT, 0x10);
        pics.raise(4);
        assert_eq!(pics.ack(), None);
    }
}
