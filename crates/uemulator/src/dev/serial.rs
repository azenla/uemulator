use alloc::collections::VecDeque;

use crate::console::ConsoleHandle;

/// Register offsets from the UART base port (COM1 0x3f8). With DLAB set in LCR, offsets 0 and 1
/// address the divisor latch instead of RBR/THR and IER.
const REG_DATA: u16 = 0; // RBR on read, THR on write, DLL with DLAB
const REG_IER: u16 = 1; // DLM with DLAB
const REG_IIR_FCR: u16 = 2; // IIR on read, FCR on write
const REG_LCR: u16 = 3;
const REG_MCR: u16 = 4;
const REG_LSR: u16 = 5;
const REG_MSR: u16 = 6;
const REG_SCRATCH: u16 = 7;

const LCR_DLAB: u8 = 0x80;

/// Only the four defined IER bits are kept; the rest read back as 0.
const IER_MASK: u8 = 0x0f;
const IER_RX_AVAILABLE: u8 = 0x01;
const IER_THR_EMPTY: u8 = 0x02;

/// FCR bits that are stored: FIFO enable (0), DMA mode (3) and receive trigger level (7:6).
/// The two self-clearing FIFO reset bits are not kept.
const FCR_STORED_MASK: u8 = 0xc9;
const FCR_FIFO_ENABLE: u8 = 0x01;
const FCR_CLEAR_RX: u8 = 0x02;

/// IIR bit 0 set means no interrupt pending; bits 3:1 give the cause.
const IIR_NO_INTERRUPT: u8 = 0x01;
const IIR_THR_EMPTY: u8 = 0x02;
const IIR_RX_AVAILABLE: u8 = 0x04;
/// IIR bits 7:6 read 11 when the FIFOs are enabled; this is how the kernel tells a 16550A from
/// an 8250.
const IIR_FIFO_ENABLED: u8 = 0xc0;

/// MCR bits 4:0 are implemented.
const MCR_MASK: u8 = 0x1f;
const MCR_DTR: u8 = 0x01;
const MCR_RTS: u8 = 0x02;
const MCR_OUT1: u8 = 0x04;
const MCR_OUT2: u8 = 0x08;
const MCR_LOOPBACK: u8 = 0x10;

/// Transmitter is always idle: output is instantaneous.
const LSR_DATA_READY: u8 = 0x01;
const LSR_THR_EMPTY: u8 = 0x20;
const LSR_TRANSMITTER_EMPTY: u8 = 0x40;

const MSR_CTS: u8 = 0x10;
const MSR_DSR: u8 = 0x20;
const MSR_RI: u8 = 0x40;
const MSR_DCD: u8 = 0x80;

/// Depth of the 16550A receive FIFO.
const FIFO_DEPTH: usize = 16;

pub(crate) struct Uart {
    console: ConsoleHandle,
    rx: VecDeque<u8>,
    /// Divisor latch bytes; stored for read-back only, the baud rate has no effect.
    dll: u8,
    dlm: u8,
    ier: u8,
    fcr: u8,
    lcr: u8,
    mcr: u8,
    scr: u8,
    /// THRE interrupt cause is armed: set by a THR write or by enabling the interrupt in IER,
    /// cleared when IIR reports it.
    thre_pending: bool,
    /// Interrupt level at the last `irq_edge` call, for edge detection.
    last_level: bool,
    /// Cleared while the virtio console is active so host input goes to hvc0, not the 16550.
    rx_enabled: bool,
}

impl Uart {
    pub fn new(console: ConsoleHandle) -> Self {
        Uart {
            console,
            rx: VecDeque::new(),
            dll: 0,
            dlm: 0,
            ier: 0,
            fcr: 0,
            lcr: 0,
            mcr: 0,
            scr: 0,
            thre_pending: false,
            last_level: false,
            rx_enabled: true,
        }
    }

    fn level(&self) -> bool {
        (self.ier & IER_RX_AVAILABLE != 0 && !self.rx.is_empty())
            || (self.ier & IER_THR_EMPTY != 0 && self.thre_pending)
    }

    /// True on a rising edge of the interrupt line. The PIC is edge triggered, so IRQ 4 must be
    /// raised once per rising edge and not while the level stays high.
    pub fn irq_edge(&mut self) -> bool {
        let level = self.level();
        let edge = level && !self.last_level;
        self.last_level = level;
        edge
    }

    pub fn set_rx_enabled(&mut self, on: bool) {
        self.rx_enabled = on;
    }

    /// Moves host input into the receive FIFO. Returns true if any byte arrived.
    pub fn poll(&mut self) -> bool {
        let mut any = false;
        if !self.rx_enabled {
            return false;
        }
        while self.rx.len() < FIFO_DEPTH {
            match self.console.guest_rx() {
                Some(byte) => {
                    self.rx.push_back(byte);
                    any = true;
                }
                None => break,
            }
        }
        any
    }

    fn lsr(&self) -> u8 {
        let data_ready = if self.rx.is_empty() {
            0
        } else {
            LSR_DATA_READY
        };
        LSR_THR_EMPTY | LSR_TRANSMITTER_EMPTY | data_ready
    }

    fn msr(&self) -> u8 {
        if self.mcr & MCR_LOOPBACK != 0 {
            // Loopback wires the outputs back to the inputs: RTS->CTS, DTR->DSR, OUT1->RI,
            // OUT2->DCD. The 8250 autoconfig probe checks this.
            let line = |mcr_bit: u8, msr_bit: u8| {
                if self.mcr & mcr_bit != 0 { msr_bit } else { 0 }
            };
            line(MCR_RTS, MSR_CTS)
                | line(MCR_DTR, MSR_DSR)
                | line(MCR_OUT1, MSR_RI)
                | line(MCR_OUT2, MSR_DCD)
        } else {
            // Outside loopback the modem inputs always read as asserted.
            MSR_CTS | MSR_DSR | MSR_DCD
        }
    }

    pub fn read(&mut self, reg: u16) -> u8 {
        let dlab = self.lcr & LCR_DLAB != 0;
        match reg {
            REG_DATA if dlab => self.dll,
            REG_DATA => {
                let byte = self.rx.pop_front().unwrap_or(0);
                self.poll();
                byte
            }
            REG_IER if dlab => self.dlm,
            REG_IER => self.ier,
            REG_IIR_FCR => {
                let fifo = if self.fcr & FCR_FIFO_ENABLE != 0 {
                    IIR_FIFO_ENABLED
                } else {
                    0
                };
                if self.ier & IER_RX_AVAILABLE != 0 && !self.rx.is_empty() {
                    IIR_RX_AVAILABLE | fifo
                } else if self.ier & IER_THR_EMPTY != 0 && self.thre_pending {
                    // Reading IIR acknowledges the THRE cause.
                    self.thre_pending = false;
                    IIR_THR_EMPTY | fifo
                } else {
                    IIR_NO_INTERRUPT | fifo
                }
            }
            REG_LCR => self.lcr,
            REG_MCR => self.mcr,
            REG_LSR => self.lsr(),
            REG_MSR => self.msr(),
            _ => self.scr,
        }
    }

    pub fn write(&mut self, reg: u16, value: u8) {
        let dlab = self.lcr & LCR_DLAB != 0;
        match reg {
            REG_DATA if dlab => self.dll = value,
            REG_DATA => {
                self.thre_pending = true;
                if self.mcr & MCR_LOOPBACK != 0 {
                    if self.rx.len() < FIFO_DEPTH {
                        self.rx.push_back(value);
                    }
                } else {
                    self.console.guest_tx(value);
                }
            }
            REG_IER if dlab => self.dlm = value,
            REG_IER => {
                let previous = self.ier;
                self.ier = value & IER_MASK;
                // Enabling the THRE interrupt while the transmitter is idle fires it at once.
                if self.ier & IER_THR_EMPTY != 0 && previous & IER_THR_EMPTY == 0 {
                    self.thre_pending = true;
                }
            }
            REG_IIR_FCR => {
                self.fcr = value & FCR_STORED_MASK;
                if value & FCR_CLEAR_RX != 0 {
                    self.rx.clear();
                }
            }
            REG_LCR => self.lcr = value,
            REG_MCR => self.mcr = value & MCR_MASK,
            REG_SCRATCH => self.scr = value,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autoconfig_probe() {
        let mut uart = Uart::new(ConsoleHandle::default());
        uart.write(REG_IER, 0);
        assert_eq!(uart.read(REG_IER), 0);
        uart.write(REG_MCR, MCR_LOOPBACK | MCR_OUT2 | MCR_RTS);
        assert_eq!(uart.read(REG_MSR) & 0xf0, MSR_CTS | MSR_DCD);
        uart.write(REG_IIR_FCR, FCR_FIFO_ENABLE);
        assert_eq!(uart.read(REG_IIR_FCR) & IIR_FIFO_ENABLED, IIR_FIFO_ENABLED);
        uart.write(REG_SCRATCH, 0x5a);
        assert_eq!(uart.read(REG_SCRATCH), 0x5a);
    }

    #[test]
    fn tx_and_rx() {
        let console = ConsoleHandle::default();
        let mut uart = Uart::new(console.clone());
        uart.write(REG_DATA, b'x');
        let mut buf = [0u8; 4];
        assert_eq!(console.read(&mut buf, None), 1);
        assert_eq!(buf[0], b'x');
        console.write(b"hi");
        assert!(uart.poll());
        uart.write(REG_IER, IER_RX_AVAILABLE);
        assert!(uart.irq_edge());
        assert_eq!(uart.read(REG_IIR_FCR) & 0xf, IIR_RX_AVAILABLE);
        assert_eq!(uart.read(REG_DATA), b'h');
        assert_eq!(uart.read(REG_DATA), b'i');
        assert_eq!(uart.read(REG_LSR) & LSR_DATA_READY, 0);
    }
}
