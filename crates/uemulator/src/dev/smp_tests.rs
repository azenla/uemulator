//! Interrupt routing between several local APICs: IPIs of every addressing mode, device
//! interrupts to the APIC they name, INIT/start-up delivery, wake-up conditions and per-vCPU
//! timers.

use super::apic::{
    APIC_BASE_ENABLE, APIC_BASE_X2APIC, APIC_DEFAULT_BASE, DELIVERY_FIXED, DELIVERY_INIT,
    DELIVERY_LOWEST_PRIORITY, DELIVERY_NMI, DELIVERY_STARTUP, X2APIC_MSR_FIRST,
};
use super::*;
use crate::console::ConsoleHandle;

const SVR_OFFSET: u32 = 0xf0;
const SVR_SOFTWARE_ENABLE: u32 = 0x1ff;
const TPR_OFFSET: u32 = 0x80;
const LDR_OFFSET: u32 = 0xd0;
const ICR_LOW_OFFSET: u32 = 0x300;
const ICR_HIGH_OFFSET: u32 = 0x310;
const LVT_TIMER_OFFSET: u32 = 0x320;
const TIMER_INITIAL_OFFSET: u32 = 0x380;
const ICR_X2APIC_MSR: u32 = X2APIC_MSR_FIRST + (ICR_LOW_OFFSET >> 4);
const ICR_LOGICAL_DESTINATION: u32 = 1 << 11;
const ICR_SHORTHAND_SHIFT: u32 = 18;
const ICR_DELIVERY_SHIFT: u32 = 8;
const ICR_ASSERT: u32 = 1 << 14;
const IPI_VECTOR: u32 = 0x41;
const BROADCAST_X2APIC: u32 = 0xffff_ffff;
const MSI_DESTINATION_SHIFT: u32 = 12;
const DEVICE_REQUESTER: u16 = 8;
const MESSAGE_VECTOR: u32 = 0x55;
const ONE_SHOT_TIMER_VECTOR: u32 = 0x60;
const TIMER_TICKS: u32 = 10;
const FAR_FUTURE_NS: u64 = 1_000_000;

fn machine(cpus: usize) -> Devices {
    let mut dev = Devices::new(
        ConsoleHandle::default(),
        DevConfig {
            cpus,
            ..Default::default()
        },
    );
    for cpu in 0..cpus {
        dev.set_current_cpu(cpu);
        dev.apic_register_write(SVR_OFFSET, SVR_SOFTWARE_ENABLE, 0)
            .unwrap();
    }
    dev.set_current_cpu(0);
    dev
}

fn x2apic_machine(cpus: usize) -> Devices {
    let mut dev = machine(cpus);
    for cpu in 0..cpus {
        dev.set_current_cpu(cpu);
        assert!(
            dev.lapic_mut()
                .set_base_msr(APIC_DEFAULT_BASE | APIC_BASE_ENABLE | APIC_BASE_X2APIC)
        );
    }
    dev.set_current_cpu(0);
    dev
}

fn pending(dev: &mut Devices, cpu: usize) -> Option<u8> {
    dev.lapic_of(cpu).pending_vector()
}

fn all_pending(dev: &mut Devices) -> Vec<Option<u8>> {
    (0..dev.cpu_count()).map(|cpu| pending(dev, cpu)).collect()
}

/// `wrmsr` to the x2APIC ICR on `sender`.
fn send_x2apic(dev: &mut Devices, sender: usize, destination: u32, low: u32) {
    dev.set_current_cpu(sender);
    let value = (destination as u64) << 32 | low as u64;
    assert!(dev.apic_msr_write(ICR_X2APIC_MSR, value, 0));
}

fn message(destination: u32) -> Source {
    Source::Message(InterruptRequest {
        requester: DEVICE_REQUESTER,
        address: APIC_DEFAULT_BASE | (destination as u64) << MSI_DESTINATION_SHIFT,
        data: MESSAGE_VECTOR,
    })
}

#[test]
fn an_ipi_reaches_only_the_addressed_cpu() {
    let mut dev = x2apic_machine(4);
    send_x2apic(&mut dev, 0, 2, IPI_VECTOR);
    assert_eq!(all_pending(&mut dev), [None, None, Some(0x41), None]);
}

#[test]
fn a_broadcast_destination_reaches_every_cpu_including_the_sender() {
    let mut dev = x2apic_machine(3);
    send_x2apic(&mut dev, 1, BROADCAST_X2APIC, IPI_VECTOR);
    assert_eq!(all_pending(&mut dev), [Some(0x41); 3]);
}

#[test]
fn the_icr_shorthands_pick_self_everyone_or_everyone_else() {
    let mut dev = x2apic_machine(3);
    send_x2apic(&mut dev, 1, 0, IPI_VECTOR | 1 << ICR_SHORTHAND_SHIFT);
    assert_eq!(all_pending(&mut dev), [None, Some(0x41), None]);
    let mut dev = x2apic_machine(3);
    send_x2apic(&mut dev, 1, 0, IPI_VECTOR | 2 << ICR_SHORTHAND_SHIFT);
    assert_eq!(all_pending(&mut dev), [Some(0x41); 3]);
    let mut dev = x2apic_machine(3);
    send_x2apic(&mut dev, 1, 0, IPI_VECTOR | 3 << ICR_SHORTHAND_SHIFT);
    assert_eq!(all_pending(&mut dev), [Some(0x41), None, Some(0x41)]);
}

#[test]
fn x2apic_logical_destinations_select_by_cluster_and_bit_mask() {
    // The logical id is derived from the APIC id: cluster id >> 4, one bit for id & 15.
    let mut dev = x2apic_machine(4);
    send_x2apic(&mut dev, 0, 0b1010, IPI_VECTOR | ICR_LOGICAL_DESTINATION);
    assert_eq!(all_pending(&mut dev), [None, Some(0x41), None, Some(0x41)]);
    // Another cluster matches none of them.
    let mut dev = x2apic_machine(4);
    send_x2apic(
        &mut dev,
        0,
        1 << 16 | 0b1111,
        IPI_VECTOR | ICR_LOGICAL_DESTINATION,
    );
    assert_eq!(all_pending(&mut dev), [None; 4]);
}

#[test]
fn xapic_flat_logical_destinations_match_the_ldr_mask() {
    let mut dev = machine(4);
    for cpu in 0..4 {
        dev.set_current_cpu(cpu);
        dev.apic_register_write(LDR_OFFSET, (1 << cpu) << 24, 0)
            .unwrap();
    }
    dev.set_current_cpu(0);
    dev.apic_register_write(ICR_HIGH_OFFSET, 0b0110 << 24, 0)
        .unwrap();
    dev.apic_register_write(ICR_LOW_OFFSET, IPI_VECTOR | ICR_LOGICAL_DESTINATION, 0)
        .unwrap();
    assert_eq!(all_pending(&mut dev), [None, Some(0x41), Some(0x41), None]);
}

#[test]
fn a_physical_xapic_destination_of_ff_is_a_broadcast() {
    let mut dev = machine(3);
    dev.apic_register_write(ICR_HIGH_OFFSET, 0xff << 24, 0)
        .unwrap();
    dev.apic_register_write(ICR_LOW_OFFSET, IPI_VECTOR, 0)
        .unwrap();
    assert_eq!(all_pending(&mut dev), [Some(0x41); 3]);
}

#[test]
fn lowest_priority_goes_to_the_least_busy_cpu_and_equal_ones_take_turns() {
    let mut dev = x2apic_machine(3);
    for (cpu, tpr) in [(0, 0x30), (1, 0x10), (2, 0x20)] {
        dev.set_current_cpu(cpu);
        dev.apic_register_write(TPR_OFFSET, tpr, 0).unwrap();
    }
    let lowest = DELIVERY_LOWEST_PRIORITY << ICR_DELIVERY_SHIFT;
    send_x2apic(&mut dev, 0, BROADCAST_X2APIC, IPI_VECTOR | lowest);
    // Only CPU 1's priority class (0x10) is below the vector's, and it is the least busy.
    assert_eq!(all_pending(&mut dev), [None, Some(0x41), None]);

    let mut dev = x2apic_machine(3);
    let mut receivers = Vec::new();
    for vector in [0x41, 0x51, 0x61] {
        send_x2apic(&mut dev, 0, BROADCAST_X2APIC, vector | lowest);
        let who = all_pending(&mut dev)
            .iter()
            .position(|pending| *pending == Some(vector as u8))
            .expect("someone got it");
        receivers.push(who);
    }
    receivers.sort_unstable();
    assert_eq!(
        receivers,
        [0, 1, 2],
        "equal priorities must not pile up on one CPU"
    );
}

#[test]
fn an_msi_goes_to_the_apic_its_address_names() {
    let mut dev = machine(4);
    dev.deliver(message(3), None);
    assert_eq!(
        all_pending(&mut dev),
        [None, None, None, Some(MESSAGE_VECTOR as u8)]
    );
}

#[test]
fn an_msi_for_an_apic_id_nobody_has_is_dropped() {
    let mut dev = machine(2);
    dev.deliver(message(9), None);
    assert_eq!(all_pending(&mut dev), [None, None]);
}

#[test]
fn an_ioapic_style_interrupt_follows_its_destination_field() {
    let mut dev = machine(4);
    dev.route_interrupt(&Interrupt {
        vector: 0x70,
        delivery_mode: DELIVERY_FIXED,
        level_triggered: false,
        destination: 1,
        logical: false,
    });
    assert_eq!(all_pending(&mut dev), [None, Some(0x70), None, None]);
}

#[test]
fn only_the_executing_cpus_apic_answers_the_mmio_window() {
    let mut dev = machine(3);
    dev.set_current_cpu(2);
    // APIC ID register, bits 31:24.
    assert_eq!(dev.mmio_read(APIC_DEFAULT_BASE + 0x20, 4), Some(2 << 24));
    dev.set_current_cpu(1);
    assert_eq!(dev.mmio_read(APIC_DEFAULT_BASE + 0x20, 4), Some(1 << 24));
}

#[test]
fn init_and_startup_wait_in_the_target_apic_for_its_vcpu() {
    let mut dev = x2apic_machine(2);
    send_x2apic(
        &mut dev,
        0,
        1,
        DELIVERY_INIT << ICR_DELIVERY_SHIFT | ICR_ASSERT,
    );
    assert!(dev.lapic_of(1).start_signal_pending());
    assert!(!dev.lapic_of(0).start_signal_pending());
    send_x2apic(
        &mut dev,
        0,
        1,
        (DELIVERY_STARTUP << ICR_DELIVERY_SHIFT) | 0x08,
    );
    let target = dev.lapic_of(1);
    assert!(target.take_init());
    assert_eq!(target.take_startup(), Some(0x08));
}

#[test]
fn a_halted_cpu_wakes_for_what_could_interrupt_it() {
    let mut dev = x2apic_machine(2);
    assert!(!dev.wake_pending(1, true), "nothing is pending");
    send_x2apic(&mut dev, 0, 1, IPI_VECTOR);
    assert!(!dev.wake_pending(1, false), "interrupts are disabled there");
    assert!(dev.wake_pending(1, true));
    assert!(!dev.wake_pending(0, true), "it was not addressed");

    let mut dev = x2apic_machine(2);
    send_x2apic(&mut dev, 0, 1, DELIVERY_NMI << ICR_DELIVERY_SHIFT);
    assert!(dev.wake_pending(1, false), "an NMI ignores IF");
    assert!(dev.any_nmi_pending());
    assert!(!dev.wake_pending(0, true));
}

#[test]
fn each_lapic_timer_interrupts_its_own_cpu_and_counts_for_the_machines_next_deadline() {
    let mut dev = machine(3);
    dev.set_current_cpu(1);
    dev.apic_register_write(LVT_TIMER_OFFSET, ONE_SHOT_TIMER_VECTOR, 0)
        .unwrap();
    dev.apic_register_write(TIMER_INITIAL_OFFSET, TIMER_TICKS, 0)
        .unwrap();
    let deadline = dev.next_deadline(0).expect("CPU 1's timer is armed");
    assert!(deadline < FAR_FUTURE_NS);
    let mut ram = vec![0u8; 4096];
    let code = vec![0u8; 1];
    let mut mem = guestmem::GuestMem::new(&mut ram, &code);
    dev.tick(deadline, &mut mem);
    assert_eq!(
        all_pending(&mut dev),
        [None, Some(ONE_SHOT_TIMER_VECTOR as u8), None]
    );
}

#[test]
fn a_cpu_count_of_zero_is_one_cpu_and_acknowledge_follows_the_executing_cpu() {
    let dev = Devices::new(ConsoleHandle::default(), DevConfig::default());
    assert_eq!(dev.cpu_count(), 1);
    let mut dev = x2apic_machine(2);
    send_x2apic(&mut dev, 0, 1, IPI_VECTOR);
    dev.set_current_cpu(0);
    assert_eq!(dev.interrupt_ack(), None);
    dev.set_current_cpu(1);
    assert!(dev.interrupt_pending());
    assert_eq!(dev.interrupt_ack(), Some(0x41));
    assert!(!dev.interrupt_pending());
}
