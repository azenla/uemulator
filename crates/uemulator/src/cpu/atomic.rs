//! Guest memory accesses with x86 ordering, for vCPUs running on separate host threads.
//!
//! x86 is TSO: loads are not reordered with other loads, stores are not reordered with other
//! stores or with earlier loads, and only a store may pass a later load. Acquire loads and
//! release stores give exactly that (plus store-to-load ordering that x86 does not require), and
//! on an x86 host they are plain `mov`s. On an aarch64 host they are `ldar`/`stlr`, which is what
//! keeps a guest spin lock (`lock cmpxchg` to take, a plain `mov` to release) correct.
//!
//! These take host addresses (the soft TLB's `addend` already produced one), so the caller has
//! done the translation and the bounds check. Natural alignment is what makes an access a single
//! host atomic; a misaligned one is split by the host anyway, so it is copied and fenced, which is
//! as much as x86 promises for it.

use core::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering, fence};

/// Loads `size` (1, 2, 4 or 8) bytes at `host` with acquire ordering.
///
/// # Safety
/// `host..host + size` must be readable memory.
#[inline(always)]
pub(crate) unsafe fn load(host: *const u8, size: u8) -> u64 {
    unsafe {
        match size {
            1 => (*host.cast::<AtomicU8>()).load(Ordering::Acquire) as u64,
            2 if host as usize & 1 == 0 => {
                (*host.cast::<AtomicU16>()).load(Ordering::Acquire) as u64
            }
            4 if host as usize & 3 == 0 => {
                (*host.cast::<AtomicU32>()).load(Ordering::Acquire) as u64
            }
            8 if host as usize & 7 == 0 => (*host.cast::<AtomicU64>()).load(Ordering::Acquire),
            _ => {
                let value = unaligned_load(host, size);
                fence(Ordering::Acquire);
                value
            }
        }
    }
}

/// Reads exactly `size` (1 to 8) bytes, whatever the alignment. A page-straddling guest access
/// reaches `load` and `store` in pieces of any size up to 7; reading or writing a whole `u64`
/// there would take in the neighbouring bytes.
unsafe fn unaligned_load(host: *const u8, size: u8) -> u64 {
    let mut bytes = [0u8; 8];
    unsafe { core::ptr::copy_nonoverlapping(host, bytes.as_mut_ptr(), size as usize) };
    u64::from_le_bytes(bytes)
}

/// The exact-width counterpart of `unaligned_load`.
unsafe fn unaligned_store(host: *mut u8, size: u8, value: u64) {
    let bytes = value.to_le_bytes();
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), host, size as usize) };
}

/// Stores the low `size` bytes of `value` at `host` with release ordering.
///
/// # Safety
/// `host..host + size` must be writable memory.
#[inline(always)]
pub(crate) unsafe fn store(host: *mut u8, size: u8, value: u64) {
    unsafe {
        match size {
            1 => (*host.cast::<AtomicU8>()).store(value as u8, Ordering::Release),
            2 if host as usize & 1 == 0 => {
                (*host.cast::<AtomicU16>()).store(value as u16, Ordering::Release)
            }
            4 if host as usize & 3 == 0 => {
                (*host.cast::<AtomicU32>()).store(value as u32, Ordering::Release)
            }
            8 if host as usize & 7 == 0 => {
                (*host.cast::<AtomicU64>()).store(value, Ordering::Release)
            }
            _ => {
                fence(Ordering::Release);
                unaligned_store(host, size, value);
            }
        }
    }
}

/// Whether `compare_exchange` can be one host atomic: `size` bytes at `host` are naturally aligned.
pub(crate) fn is_atomic_operand(host: *const u8, size: u8) -> bool {
    matches!(size, 1 | 2 | 4 | 8) && host as usize & (size as usize - 1) == 0
}

/// Replaces the `size` bytes at `host` with `new` if they still hold `expected`, with the full
/// barrier a `lock` prefix implies. `Err` carries the value found.
///
/// # Safety
/// `host..host + size` must be writable memory, and `is_atomic_operand(host, size)` must hold.
pub(crate) unsafe fn compare_exchange(
    host: *mut u8,
    size: u8,
    expected: u64,
    new: u64,
) -> Result<(), u64> {
    const ORDER: Ordering = Ordering::SeqCst;
    unsafe {
        match size {
            1 => (*host.cast::<AtomicU8>())
                .compare_exchange(expected as u8, new as u8, ORDER, ORDER)
                .map(drop)
                .map_err(u64::from),
            2 => (*host.cast::<AtomicU16>())
                .compare_exchange(expected as u16, new as u16, ORDER, ORDER)
                .map(drop)
                .map_err(u64::from),
            4 => (*host.cast::<AtomicU32>())
                .compare_exchange(expected as u32, new as u32, ORDER, ORDER)
                .map(drop)
                .map_err(u64::from),
            _ => (*host.cast::<AtomicU64>())
                .compare_exchange(expected, new, ORDER, ORDER)
                .map(drop),
        }
    }
}

/// Atomically sets `bits` in the 8-byte entry at `host` if it still holds `expected`; the page
/// walker's Accessed/Dirty update. `Err` carries the value found.
///
/// # Safety
/// `host` must point at an 8-byte aligned, writable entry.
pub(crate) unsafe fn update_entry_u64(host: *mut u8, expected: u64, new: u64) -> Result<(), u64> {
    unsafe { compare_exchange(host, 8, expected, new) }
}

/// The 32-bit counterpart of `update_entry_u64`.
///
/// # Safety
/// `host` must point at a 4-byte aligned, writable entry.
pub(crate) unsafe fn update_entry_u32(host: *mut u8, expected: u32, new: u32) -> Result<(), u32> {
    unsafe { compare_exchange(host, 4, expected as u64, new as u64).map_err(|found| found as u32) }
}

/// 16-byte compare-and-swap for `cmpxchg16b`. `None`: this host has no such instruction for the
/// guest to share, and the caller must run the access in an exclusive section instead.
/// `Some(Ok(()))` swapped; `Some(Err(found))` did not.
///
/// # Safety
/// `host` must point at 16-byte aligned, writable memory.
pub(crate) unsafe fn compare_exchange_u128(
    host: *mut u8,
    expected: u128,
    new: u128,
) -> Option<Result<(), u128>> {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let (expected_low, expected_high) = (expected as u64, (expected >> 64) as u64);
        let (new_low, new_high) = (new as u64, (new >> 64) as u64);
        let (found_low, found_high): (u64, u64);
        // Load-exclusive pair, compare both halves, store-exclusive pair; retry when another
        // observer broke the reservation. A mismatch leaves the loop with `clrex`.
        core::arch::asm!(
            "2:",
            "ldaxp {found_low}, {found_high}, [{host}]",
            "cmp {found_low}, {expected_low}",
            "ccmp {found_high}, {expected_high}, #0, eq",
            "b.ne 3f",
            "stlxp {status:w}, {new_low}, {new_high}, [{host}]",
            "cbnz {status:w}, 2b",
            "b 4f",
            "3:",
            "clrex",
            "4:",
            host = in(reg) host,
            expected_low = in(reg) expected_low,
            expected_high = in(reg) expected_high,
            new_low = in(reg) new_low,
            new_high = in(reg) new_high,
            found_low = out(reg) found_low,
            found_high = out(reg) found_high,
            status = out(reg) _,
            options(nostack),
        );
        let found = found_low as u128 | ((found_high as u128) << 64);
        Some(if found == expected {
            Ok(())
        } else {
            Err(found)
        })
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        #[target_feature(enable = "cmpxchg16b")]
        unsafe fn exchange(host: *mut u8, expected: u128, new: u128) -> u128 {
            unsafe {
                core::arch::x86_64::cmpxchg16b(
                    host.cast::<u128>(),
                    expected,
                    new,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
            }
        }
        // CPUID.1:ECX bit 13. Every x86-64 part but the earliest Opterons has it.
        const CPUID_ECX_CMPXCHG16B: u32 = 1 << 13;
        if core::arch::x86_64::__cpuid(1).ecx & CPUID_ECX_CMPXCHG16B == 0 {
            return None;
        }
        let found = exchange(host, expected, new);
        Some(if found == expected {
            Ok(())
        } else {
            Err(found)
        })
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = (host, expected, new);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    /// 16-byte aligned scratch memory.
    #[repr(align(16))]
    struct Aligned([u8; 64]);

    #[test]
    fn loads_and_stores_round_trip_every_size_and_alignment() {
        let mut memory = Aligned([0; 64]);
        let base = memory.0.as_mut_ptr();
        for size in [1u8, 2, 4, 8] {
            for offset in 0..9usize {
                let value = 0x1122_3344_5566_7788u64 >> (8 * (8 - size as u32) % 64);
                let masked = if size == 8 {
                    value
                } else {
                    value & ((1u64 << (8 * size as u32)) - 1)
                };
                unsafe {
                    store(base.add(offset), size, masked);
                    assert_eq!(load(base.add(offset), size), masked, "{size} at {offset}");
                }
            }
        }
    }

    #[test]
    fn every_size_up_to_eight_touches_exactly_its_bytes() {
        // The pieces of a page-straddling access are 1 to 7 bytes: 3, 5, 6 and 7 are not a host
        // atomic width and must not read or write the bytes around them.
        for size in 1u8..=8 {
            for offset in 8..17usize {
                let mut memory = Aligned([0xaa; 64]);
                let base = memory.0.as_mut_ptr();
                let value = 0x0102_0304_0506_0708u64;
                let expected = if size == 8 {
                    value
                } else {
                    value & ((1u64 << (8 * size as u32)) - 1)
                };
                unsafe { store(base.add(offset), size, value) };
                for (position, byte) in memory.0.iter().enumerate() {
                    let inside = (offset..offset + size as usize).contains(&position);
                    assert_eq!(
                        *byte == 0xaa,
                        !inside,
                        "size {size} at {offset}, byte {position}"
                    );
                }
                assert_eq!(unsafe { load(base.add(offset), size) }, expected);
            }
        }
    }

    #[test]
    fn a_store_writes_exactly_its_width() {
        let mut memory = Aligned([0xff; 64]);
        let base = memory.0.as_mut_ptr();
        unsafe { store(base.add(4), 2, 0) };
        assert_eq!(memory.0[3], 0xff);
        assert_eq!(&memory.0[4..6], [0, 0]);
        assert_eq!(memory.0[6], 0xff);
    }

    #[test]
    fn compare_exchange_swaps_only_on_a_match() {
        let mut memory = Aligned([0; 64]);
        let base = memory.0.as_mut_ptr();
        unsafe {
            store(base, 4, 7);
            assert_eq!(compare_exchange(base, 4, 6, 9), Err(7));
            assert_eq!(load(base, 4), 7);
            assert_eq!(compare_exchange(base, 4, 7, 9), Ok(()));
            assert_eq!(load(base, 4), 9);
        }
    }

    #[test]
    fn only_naturally_aligned_operands_are_atomic() {
        let mut memory = Aligned([0; 64]);
        let base = memory.0.as_mut_ptr();
        unsafe {
            assert!(is_atomic_operand(base, 8));
            assert!(!is_atomic_operand(base.add(4), 8));
            assert!(is_atomic_operand(base.add(4), 4));
            assert!(!is_atomic_operand(base.add(2), 4));
            assert!(!is_atomic_operand(base, 16));
        }
    }

    #[test]
    fn threads_incrementing_with_compare_exchange_lose_no_update() {
        const THREADS: usize = 4;
        const INCREMENTS: u64 = 20_000;
        let memory = Arc::new(Aligned([0; 64]));
        let address = memory.0.as_ptr() as usize;
        let workers: Vec<_> = (0..THREADS)
            .map(|_| {
                std::thread::spawn(move || {
                    let host = address as *mut u8;
                    for _ in 0..INCREMENTS {
                        loop {
                            let seen = unsafe { load(host, 8) };
                            if unsafe { compare_exchange(host, 8, seen, seen + 1) }.is_ok() {
                                break;
                            }
                        }
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(
            unsafe { load(memory.0.as_ptr(), 8) },
            THREADS as u64 * INCREMENTS
        );
    }

    #[test]
    fn the_sixteen_byte_exchange_swaps_both_halves_or_neither() {
        let mut memory = Aligned([0; 64]);
        let base = memory.0.as_mut_ptr();
        let first = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128;
        let second = 0x1111_2222_3333_4444_5555_6666_7777_8888u128;
        unsafe {
            store(base, 8, first as u64);
            store(base.add(8), 8, (first >> 64) as u64);
            let Some(mismatch) = compare_exchange_u128(base, second, 0) else {
                return; // no 16-byte atomic on this host
            };
            assert_eq!(mismatch, Err(first));
            assert_eq!(load(base, 8), first as u64);
            assert_eq!(compare_exchange_u128(base, first, second), Some(Ok(())));
            assert_eq!(load(base, 8), second as u64);
            assert_eq!(load(base.add(8), 8), (second >> 64) as u64);
        }
    }

    #[test]
    fn message_passing_through_release_and_acquire_never_shows_stale_data() {
        // The pattern a guest spin lock relies on: data written before a flag is visible to
        // whoever sees the flag. On an aarch64 host with plain loads and stores this fails.
        const ROUNDS: u64 = 200_000;
        let memory = Arc::new(Aligned([0; 64]));
        let address = memory.0.as_ptr() as usize;
        let stop = Arc::new(AtomicBool::new(false));
        let reader_stop = stop.clone();
        let reader = std::thread::spawn(move || {
            let host = address as *mut u8;
            let mut seen = 0u64;
            let mut violations = 0u64;
            while !reader_stop.load(Ordering::Relaxed) {
                let flag = unsafe { load(host.add(8), 8) };
                let data = unsafe { load(host, 8) };
                if data < flag {
                    violations += 1;
                }
                seen = seen.max(flag);
            }
            (violations, seen)
        });
        let host = address as *mut u8;
        for round in 1..=ROUNDS {
            unsafe {
                store(host, 8, round);
                store(host.add(8), 8, round);
            }
        }
        stop.store(true, Ordering::Relaxed);
        let (violations, _) = reader.join().unwrap();
        assert_eq!(violations, 0);
    }
}
