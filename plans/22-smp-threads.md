# One host thread per vCPU

With `MachineConfig::cpus(n)` the guest has n vCPUs. Time-sharing them on one host thread (`plans/03-cpu.md`, "SMP") is the deterministic mode and the only one a freestanding host without threads can run. This plan is the other mode: every vCPU on its own host thread, so a parallel guest workload really runs in parallel. Four simultaneous 40 MiB `md5sum`s finish in the time one takes alone.

`SmpMode` picks the mode (`MachineConfig::smp`, CLI `--smp auto|timeshare|threads`, saved in `machine.toml`):

- `Auto` (default): threads when everything below holds, time-sharing otherwise.
- `Timeshare`: never threads.
- `Threads`: `EmulatedMachine::new` fails instead of falling back.

Threads need: more than one vCPU, `ClockMode::Realtime` (the default), no `verify_jit`, a host clock, and a way to start threads. Hosted builds have `std::thread` and `Instant`. A freestanding build (UEFI, WebAssembly) gets threads only if the embedder gives `MachineConfig::thread_spawner` (a closure that runs a `Box<dyn FnOnce() + Send>` on a new thread) and a `host_clock`; the host must offer real parallel threads (WebAssembly: shared memory and atomics). Without them the machine time-shares exactly as before.

Why realtime only: the virtual clock is the instruction count, which cannot be one deterministic timeline across threads, and Linux checks that the TSC is the same on every CPU. In threaded mode guest time is the shared host clock (`Cpu::wall_clock`), so every vCPU reads the same TSC.

## What is shared, what is private

Shared by every vCPU, as handles (`Arc`) each `Cpu` clones:

- RAM: `ram::GuestRam`, one allocation. It dereferences to `[u8]`, so code that already took slices works, but it is a data race in Rust's terms when a guest races with itself. That is what a guest asked for; anything that needs a defined value goes through `cpu/atomic.rs` (raw host pointers, atomics).
- The code-page flags: `ram::CodePages`, one atomic byte per page, "some vCPU has code from this page".
- The devices: `dev::handle::DevHandle`. `lock()` is a no-op counter bump with one host thread and the device lock (QEMU's "big lock") with several. Reentrant per handle, because guest-facing code often locks in an `if let` scrutinee and again in the body. Every access to a device, interrupt controller or timer happens inside it, so the device models stay single-threaded code. `Devices::current_cpu` is set on every outermost lock from the handle's vCPU number.
- `cpu::vcpus::VcpuLinks`: one `attention` word per vCPU, the exclusive section, the invalidation log, the exit reason.

Private to a vCPU: registers and everything architectural, the soft TLB, the decode cache, and **its own JIT** (own code buffer and block cache, own chain links). The JIT patches chain links and the jump cache in memory another thread would be executing; per-thread instances need none of that to be thread safe, and a vCPU that has not run yet owns no JIT at all. The price is that each vCPU that runs code translates it for itself.

## Signalling

A vCPU looks only at its own `attention` word, at block boundaries (`before_step`, which runs at least every `EVENT_INTERVAL` = 4096 instructions even inside chained JIT blocks, and per instruction in the interpreter). Nobody writes to another vCPU's `Cpu`, with one exception, the exclusive section below.

| bit | set by | meaning |
|-----|--------|---------|
| IRQ, NMI, START | the device model, at every release of the device lock (`Devices::publish_wake_hints`, recomputed from scratch from the local APICs and the PIC) | a deliverable interrupt, an NMI, an INIT or start-up IPI is waiting. The hot loop reads these instead of taking the device lock every block |
| EXCLUSIVE | `Cpu::with_exclusive` | park at the next boundary |
| INVALIDATE | `VcpuLinks::publish_invalidation` | translations of some pages are stale |
| EXIT | `VcpuLinks::request_exit` | the machine is ending |

A halted vCPU sleeps in `Waker::wait` (park with a timeout; 100 ms, or 1 ms while an `mwait` watch has to be polled). New device bits and EXIT unpark it. EXCLUSIVE and INVALIDATE do not: a halted vCPU is not running, and it handles both before it executes anything.

The device thread is the thread that called `run_threaded`. It ticks the devices (PIT, HPET, every local APIC timer, the virtio queues, the UART) at the earliest deadline or every millisecond, whichever is sooner, so host input is seen within a millisecond. A vCPU that touched the devices wakes it if it sleeps (`poke_device_thread`), so a virtio notify does not wait out the millisecond. The same thread turns `reset_requested`, `shutdown_requested` and the stop flag into an exit, and calls the embedder's poll hook. Device time only moves forward under the device lock (`Devices::clamp_time`): a vCPU may have sampled the clock before waiting for the lock, and a counter read at the older time would run backwards for the guest. On a freestanding host the device thread waits in the poll hook (`poll(true)`) instead of sleeping; a halted vCPU thread there can only spin (`Waker::wait`), so the embedder's threads should be ones it does not mind burning.

## Exit

The first vCPU to leave `run_slice` records its reason (`request_exit`; first wins), which sets EXIT everywhere. `run_threaded` returns once every thread has handed its `Cpu` back, so a reboot (`reboot(true)`) rebuilds a whole new set. A panicking vCPU thread reports `Internal` instead of hanging the machine (`ThreadGuard`).

The machine ends with `Shutdown` when every vCPU is halted, none is waiting for a start-up IPI with interrupts enabled (`wakeable`), and no NMI is pending: the `cli; hlt` of a guest that powered off without ACPI. A halted BSP with an AP still working does not end the run.

## Exclusive section

`Cpu::with_exclusive(f)` runs `f` with every other vCPU parked at a block boundary or halted. It is QEMU's `start_exclusive`: one owner at a time (an owner waiting for the lock stays parkable, because the current owner is waiting for it), `running` counts the vCPUs that are executing, the owner sets `exclusive_requested`, sets EXCLUSIVE on the others and waits for the count to reach zero. Entering the running state increments the count first and then checks the flag; the owner sets the flag first and then reads the count; so one of them sees the other. Never call it with the device lock held.

It is used for two things:

1. **Flagging a page as code** (`mark_code`). The flag is set, the fence runs, every vCPU's soft TLB loses its write tags for the page (`revoke_write_tags_for`, done by the owner on the parked vCPUs' `Cpu`s through `VcpuLink::cpu`), and only then does the caller read the guest bytes to decode. A store by another vCPU is then either already in those bytes or takes the slow path, sees the flag and invalidates. All three decode paths (`fetch_decode`, the JIT's `translate`, the straddling-instruction path) flag before they read.
2. **A locked access no host atomic covers** (see below): split across a page, not naturally aligned, or a 16-byte compare-exchange where the host has no 16-byte atomic.

## Self-modifying and cross-modifying code

A store to a page with the flag set takes the slow path (`phys_write`): it stores, fences, then checks the flag (the other half of the Dekker pair above) and calls `Cpu::code_written`. That drops this vCPU's translations of the page, clears the flag and appends the page to the shared invalidation log, setting INVALIDATE on every other vCPU, which drops its own translations at its next boundary (`drain_invalidations`; a vCPU that fell more than 4096 entries behind throws everything away instead). Device DMA does the same from the device thread (`GuestMem::write_raw` stores, fences and then records the code pages it hit; the thread publishes them).

Another vCPU may therefore run a stale translation for a few instructions after the store. That is permitted: cross-modifying code must be synchronized by the guest (SDM vol. 3, 8.1.3), with a serializing instruction on the executing CPU, which Linux's `text_poke` does with an IPI. An interrupt is delivered only after `thread_attention` has applied the log, so the IPI is a barrier by construction, and `cpuid` asks the running block to end (`Cpu::serialize`) when invalidations are pending.

`flush_decode` does not clear flags in threaded mode (other vCPUs' caches depend on them).

## Atomics and ordering

x86 is TSO. Acquire loads and release stores are exactly that and are plain `mov`s on an x86 host; on aarch64 they are `ldar`/`stlr`, which is what makes a guest spin lock (`lock cmpxchg` to take it, a plain `mov` to release) work. QEMU declines to enable multi-threaded TCG for an x86 guest on an aarch64 host by default for this reason; here it is on and paid for:

- Interpreter: with `Cpu::ordered_memory` set, `mem_read`, `mem_write`, `phys_read`, `phys_write` and the page walker use `atomic::load`/`store`. Natural alignment makes one host atomic; every other size from 1 to 8 (including the 3, 5, 6, 7 byte pieces of a page-straddling access) is an exact-width copy with a fence. Reading or writing a whole `u64` there takes in the neighbouring bytes (the first version did, and zeroed the instruction after a straddling `cmpxchg16b` in kernel text; `every_size_up_to_eight_touches_exactly_its_bytes` and `ordered_memory_accesses_touch_exactly_their_own_bytes_at_every_offset` pin it).
- aarch64 JIT: with `Jit::set_ordered_memory`, `Gen::load_access` emits `add; ldar` and `store_access` `add; stlr`, and `tlb_probe` sends a misaligned access to the slow path, because `ldar`/`stlr` fault on one. Blocks compiled before the flag was set are plain, so it is set before the first translation. The x86-64 JIT needs nothing.
- Locked instructions (any `lock`-prefixed read-modify-write, and `xchg` with memory, which is locked without the prefix) are atomic through `Cpu::exec_locked`, not a second implementation of each. The instruction runs normally with `Cpu::rmw` armed; its `load_operand` of the destination becomes `locked_load` (translate for writing, read atomically, remember address, size and value) and its `store_operand` becomes `locked_store`, a compare-exchange of the remembered value. If another vCPU got there first the store fails with `Exception::AtomicRetry`, the registers and flags are restored and the instruction runs again. `cmpxchg8b` goes through the same two calls; `cmpxchg16b` is a 16-byte host atomic (`ldaxp`/`stlxp` on aarch64, `cmpxchg16b` on x86-64). Where no host atomic applies, the instruction is re-run under `with_exclusive` with `Rmw::Plain` set, so that `exec` does not take the locked path again from inside the section (that waited for itself forever; `with_exclusive` also runs a nested request in place). An instruction with a memory operand that finishes still armed or loaded bypassed the pair and was not atomic: `check_locked_instruction_was_atomic` asserts under `debug_assertions` and logs once otherwise. The JIT never inlines a locked instruction or `xchg`/`cmpxchg`/`xadd`; they go through `helper_exec` into this code.
- Page-table Accessed/Dirty updates, of the guest's tables and of the nested (NPT) tables a guest hypervisor keeps, are compare-exchanges on the entry and walk again when it changed (`update_ram_u64`/`u32`), as hardware does. A plain read-modify-write would race Linux's atomic `ptep_set_wrprotect` (copy-on-write on fork) and write the old writable bit back.

Rosetta runs x86 code with hardware TSO on, so ordering bugs show only in a native aarch64 test run.

## Not supported, and costs

- `verify_jit` (lockstep needs one deterministic timeline) and firmware images and the BIOS disk shim (as with time-sharing).
- Each vCPU's JIT reserves its own 480 MiB code mapping (lazily committed); 64 vCPUs boot Alpine on a 14-core Mac. A misaligned access on aarch64 takes the slow path.
- The device lock is one lock: guest MMIO and port traffic from many vCPUs serializes on it. Local APIC accesses are MSRs in x2APIC mode and take it too.

## Tests

- `cpu/atomic.rs`: sizes 1 to 8 at every alignment touch exactly their bytes, compare-exchange loses no update across threads, the 16-byte exchange, a message-passing litmus test.
- `cpu/smp_tests.rs` (threaded, on both engines, many with an aarch64-only failure when the ordering is removed): INIT/SIPI and shutdown with four threads, 20000 `lock inc`s per vCPU lose no update, a `lock cmpxchg` lock released by a plain store keeps a plain counter exact, the same with `xchg`, message passing with the BSP's JIT code as the consumer, an AP taking an IPI through its IVT, `mwait` woken by a store from another thread, code overwritten by another thread is translated again, a store cannot skip `code_written` after another thread flagged the page, 64 pages flagged by four threads at once, a locked increment of a misaligned dword and a locked add straddling two pages (the exclusive replay), a `lock bts` spin lock, MONITOR armed before the store that MWAIT must see.
- `machine/tests.rs`: a second thread started through the public API, through the embedder's `thread_spawner` (once per vCPU), reboot joins and rebuilds every thread, stop with every vCPU spinning, shutdown when all halt with IF=0, `SmpMode::Threads` refuses the virtual clock.
- Real guests (not tests): Alpine 6.18 with 4 vCPUs and with 64, all of them taking timer and device interrupts and IPIs, TSC stable (`clocksource: Switched to clocksource tsc`), CPU hot-unplug and plug through sysfs, a 150-process fork storm, four parallel hashes in the time of one.
