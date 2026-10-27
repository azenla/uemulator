# Memory

## Physical

- Guest RAM: one anonymous `mmap` region of `memory_mib` (lazy-committed, host page size aware: macOS arm64 uses 16 KiB pages, guest uses 4 KiB; never rely on host page protection for guest page granularity).
- Layout: RAM at `0..ram_size`; MMIO/port space handled by the `Bus` (no PCI hole needed without PCI; keep a gap `3 GiB..4 GiB` unmapped for forward-compat).
- `Bus`: `ram_ptr(gpa)`, `mmio_read/write(gpa,size)` (stubbed until a device needs it), `io_in/io_out(port,size)` dispatch to devices registered by port range; unmapped port reads return all ones.

## Soft TLB (`tlb.rs`)

- Two MMU indexes: kernel (CPL 0–2) and user (CPL 3), plus a physical (paging off) index. Size 256–1024 entries per index, direct mapped, power of two, `#[repr(C)]` so the JIT inlines the lookup.
- Entry: `{ tag_read, tag_write, tag_exec: u64 (page-aligned vaddr | flag bits), addend: i64 /* host_ptr - vaddr */, gpa }`. Mismatch or flag bits set (MMIO, code-page, not-dirty) → slow path.
- Fast path (JIT inline, interpreter calls the same function): `idx = (va >> 12) & mask; if va_page == entry.tag_x { host = va + entry.addend }`. Accesses crossing a page boundary always go slow.
- Flush: CR3 write, CR0.PG/WP and CR4.PGE/PAE/PSE changes, `invlpg` (single page), CPL change selects the other index (no flush needed), INVPCID not advertised.
- Global pages (CR4.PGE) survive CR3 writes if tracked; optimization, not required.

## Page walk (`mmu.rs`)

4-level long-mode walk (4 KiB/2 MiB/1 GiB), PAE (3-level, needed by the decompressor transitions if any), 32-bit non-PAE low priority. Honors U/S, R/W, NX (EFER.NXE), CR0.WP, reserved bits (raise `#PF` with RSVD), sets A and D bits in guest memory atomically (single-threaded: plain RMW). No SMEP/SMAP (not advertised). Canonical-address checks (#GP/#SS) in the translator.

Page fault error code and CR2 are set in `Cpu::page_fault` (`mem.rs`). Walk reads page-table entries through the physical RAM map directly (never the soft TLB).

## Code-page tracking (self-modifying code)

- Key TBs by **guest physical** page. Each physical page has a list of TB ids and a `has_code` bit.
- When a TB is translated, its pages' TLB write entries get the `CODE` flag (forcing the slow store path), as QEMU does with `TLB_NOTDIRTY`. Slow store checks the page's TB list, invalidates overlapping TBs, clears the flag when the list is empty.
- Exec TLB entry tag includes the physical page so aliased virtual mappings of the same physical page are caught (`text_poke` writes through an alias mapping).
- TBs spanning two pages register on both pages; the second page's gpa is re-translated at lookup time (key = gpa of pc + hflags + a second-page check).
- Invalidation unlinks chained jumps into the dead TB (JIT) and drops jump-cache entries.
- Also invalidate on DMA-like host writes (none in v1) — kernel loader writes before first translation.

## Atomicity

Single vCPU thread. LOCK and cmpxchg are plain RMW in the helper; no host atomics needed. The host console thread talks to the vCPU thread only through queues and `exit_request`/`irq` atomics.
