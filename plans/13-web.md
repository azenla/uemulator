# WebAssembly port

`crates/uemulator-web` builds the library to `wasm32-unknown-unknown` (empty crate on any other target) and `crates/uemulator-web/www` is the page around it. `hack/web.sh [serve]` builds into `target/web` and serves it on `http://127.0.0.1:8080/`; `hack/web/serve.py` adds the two headers that make the page cross-origin isolated.

## What is ported

- Interpreter only: no JIT (`JitMode::Off`; the library builds with `default-features = false`), no virtio disk (`Disk` needs `std::fs`) and no network (no sockets). Console only: virtio-console (`hvc0`), the 16550 carries early boot text.
- Boots the Alpine netboot kernel and initramfs to userspace in about 1 s of guest time and roughly 15 s wall clock in Chrome; with the distro's init it ends at `Mounting boot media: failed` like the UEFI app, so the page defaults to `rdinit=/bin/sh`, which gives a working busybox shell (keyboard included).

## Library changes

`target_os = "uefi"` became `cfg(freestanding)`, set by `crates/uemulator/build.rs` for UEFI and for any `wasm32` target. Freestanding means `no_std`, no threads, no clock and no `std::fs`: the vCPU runs on the caller's thread inside `EmulatedMachine::wait` and the host is reached through `MachineConfig::poll`, exactly as in 12-uefi.md. `libm` is a dependency on the same targets. The RTC starts at `unix_time`, which the page passes from `Date.now()`; guest time is virtual.

## Shape of the port

- No wasm-bindgen: `uemulator-web` exports a small C ABI (`uw_alloc`, `uw_create`, `uw_run`, `uw_error_*`) and imports `host_output`, `host_input`, `host_resize`, `host_should_stop`, `host_idle`, `host_log` from the `host` module. Building needs only `rustup target add wasm32-unknown-unknown`.
- `uw_run` blocks until the guest exits, so it runs in a Web Worker (`www/worker.js`). A worker cannot receive messages during a synchronous call, so the page talks to it through a `SharedArrayBuffer` (layout in `www/protocol.js`): a byte ring for keystrokes, a stop flag, and the terminal size. That is why the page must be cross-origin isolated (COOP `same-origin`, COEP `require-corp`) and cannot be opened from `file://`.
- Guest output is batched in the worker and posted every 16 ms. An idle guest sleeps in `Atomics.wait` on the ring head, woken by the next keystroke.
- wasm32 has 32-bit `usize`; the library needed no change for it, but guest RAM is one `Vec`, so the page offers 128 to 1024 MiB (the module's memory is limited to 4 GiB).

## Not done

- A wasm JIT (translate blocks to wasm functions and instantiate them), which is the only route to more than interpreter speed.
- Disk (a `BlockBackend` over OPFS or an in-memory image) and network (a WebSocket or fetch gateway for `NetHandle`).
- Automated tests: the crate is exercised by hand (and with headless Chrome over CDP during development), since the library's own suite does not run on wasm.
