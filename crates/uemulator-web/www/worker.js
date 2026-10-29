// Runs the emulator. `uw_run` blocks until the guest exits, so it lives here and not on the page.
// While it runs this worker cannot receive messages: keystrokes, resizes and the stop request come
// through the SharedArrayBuffer `control` that main.js shares with us (layout in protocol.js).
import { CTRL, RING_BYTES, RING_OFFSET } from "./protocol.js";

const OUTPUT_FLUSH_MS = 16;
const OUTPUT_FLUSH_BYTES = 1 << 16;
// Also the latency of a guest timer that falls due while the guest is halted.
const IDLE_WAIT_MS = 1;

let wasm;
let control;
let ring;
let outputChunks = [];
let outputBytes = 0;
let lastFlush = 0;
let lastResizeGeneration = 0;
const decoder = new TextDecoder();

function memoryBytes(ptr, len) {
  return new Uint8Array(wasm.memory.buffer, ptr, len);
}

function flushOutput() {
  lastFlush = performance.now();
  if (outputBytes === 0) return;
  const merged = new Uint8Array(outputBytes);
  let offset = 0;
  for (const chunk of outputChunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  outputChunks = [];
  outputBytes = 0;
  postMessage({ type: "output", bytes: merged.buffer }, [merged.buffer]);
}

const host = {
  host_output(ptr, len) {
    outputChunks.push(memoryBytes(ptr, len).slice());
    outputBytes += len;
    if (outputBytes >= OUTPUT_FLUSH_BYTES) flushOutput();
  },
  host_input(ptr, cap) {
    if (performance.now() - lastFlush >= OUTPUT_FLUSH_MS) flushOutput();
    const head = Atomics.load(control, CTRL.HEAD);
    let tail = Atomics.load(control, CTRL.TAIL);
    let count = 0;
    const out = memoryBytes(ptr, cap);
    while (tail !== head && count < cap) {
      out[count++] = ring[tail];
      tail = (tail + 1) % RING_BYTES;
    }
    Atomics.store(control, CTRL.TAIL, tail);
    return count;
  },
  host_resize(ptr) {
    const generation = Atomics.load(control, CTRL.RESIZE_GENERATION);
    if (generation === lastResizeGeneration) return 0;
    lastResizeGeneration = generation;
    // `ptr` is two u16; it is 2-aligned because it points at a Rust `[u16; 2]`.
    const size = new Uint16Array(wasm.memory.buffer, ptr, 2);
    size[0] = Atomics.load(control, CTRL.COLS);
    size[1] = Atomics.load(control, CTRL.ROWS);
    return 1;
  },
  host_should_stop() {
    return Atomics.load(control, CTRL.STOP);
  },
  host_idle() {
    flushOutput();
    const head = Atomics.load(control, CTRL.HEAD);
    if (head === Atomics.load(control, CTRL.TAIL)) {
      Atomics.wait(control, CTRL.HEAD, head, IDLE_WAIT_MS);
    }
  },
  host_now_ms() {
    return performance.now();
  },
  host_log(ptr, len) {
    postMessage({ type: "log", text: decoder.decode(memoryBytes(ptr, len)) });
  },
};

function lastError() {
  const text = decoder.decode(memoryBytes(wasm.uw_error_ptr(), wasm.uw_error_len()));
  return text;
}

// Copies into module memory. `uw_alloc` may grow (and so detach) the memory, so the view is made
// after it returns.
function upload(bytes) {
  const ptr = wasm.uw_alloc(bytes.byteLength);
  memoryBytes(ptr, bytes.byteLength).set(new Uint8Array(bytes.buffer ?? bytes, bytes.byteOffset ?? 0, bytes.byteLength));
  return ptr;
}

async function boot(message) {
  control = new Int32Array(message.control, 0, RING_OFFSET / 4);
  ring = new Uint8Array(message.control, RING_OFFSET, RING_BYTES);
  const response = await fetch(new URL("uemulator_web.wasm", import.meta.url));
  const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), { host });
  wasm = instance.exports;

  const kernel = new Uint8Array(message.kernel);
  const initrd = message.initrd ? new Uint8Array(message.initrd) : new Uint8Array(0);
  const cmdline = new TextEncoder().encode(message.cmdline);
  const kernelPtr = upload(kernel);
  const initrdPtr = upload(initrd);
  const cmdlinePtr = upload(cmdline);
  const failed = wasm.uw_create(
    kernelPtr, kernel.length, initrdPtr, initrd.length, cmdlinePtr, cmdline.length,
    message.memoryMib, Math.floor(Date.now() / 1000), message.cols, message.rows,
  );
  if (failed) {
    postMessage({ type: "exit", code: -2, message: lastError() });
    return;
  }
  postMessage({ type: "started" });
  lastFlush = performance.now();
  const code = wasm.uw_run();
  flushOutput();
  postMessage({ type: "exit", code, message: code === 4 ? lastError() : "" });
}

onmessage = (event) => {
  boot(event.data).catch((error) => {
    postMessage({ type: "exit", code: -2, message: String(error?.stack ?? error) });
  });
};
