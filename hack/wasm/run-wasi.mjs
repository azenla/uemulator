// Runs a wasm32-wasip1 binary (a cargo test binary, see .cargo/config.toml) under Node WASI, with
// the `host` imports the library's WebAssembly JIT needs. wasmtime would run the binary too, but a
// guest cannot ask it to compile new modules, so the JIT would never be exercised.
import { readFileSync } from "node:fs";
import { WASI } from "node:wasi";
import { createJitHost } from "../../crates/uemulator-web/www/jit_host.js";

const [wasmPath, ...args] = process.argv.slice(2);
const wasi = new WASI({
  version: "preview1",
  args: [wasmPath, ...args],
  env: process.env,
  preopens: { "/": "/", ".": "." },
  returnOnExit: true,
});
const jit = createJitHost();
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {
  wasi_snapshot_preview1: wasi.wasiImport,
  host: jit.host,
});
jit.attach(instance.exports);
process.exitCode = wasi.start(instance);
