#!/bin/sh
# cargo's runner for wasm32-wasip1 (.cargo/config.toml): `node run-wasi.mjs <binary> <args...>`.
exec node --stack-size=8000 "$(dirname "$0")/run-wasi.mjs" "$@"
