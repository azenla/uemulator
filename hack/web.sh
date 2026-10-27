#!/bin/sh
# Builds the WebAssembly port into target/web and, with `serve`, serves it on http://127.0.0.1:8080/
# (PORT overrides). Needs `rustup target add wasm32-unknown-unknown`.
#
#   hack/web.sh          build only
#   hack/web.sh serve    build, then serve
#
# Kernel and initramfs: pick files on the page, or run `uemulator quick alpine` once; its cached
# netboot files are linked under assets/ and prefilled.
set -e

cd "$(dirname "${0}")/.." || exit 1

out=target/web
cargo build --release -p uemulator-web --target wasm32-unknown-unknown

rm -rf "$out"
mkdir -p "$out"
cp crates/uemulator-web/www/* "$out/"
cp target/wasm32-unknown-unknown/release/uemulator_web.wasm "$out/"

cache="${UEMULATOR_CACHE:-${UEMULATOR_HOME:-$HOME/.uemulator}/cache}/quick/alpine/artifacts"
if [ -f "$cache/vmlinuz-virt" ] && [ -f "$cache/initramfs-virt" ]; then
    mkdir -p "$out/assets"
    ln -s "$cache/vmlinuz-virt" "$out/assets/vmlinuz-virt"
    ln -s "$cache/initramfs-virt" "$out/assets/initramfs-virt"
fi

if [ "$1" = serve ]; then
    exec python3 hack/web/serve.py "$out" "${PORT:-8080}"
fi
echo "built $out"
