#!/usr/bin/env python3
"""Static file server for the web build. SharedArrayBuffer (keyboard input into the worker that
runs the guest) needs the page to be cross-origin isolated, which takes these two headers."""
import http.server
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".js": "text/javascript",
    }

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    root, port = sys.argv[1], int(sys.argv[2])
    handler = lambda *a, **k: Handler(*a, directory=root, **k)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"http://127.0.0.1:{port}/")
    server.serve_forever()
