#!/usr/bin/env python3
"""Serve the lfm2d web demos and proxy their calls to a daemon.

    python3 demo/web/server.py --upstream http://lfm2d-host:8088

Binds the host's tailnet address by default (`tailscale ip -4`) and refuses a
wildcard bind: the demos are for the tailnet, not the LAN. The daemon sends no
CORS headers, so the pages call `/api/<daemon path>` here and this forwards
only the routes in ALLOWED. Request bodies are never logged.

A server-sent-event answer (`/v1/chat` with `stream: true`) is relayed as it
arrives, and a browser that hangs up hangs up the upstream: the daemon cancels
a chat turn when its client goes, and this proxy is that client.

The council (`/council`, council.py) keeps its state here rather than in the
page, so its routes are `/council/api/*`, answered by that module, which calls
the daemon itself. It starts on the first request there.
"""
import argparse
import http.client
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import council

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 1 << 20
ALLOWED = {
    ("GET", "/v1/adjudicator"),
    ("GET", "/v1/models"),
    ("GET", "/v1/opinion/specs"),
    ("POST", "/v1/opinion/specs"),
    ("POST", "/v1/opinion"),
    ("POST", "/v1/adjudicate"),
    ("POST", "/v1/chat"),
    ("POST", "/v1/probe"),
    ("POST", "/v1/tokenize"),
    ("POST", "/embed"),
}
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript",
         ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".md": "text/markdown; charset=utf-8"}


def resolve_host(host, run=subprocess.run):
    """The explicit host, else the tailnet IPv4; never a wildcard."""
    if host is not None:
        if host in ("", "0.0.0.0", "::"):
            sys.exit("server.py: refusing a wildcard bind; pass a tailnet or loopback --host")
        return host
    try:
        out = run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        sys.exit(f"server.py: no --host and `tailscale ip -4` failed ({e})")
    lines = out.stdout.split() if out.returncode == 0 else []
    if not lines:
        sys.exit("server.py: no --host and no tailnet address; pass --host")
    return lines[0]


def make_server(host, port, upstream):
    upstream = upstream.rstrip("/")
    board = council.Council(council.Daemon(upstream))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

        def send(self, status, body, ctype="application/json"):
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def error(self, status, message):
            self.send(status, json.dumps({"error": {"message": message}}).encode())

        def proxy(self):
            path = urllib.parse.urlsplit(self.path).path[len("/api"):]
            if (self.command, path) not in ALLOWED:
                return self.error(404, f"not proxied: {self.command} {path}")
            n = int(self.headers.get("content-length") or 0)
            if n > MAX_BODY:
                return self.error(413, "body too large")
            body = self.rfile.read(n) if self.command == "POST" else None
            req = urllib.request.Request(upstream + path, body, method=self.command,
                                         headers={"content-type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    ctype = r.headers.get("content-type", "application/json")
                    if ctype.startswith("text/event-stream"):
                        return self.relay(r, ctype)
                    self.send(r.status, r.read(), ctype)
            except urllib.error.HTTPError as e:
                self.send(e.code, e.read(), e.headers.get("content-type", "application/json"))
            except (urllib.error.URLError, OSError) as e:
                self.error(502, f"upstream unreachable: {e}")

        def relay(self, r, ctype):
            """Pass an event stream through chunk by chunk. No length: this is
            HTTP/1.0, so closing the connection ends the body. The socket
            timeout bounds the wait between events (the daemon's keep-alive
            comes every 15 s), not the whole stream."""
            self.send_response(r.status)
            self.send_header("content-type", ctype)
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.close_connection = True
            while True:
                last = False
                try:
                    chunk = r.read1(1 << 16)
                except (OSError, http.client.HTTPException) as e:
                    # The upstream broke after the 200 went out, so an HTTP
                    # error can no longer be sent: end the stream the way the
                    # daemon ends one that fails late, with an `error` event.
                    body = json.dumps({"status": 502, "error": {
                        "type": "upstream", "message": f"upstream stream broke: {e!r}"}})
                    chunk, last = f"\n\nevent: error\ndata: {body}\n\n".encode(), True
                if not chunk:
                    return
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # The reader left: returning closes the upstream (the
                    # caller's `with`), which cancels the turn upstream.
                    return
                if last:
                    return

        def static(self):
            raw = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
            name = "index" if raw == "/" else raw.strip("/")
            if not name or "/" in name or name.startswith("."):
                return self.error(404, "not found")
            target = (STATIC / name).resolve()
            if target.suffix == "":
                target = target.with_suffix(".html")
            if not target.is_relative_to(STATIC) or not target.is_file():
                return self.error(404, "not found")
            self.send(200, target.read_bytes(), TYPES.get(target.suffix, "application/octet-stream"))

        def do_GET(self):
            if self.path.startswith("/api/"):
                return self.proxy()
            if self.path.startswith("/council/api/"):
                return board.handle(self, "GET")
            self.static()

        def do_POST(self):
            if self.path.startswith("/api/"):
                return self.proxy()
            if self.path.startswith("/council/api/"):
                return board.handle(self, "POST")
            self.error(404, "not found")

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.council = board
    return httpd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--upstream", default=os.environ.get("LFM2D_URL"),
                    help="daemon base URL (or LFM2D_URL)")
    ap.add_argument("--host", default=None, help="bind address (default: tailnet IPv4)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if not a.upstream:
        ap.error("--upstream (or LFM2D_URL) is required")
    host = resolve_host(a.host)
    httpd = make_server(host, a.port, a.upstream)
    print(f"demos on http://{host}:{a.port}/  ->  {a.upstream}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
