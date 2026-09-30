#!/usr/bin/env python3
"""
Exoplanet Prototype — Local server + NASA TAP proxy
All data processing happens in the browser. This script only:
  1. Serves the HTML file
  2. Forwards /nasa-proxy?url=... requests to NASA (bypasses browser CORS restriction)

Usage:
    python server.py
    Then open: http://localhost:8000/exoplanet_prototype.html
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.request, urllib.parse, os, sys

PORT = 8000
ALLOWED_HOST = "exoplanetarchive.ipac.caltech.edu"


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/nasa-proxy"):
            self._proxy()
        else:
            super().do_GET()

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _proxy(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        target = params.get("url", [""])[0]

        if not target:
            self.send_error(400, "Missing ?url= parameter")
            return
        if ALLOWED_HOST not in target:
            self.send_error(403, f"Only {ALLOWED_HOST} is proxied")
            return

        try:
            req = urllib.request.Request(
                target,
                headers={"User-Agent": "ExoplanetPrototype/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(502, f"Upstream error: {e}")

    def log_message(self, fmt, *args):
        # Only log proxy requests, suppress static file noise
        if "/nasa-proxy" in (args[0] if args else ""):
            print(f"  [proxy] {args[0][:80]}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"\n  🔭  Exoplanet Prototype Server")
    print(f"  ─────────────────────────────────────────")
    print(f"  Open → http://localhost:{PORT}/exoplanet_prototype.html")
    print(f"  Stop → Ctrl+C\n")
    try:
        HTTPServer(("", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
