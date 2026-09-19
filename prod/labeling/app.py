"""Blind labeling server. Serves narrative + class definitions only.

    python -m prod.labeling.app
    # then open http://127.0.0.1:8765

Sealed NHTSA/model files are not on any route.
"""
from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from prod.paths import PROD_OUT, ensure_prod_out

HERE = Path(__file__).resolve().parent
QUEUE = PROD_OUT / "gold_current_v1_queue.json"
LABELS = PROD_OUT / "gold_current_v1_first_pass.jsonl"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def log_message(self, fmt, *args):
        print(f"[label] {self.address_string()} {fmt % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/queue":
            return self._send_json_file(QUEUE)
        if parsed.path == "/api/progress":
            return self._send_json(self._progress())
        if parsed.path in ("/", "/index.html"):
            self.path = "/index.html"
            return SimpleHTTPRequestHandler.do_GET(self)
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/label":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        cmplid = str(body.get("cmplid", ""))
        accept = body.get("accept") or []
        if not cmplid or not isinstance(accept, list) or not accept:
            return self._send_json({"ok": False, "error": "cmplid and non-empty accept required"}, 400)
        record = {
            "cmplid": cmplid,
            "accept": [str(x) for x in accept],
            "primary": str(accept[0]),
            "note": body.get("note") or None,
            "pass": "first_blind",
        }
        ensure_prod_out()
        existing = self._load_labels()
        existing = [r for r in existing if r["cmplid"] != cmplid]
        existing.append(record)
        with LABELS.open("w", encoding="utf-8") as f:
            for r in existing:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return self._send_json({"ok": True, "progress": self._progress()})

    def _load_labels(self):
        if not LABELS.exists():
            return []
        rows = []
        for line in LABELS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def _progress(self):
        queue_n = 0
        if QUEUE.exists():
            queue_n = len(json.loads(QUEUE.read_text(encoding="utf-8")).get("rows", []))
        done_ids = {r["cmplid"] for r in self._load_labels()}
        return {"done": len(done_ids), "total": queue_n, "done_ids": sorted(done_ids)}

    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json_file(self, path: Path):
        if not path.exists():
            return self._send_json({"error": f"missing {path.name}; run python -m prod.sample_current"}, 404)
        return self._send_json(json.loads(path.read_text(encoding="utf-8")))


def main():
    ensure_prod_out()
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Blind labeling UI: http://127.0.0.1:8765")
    print(f"Queue:  {QUEUE}")
    print(f"Labels: {LABELS}")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
