# -*- coding: utf-8 -*-
"""
EduTerm CMPE 492 - Human Student Evaluation Collector Server
Zero external dependencies (Python stdlib).

Usage:
    python collector_server.py --port 8765
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "outputs" / "human_eval_submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "submissions_summary.csv"


class EvalHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "service": "eduterm-human-eval"})
            return
        if self.path == "/stats":
            files = sorted(OUT_DIR.glob("sub_*.json"))
            self._send_json(200, {"total_submissions": len(files), "files": [f.name for f in files[-20:]]})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/submit":
            self._send_json(404, {"error": "Not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": f"Invalid JSON: {exc}"})
            return

        pid = str(data.get("participant_id", "anon")).strip() or "anon"
        safe_pid = "".join(c for c in pid if c.isalnum() or c in ("-", "_"))[:24] or "anon"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = OUT_DIR / f"sub_{ts}_{safe_pid}_{int(time.time() % 1000)}.json"
        data["received_at"] = datetime.now().isoformat()
        fname.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        ab_items = data.get("ab_evaluations", [])
        dpo_wins = sum(1 for x in ab_items if x.get("preferred_model") == "onpolicy_dpo_v2")
        hyb_wins = sum(1 for x in ab_items if x.get("preferred_model") == "task_separated_hybrid")
        ties = sum(1 for x in ab_items if x.get("preferred_model") == "tie")

        cefr_items = data.get("cefr_evaluations", [])
        avg_clarity = round(
            sum(float(x.get("clarity_score", 0)) for x in cefr_items) / max(len(cefr_items), 1), 2
        )
        avg_purity = round(
            sum(float(x.get("turkish_score", 0)) for x in cefr_items) / max(len(cefr_items), 1), 2
        )

        write_header = not CSV_PATH.exists()
        with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "received_at", "participant_id", "department", "cefr_level",
                    "avg_clarity", "avg_turkish", "dpo_v2_wins", "hybrid_wins", "ties", "file"
                ])
            writer.writerow([
                data["received_at"], pid, data.get("department", ""), data.get("self_level", ""),
                avg_clarity, avg_purity, dpo_wins, hyb_wins, ties, fname.name,
            ])

        print(f"[SAVED] {fname.name} | dpo_v2={dpo_wins} hybrid={hyb_wins} tie={ties} | clarity={avg_clarity}")
        self._send_json(200, {"ok": True, "saved_file": fname.name})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), EvalHandler)
    print(f"EduTerm Human Eval Collector listening on http://{args.host}:{args.port}")
    print(f"Saving submissions to: {OUT_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
