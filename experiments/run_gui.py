"""Launcher for EduTerm AI Web GUI.

Usage:
    python run_gui.py
    python run_gui.py --port 7860 --no-browser
"""

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Add experiments folder to path
EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import uvicorn
from gui.server import app


def open_browser(url: str, delay: float = 1.2):
    time.sleep(delay)
    print(f"Opening browser at: {url}")
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="EduTerm AI Web GUI Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print("=" * 65)
    print(f"  EduTerm AI — Speech-to-Terminology Studio")
    print(f"  Local URL: {url}")
    print("=" * 65)

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
