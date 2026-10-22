"""Launcher for EduTerm AI Web GUI.

Usage:
    python run_gui.py
    python run_gui.py --port 7860 --no-browser
"""

import argparse
import logging
import os
import secrets
import sys
import threading
import time
import warnings
import webbrowser
from pathlib import Path

# Add experiments folder to path
EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

# The local project key is the one the user edits for this app. Windows may also
# provide a stale EVREN_API_KEY inherited by the launcher, which otherwise silently
# takes precedence over experiments/.env in evren_llm.py.
project_env = EXPERIMENTS_DIR / ".env"
if project_env.exists():
    for env_line in project_env.read_text(encoding="utf-8").splitlines():
        env_line = env_line.strip()
        if env_line.startswith("EVREN_API_KEY="):
            project_key = env_line.split("=", 1)[1].strip().strip('"').strip("'")
            if project_key:
                os.environ["EVREN_API_KEY"] = project_key
            break

# Suppress unauthenticated HF Hub warning and noisy framework deprecations
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", message=r".*triton not found.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
try:
    from transformers.utils import logging as tf_logging
    tf_logging.set_verbosity_error()
except Exception:
    pass

import uvicorn
from gui.server import app


def open_browser(url: str, delay: float = 1.2):
    time.sleep(delay)
    print(f"Opening browser at: {url}")
    webbrowser.open(url)


def is_port_in_use(host: str, port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def is_eduterm_healthy(host: str, port: int, token: str = None) -> bool:
    import urllib.request
    endpoints = ["/api/health", "/api/active-job"]
    for ep in endpoints:
        try:
            url = f"http://{host}:{port}{ep}"
            if token:
                url += f"?token={token}"
            req = urllib.request.Request(url, headers={"User-Agent": "EduTerm-Probe"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            continue
    return False


def find_next_available_port(host: str, start_port: int, max_trials: int = 50) -> int:
    import socket
    for p in range(start_port, start_port + max_trials):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"Could not bind to any port between {start_port} and {start_port + max_trials}")


def save_active_server_info(host: str, port: int, launch_url: str):
    import json
    data_dir = EXPERIMENTS_DIR / "gui" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    info_file = data_dir / "active_server.json"
    try:
        info_file.write_text(
            json.dumps({
                "host": host,
                "port": port,
                "url": launch_url,
                "pid": os.getpid(),
                "time": time.time(),
            }, indent=2),
            encoding="utf-8"
        )
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="EduTerm AI Web GUI Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--token", default=None, help="Optional authentication token (required if exposed beyond localhost)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--force", action="store_true", help="Force starting server on next free port even if default port is occupied")
    args = parser.parse_args()

    # If bound beyond localhost without a token, generate a random security token
    token = args.token or os.environ.get("EDUTERM_AUTH_TOKEN")
    if not token and args.host not in ("127.0.0.1", "localhost"):
        token = secrets.token_hex(16)
        print(f"[SECURITY] Server bound to external host {args.host}. Generated auth token: {token}")

    if token:
        os.environ["EDUTERM_AUTH_TOKEN"] = token

    target_port = args.port

    if is_port_in_use(args.host, target_port):
        if not args.force and is_eduterm_healthy(args.host, target_port, token):
            url = f"http://{args.host}:{target_port}"
            launch_url = f"{url}?token={token}" if token else url
            print("=" * 65)
            print(f"  EduTerm AI Studio is ALREADY running and healthy at {launch_url}")
            print(f"  Models are already loaded in GPU memory! Opening browser...")
            print("=" * 65)
            if not args.no_browser:
                webbrowser.open(launch_url)
            return

        # Port is in use by non-responsive process or --force requested: find next open port
        print(f"[PORT CHECK] Port {target_port} is busy or unreachable. Finding next available port...")
        new_port = find_next_available_port(args.host, target_port + 1)
        print(f"[PORT FAILOVER] Switched from port {target_port} to available port {new_port}!")
        target_port = new_port

    url = f"http://{args.host}:{target_port}"
    launch_url = f"{url}?token={token}" if token else url
    save_active_server_info(args.host, target_port, launch_url)

    print("=" * 65)
    print(f"  EduTerm AI — Speech-to-Terminology Studio")
    print(f"  Local URL: {launch_url}")
    print("=" * 65)

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(launch_url,), daemon=True).start()

    uvicorn.run(app, host=args.host, port=target_port, log_level="info")


if __name__ == "__main__":
    main()
