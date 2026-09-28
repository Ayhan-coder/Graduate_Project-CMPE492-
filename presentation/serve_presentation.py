#!/usr/bin/env python3
"""EduTerm AI - Graduation Thesis Defense Presentation Server.

Starts a local zero-dependency HTTP server for the interactive presentation
and opens the presentation in the default browser.
"""

from __future__ import annotations

import functools
import http.server
import os
import socket
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

PRESENTATION_DIR = Path(__file__).resolve().parent


def find_free_port(start_port: int = 8080, max_attempts: int = 50) -> int:
    """Find an open TCP port on localhost."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found in range.")


def main():
    port = find_free_port(8080)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PRESENTATION_DIR))

    # Enable SO_REUSEADDR
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/index.html"
        print("=" * 66)
        print("  EduTerm AI — Bitirme Projesi Savunma Sunumu (Defense Deck)")
        print(f"  Yerel URL: {url}")
        print("=" * 66)
        print("  Klavye Kısayolları:")
        print("    ▶ Sağ Ok / Boşluk (Space) : Sonraki Slayt")
        print("    ◀ Sol Ok / Geri (Back)    : Önceki Slayt")
        print("    🎤 P Tuşu                  : Konuşmacı Modu (Presenter View)")
        print("    📑 O Tuşu                  : Tüm Slaytlar (Overview Grid)")
        print("    ⛶ F Tuşu                  : Tam Ekran (Fullscreen)")
        print("=" * 66)
        print("  Durdurmak için Ctrl+C tuşlarına basabilirsiniz.")
        print("=" * 66)

        # Open browser in a separate thread
        def open_browser():
            time.sleep(0.6)
            webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[BİLGİ] Sunucu kapatıldı.")


if __name__ == "__main__":
    main()
