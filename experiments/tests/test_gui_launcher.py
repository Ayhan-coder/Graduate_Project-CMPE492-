"""Unit tests for the EduTerm AI GUI launcher script in experiments/run_gui.py."""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_gui


class LauncherPortAndHealthTests(unittest.TestCase):
    @patch("socket.socket")
    def test_is_port_in_use_true_and_false(self, mock_socket_cls):
        mock_socket = MagicMock()
        mock_socket_cls.return_value.__enter__.return_value = mock_socket

        # 0 means connection succeeded -> port is in use
        mock_socket.connect_ex.return_value = 0
        self.assertTrue(run_gui.is_port_in_use("127.0.0.1", 7860))

        # Non-zero means connection failed -> port is free
        mock_socket.connect_ex.return_value = 10061
        self.assertFalse(run_gui.is_port_in_use("127.0.0.1", 7860))

    @patch("urllib.request.urlopen")
    def test_is_eduterm_healthy_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        # Without token
        self.assertTrue(run_gui.is_eduterm_healthy("127.0.0.1", 7860))

        # With token
        self.assertTrue(run_gui.is_eduterm_healthy("127.0.0.1", 7860, token="secret123"))

    @patch("urllib.request.urlopen", side_effect=OSError("Connection refused"))
    def test_is_eduterm_healthy_failure(self, mock_urlopen):
        self.assertFalse(run_gui.is_eduterm_healthy("127.0.0.1", 7860))

    @patch("socket.socket")
    def test_find_next_available_port_finds_first_free(self, mock_socket_cls):
        mock_socket = MagicMock()
        mock_socket_cls.return_value.__enter__.return_value = mock_socket

        # Port 7860 throws OSError (in use), port 7861 succeeds
        def bind_side_effect(addr):
            host, port = addr
            if port == 7860:
                raise OSError("Address already in use")
            return None

        mock_socket.bind.side_effect = bind_side_effect
        found_port = run_gui.find_next_available_port("127.0.0.1", 7860, max_trials=5)
        self.assertEqual(found_port, 7861)

    @patch("socket.socket")
    def test_find_next_available_port_exhaustion_raises(self, mock_socket_cls):
        mock_socket = MagicMock()
        mock_socket_cls.return_value.__enter__.return_value = mock_socket
        mock_socket.bind.side_effect = OSError("Address in use")

        with self.assertRaises(RuntimeError) as ctx:
            run_gui.find_next_available_port("127.0.0.1", 7860, max_trials=3)
        self.assertIn("Could not bind to any port", str(ctx.exception))


class LauncherHelperAndMainTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.patcher = patch.object(run_gui, "EXPERIMENTS_DIR", Path(self.temp_dir))
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_active_server_info_creates_json(self):
        run_gui.save_active_server_info("127.0.0.1", 7860, "http://127.0.0.1:7860")
        info_file = Path(self.temp_dir) / "gui" / "data" / "active_server.json"
        self.assertTrue(info_file.exists())
        data = json.loads(info_file.read_text(encoding="utf-8"))
        self.assertEqual(data["host"], "127.0.0.1")
        self.assertEqual(data["port"], 7860)
        self.assertEqual(data["url"], "http://127.0.0.1:7860")
        self.assertEqual(data["pid"], os.getpid())

    @patch("webbrowser.open")
    @patch("time.sleep")
    def test_open_browser(self, mock_sleep, mock_browser):
        run_gui.open_browser("http://127.0.0.1:7860", delay=0.01)
        mock_sleep.assert_called_once_with(0.01)
        mock_browser.assert_called_once_with("http://127.0.0.1:7860")

    @patch("uvicorn.run")
    @patch("run_gui.is_port_in_use", return_value=False)
    def test_main_starts_uvicorn_when_port_free(self, mock_port_in_use, mock_uvicorn):
        with patch.object(sys, "argv", ["run_gui.py", "--port", "8000", "--no-browser"]):
            run_gui.main()
            mock_uvicorn.assert_called_once()
            _, kwargs = mock_uvicorn.call_args
            self.assertEqual(kwargs["port"], 8000)
            self.assertEqual(kwargs["host"], "127.0.0.1")

    @patch("uvicorn.run")
    @patch("webbrowser.open")
    @patch("run_gui.is_eduterm_healthy", return_value=True)
    @patch("run_gui.is_port_in_use", return_value=True)
    def test_main_reuses_healthy_running_server(self, mock_port_in_use, mock_healthy, mock_browser, mock_uvicorn):
        with patch.object(sys, "argv", ["run_gui.py", "--port", "7860"]):
            run_gui.main()
            # Must NOT start another uvicorn instance
            mock_uvicorn.assert_not_called()
            # Must open existing URL
            mock_browser.assert_called_once_with("http://127.0.0.1:7860")

    @patch("uvicorn.run")
    @patch("run_gui.find_next_available_port", return_value=7861)
    @patch("run_gui.is_eduterm_healthy", return_value=False)
    @patch("run_gui.is_port_in_use", return_value=True)
    def test_main_failover_when_port_busy_and_unhealthy(self, mock_port_in_use, mock_healthy, mock_find_port, mock_uvicorn):
        with patch.object(sys, "argv", ["run_gui.py", "--port", "7860", "--no-browser"]):
            run_gui.main()
            mock_find_port.assert_called_once_with("127.0.0.1", 7861)
            mock_uvicorn.assert_called_once()
            _, kwargs = mock_uvicorn.call_args
            self.assertEqual(kwargs["port"], 7861)

    @patch("uvicorn.run")
    @patch("run_gui.is_port_in_use", return_value=False)
    def test_main_generates_token_for_external_bind(self, mock_port_in_use, mock_uvicorn):
        with patch.object(sys, "argv", ["run_gui.py", "--host", "0.0.0.0", "--no-browser"]):
            orig_token = os.environ.get("EDUTERM_AUTH_TOKEN")
            try:
                run_gui.main()
                token = os.environ.get("EDUTERM_AUTH_TOKEN")
                self.assertIsNotNone(token)
                self.assertEqual(len(token), 32)  # 16 bytes hex
            finally:
                if orig_token is not None:
                    os.environ["EDUTERM_AUTH_TOKEN"] = orig_token
                else:
                    os.environ.pop("EDUTERM_AUTH_TOKEN", None)


if __name__ == "__main__":
    unittest.main()
