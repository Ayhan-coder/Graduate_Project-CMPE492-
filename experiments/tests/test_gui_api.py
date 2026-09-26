"""Unit and integration tests for GUI server endpoints, background workers, and business logic."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from gui.pipeline import (
    detect_lang,
    download_youtube_audio,
    extract_youtube_id,
    get_youtube_info,
    parse_timestamp_segments,
)
from gui.server import (
    JOBS,
    JOB_TTL_S,
    TEMP_DIR,
    _push_event,
    _run_pipeline_job,
    align_terms_with_segments,
    app,
    cleanup_expired_jobs,
    cleanup_temp_dir,
    get_allowed_model_ids,
)

client = TestClient(app)


class PipelineHelperTests(unittest.TestCase):
    def test_detect_lang_needs_several_turkish_letters(self):
        self.assertEqual(detect_lang("Hello world project management"), "en")
        self.assertEqual(detect_lang("Proje izleme ve kontrolü"), "tr")

    def test_parse_timestamp_segments(self):
        raw = "<|0.00|> Hello world <|2.40|> Next sentence <|5.00|>"
        segs = parse_timestamp_segments(raw, "Hello world Next sentence")
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["start"], 0.0)
        self.assertEqual(segs[0]["text"], "Hello world")
        self.assertEqual(segs[1]["start"], 2.4)
        self.assertEqual(segs[1]["end"], 5.0)

    def test_parse_timestamp_fallback_sentences(self):
        segs = parse_timestamp_segments("", "First sentence. Second sentence!")
        self.assertEqual(len(segs), 2)
        self.assertIsNone(segs[0]["start"])
        self.assertEqual(segs[0]["text"], "First sentence.")

    def test_extract_youtube_id(self):
        cases = [
            ("https://www.youtube.com/watch?v=rz5JCvMlCMs", "rz5JCvMlCMs"),
            ("https://youtu.be/rz5JCvMlCMs", "rz5JCvMlCMs"),
            ("https://www.youtube.com/watch?v=rz5JCvMlCMs&t=42s", "rz5JCvMlCMs"),
            ("https://www.youtube.com/shorts/2IK3DFHRFfw", "2IK3DFHRFfw"),
            ("https://www.youtube.com/embed/BaH04m_fS1c", "BaH04m_fS1c"),
            ("rz5JCvMlCMs", "rz5JCvMlCMs"),
            ("  https://youtu.be/yzsHZVU_Y7w  ", "yzsHZVU_Y7w"),
            ("not-a-youtube-url", None),
            ("", None),
        ]
        for inp, expected in cases:
            self.assertEqual(extract_youtube_id(inp), expected, f"Failed on input: {inp}")

    def test_get_youtube_info_local_cache(self):
        info = get_youtube_info("https://www.youtube.com/watch?v=rz5JCvMlCMs")
        self.assertEqual(info["id"], "rz5JCvMlCMs")
        self.assertIn("PROJE YÖNETİMİ", info["title"])
        self.assertGreater(info["duration"], 0)
        self.assertTrue(info["thumbnail"].startswith("https://"))

    def test_download_youtube_audio_local_slice(self):
        with tempfile.TemporaryDirectory() as td:
            out_wav = Path(td) / "slice.wav"
            dur = download_youtube_audio("https://www.youtube.com/watch?v=rz5JCvMlCMs", out_wav, start_s=0, duration_s=2)
            self.assertTrue(out_wav.exists())
            self.assertGreaterEqual(dur, 1.8)
            self.assertLessEqual(dur, 2.2)


class ServerLogicTests(unittest.TestCase):
    def test_align_terms_exact_match(self):
        terms = [{"term": "sprint", "explanation": "A short time-boxed period"}]
        segments = [
            {"start": 12.5, "end": 15.0, "text": "In the current sprint we complete tasks."},
        ]
        aligned = align_terms_with_segments(terms, segments)
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0]["start_s"], 12.5)
        self.assertEqual(aligned[0]["end_s"], 15.0)
        self.assertEqual(aligned[0]["timestamp"], "00:12")

    def test_align_terms_token_overlap(self):
        terms = [{"term": "agile software development", "explanation": "Iterative approach"}]
        segments = [
            {"start": 75.2, "end": 80.0, "text": "We adopt agile software development methods today."},
        ]
        aligned = align_terms_with_segments(terms, segments)
        self.assertEqual(aligned[0]["start_s"], 75.2)
        self.assertEqual(aligned[0]["timestamp"], "01:15")

    def test_align_terms_unmatched(self):
        terms = [
            {"term": "unrelated term", "explanation": "None"},
            {"term": "", "explanation": "Empty"},
        ]
        segments = [{"start": 5.0, "end": 10.0, "text": "Hello world."}]
        aligned = align_terms_with_segments(terms, segments)
        self.assertIsNone(aligned[0]["start_s"])
        self.assertIsNone(aligned[0]["timestamp"])
        self.assertIsNone(aligned[1]["start_s"])
        self.assertIsNone(aligned[1]["timestamp"])

    def test_push_event(self):
        test_job = "job-event-test"
        JOBS[test_job] = {"events": []}
        _push_event(test_job, "test_ev", {"msg": "hello"})
        self.assertEqual(len(JOBS[test_job]["events"]), 1)
        self.assertEqual(JOBS[test_job]["events"][0]["event"], "test_ev")
        self.assertEqual(JOBS[test_job]["events"][0]["data"]["msg"], "hello")
        del JOBS[test_job]

    @patch("gui.server.transcribe_audio")
    @patch("gui.server.extract_terms")
    def test_run_pipeline_job_success(self, mock_extract, mock_transcribe):
        mock_transcribe.return_value = {
            "text": "Bugün yazılım mühendisliği projelerini konuşuyoruz.",
            "segments": [{"start": 0.0, "end": 3.5, "text": "Bugün yazılım mühendisliği projelerini konuşuyoruz."}],
            "duration_s": 3.5,
            "latency_s": 0.1,
            "rtf": 0.03,
            "model": "whisper-test",
        }
        mock_extract.return_value = {
            "terms": [{"term": "yazılım mühendisliği", "explanation": "Software engineering"}],
            "n_terms": 1,
            "latency_s": 0.2,
            "raw": "...",
            "model": "qwen-test",
        }

        job_id = "test-pipe-job-1"
        JOBS[job_id] = {"status": "queued", "events": [], "result": None}
        wav_dummy = Path("dummy.wav")

        _run_pipeline_job(
            job_id=job_id,
            audio_wav=wav_dummy,
            asr_model="whisper-test",
            lang="tr",
            llm_model="qwen-test",
            video_meta={"id": "vid123", "title": "Test Video"},
        )

        self.assertEqual(JOBS[job_id]["status"], "completed")
        res = JOBS[job_id]["result"]
        self.assertIsNotNone(res)
        self.assertEqual(res["lang"], "tr")
        self.assertEqual(res["llm"]["n_terms"], 1)
        self.assertEqual(res["llm"]["terms"][0]["timestamp"], "00:00")
        del JOBS[job_id]

    @patch("gui.server.logger.error")
    @patch("gui.server.transcribe_audio", side_effect=RuntimeError("Whisper failed with /opt/secret/path"))
    def test_run_pipeline_job_failure(self, mock_transcribe, mock_log_err):
        job_id = "test-pipe-fail-1"
        JOBS[job_id] = {"status": "queued", "events": [], "result": None}

        _run_pipeline_job(
            job_id=job_id,
            audio_wav=Path("dummy.wav"),
            asr_model="whisper-test",
            lang="en",
            llm_model="qwen-test",
        )

        self.assertEqual(JOBS[job_id]["status"], "error")
        event_types = [e["event"] for e in JOBS[job_id]["events"]]
        self.assertIn("job_error", event_types)
        job_err_event = next(e for e in JOBS[job_id]["events"] if e["event"] == "job_error")
        # Ensure internal error paths are not leaked to client
        self.assertEqual(job_err_event["data"]["message"], "Processing failed due to an internal server error.")
        self.assertNotIn("/opt/secret", job_err_event["data"]["message"])
        mock_log_err.assert_called()
        del JOBS[job_id]

    @patch("gui.server.clear_gpu_memory")
    @patch("gui.server.transcribe_audio")
    def test_run_pipeline_job_cancelled(self, mock_transcribe, mock_clear_gpu):
        cancel_ev = threading.Event()
        cancel_ev.set()
        job_id = "test-pipe-cancel-1"
        JOBS[job_id] = {"status": "queued", "events": [], "result": None, "cancel_event": cancel_ev}

        _run_pipeline_job(
            job_id=job_id,
            audio_wav=Path("dummy.wav"),
            asr_model="whisper-test",
            lang="en",
            llm_model="qwen-test",
            cancel_event=cancel_ev,
        )

        self.assertEqual(JOBS[job_id]["status"], "cancelled")
        event_types = [e["event"] for e in JOBS[job_id]["events"]]
        self.assertIn("cancelled", event_types)
        mock_clear_gpu.assert_called()
        del JOBS[job_id]


class GuiApiTests(unittest.TestCase):
    def test_health(self):
        res = client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["ok"])

    def test_youtube_info_endpoint(self):
        res = client.post("/api/youtube-info", json={"url": "https://www.youtube.com/watch?v=BaH04m_fS1c"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["id"], "BaH04m_fS1c")
        self.assertIn("Agile", data["title"])
        self.assertIn("thumbnail", data)

    def test_youtube_info_endpoint_empty_url(self):
        res = client.post("/api/youtube-info", json={"url": "   "})
        self.assertEqual(res.status_code, 400)

    def test_youtube_info_endpoint_ssrf_rejected(self):
        ssrf_payloads = [
            "http://127.0.0.1:8000/internal-admin",
            "http://169.254.169.254/latest/meta-data/",
            "file:///etc/passwd",
            "https://attacker.com/exploit",
            "not-a-youtube-url",
        ]
        for payload in ssrf_payloads:
            res = client.post("/api/youtube-info", json={"url": payload})
            self.assertEqual(res.status_code, 400, f"Failed to reject SSRF payload: {payload}")

    @patch("gui.server.logger.error")
    @patch("gui.server.get_youtube_info", side_effect=RuntimeError("internal stack trace with /secret/local/ffmpeg"))
    def test_youtube_info_endpoint_sanitizes_internal_error(self, mock_get_info, mock_log_err):
        res = client.post("/api/youtube-info", json={"url": "https://www.youtube.com/watch?v=rz5JCvMlCMs"})
        self.assertEqual(res.status_code, 400)
        self.assertNotIn("/secret/local", res.text)
        self.assertIn("Failed to retrieve YouTube video metadata", res.json()["detail"])
        mock_log_err.assert_called()

    def test_process_youtube_endpoint_ssrf_rejected(self):
        ssrf_payloads = [
            "http://127.0.0.1:8000/internal-admin",
            "http://169.254.169.254/latest/meta-data/",
            "file:///etc/passwd",
            "https://attacker.com/exploit",
        ]
        for payload in ssrf_payloads:
            res = client.post("/api/process-youtube", json={"url": payload})
            self.assertEqual(res.status_code, 400, f"Failed to reject SSRF payload in process-youtube: {payload}")

    def test_export_missing_job(self):
        self.assertEqual(client.get("/api/export-json/does-not-exist").status_code, 404)
        self.assertEqual(client.get("/api/export-csv/does-not-exist").status_code, 404)

    def test_export_csv_valid_job(self):
        job_id = "test-export-csv-job"
        JOBS[job_id] = {
            "status": "completed",
            "result": {
                "video_meta": {"id": "test_video"},
                "asr": {"model": "whisper-turbo"},
                "llm": {
                    "model": "qwen-4b",
                    "terms": [
                        {"term": "Sprint", "timestamp": "01:23", "explanation": "Timebox"},
                        {"term": "Backlog", "timestamp": None, "explanation": "To-do list"},
                    ],
                },
                "lang": "en",
            },
        }
        res = client.get(f"/api/export-csv/{job_id}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "text/csv; charset=utf-8")
        self.assertIn(f"attachment; filename=terms_{job_id[:8]}.csv", res.headers.get("content-disposition", ""))

        reader = csv.DictReader(io.StringIO(res.text))
        rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["term"], "Sprint")
        self.assertEqual(rows[0]["timestamp"], "01:23")
        self.assertEqual(rows[0]["model"], "whisper-turbo|qwen-4b")
        self.assertEqual(rows[1]["term"], "Backlog")
        self.assertEqual(rows[1]["timestamp"], "")
        del JOBS[job_id]

    def test_export_json_valid_job(self):
        job_id = "test-export-json-job"
        sample_payload = {
            "job_id": job_id,
            "lang": "tr",
            "asr": {"text": "Örnek metin", "model": "whisper-large-v3"},
            "llm": {"terms": [{"term": "örnek"}], "model": "qwen3.5-4b"},
        }
        JOBS[job_id] = {"status": "completed", "result": sample_payload}
        res = client.get(f"/api/export-json/{job_id}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "application/json")
        self.assertIn(f"attachment; filename=eduterm_{job_id[:8]}.json", res.headers.get("content-disposition", ""))
        self.assertEqual(res.json(), sample_payload)
        del JOBS[job_id]

    def test_job_status(self):
        self.assertEqual(client.get("/api/job/does-not-exist").status_code, 404)
        JOBS["test-job-123"] = {"status": "processing", "result": None, "meta": {"title": "Test"}}
        res = client.get("/api/job/test-job-123")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "processing")
        self.assertEqual(data["meta"]["title"], "Test")
        del JOBS["test-job-123"]

    def test_homepage(self):
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("EduTerm", res.text)
        self.assertIn("diagLang", res.text)
        self.assertIn("EN-1: Agile Methodologies", res.text)

    def test_system_info(self):
        res = client.get("/api/system-info")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("cuda", data)
        self.assertIn("gpu", data)

    def test_models(self):
        res = client.get("/api/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(len(data["asr_models"]), 4)
        self.assertGreaterEqual(len(data["llm_models"]), 3)
        llm_ids = [m["id"] for m in data["llm_models"]]
        self.assertIn("Qwen/Qwen3-8B", llm_ids)
        self.assertIn("Qwen/Qwen3.5-9B", llm_ids)
        self.assertIn("Qwen/Qwen3.5-4B", llm_ids)
        self.assertIn("google/gemma-4-E4B-it", llm_ids)
        self.assertIn("meta-llama/Llama-3.1-8B-Instruct", llm_ids)
        default_llm = next(m for m in data["llm_models"] if m.get("default"))
        self.assertEqual(default_llm["id"], "Qwen/Qwen3.5-4B")

    def test_get_audio(self):
        # 404 when job does not exist
        self.assertEqual(client.get("/api/audio/missing-job").status_code, 404)

        # 404 when audio_file does not exist on disk
        job_id = "test-audio-job-nonexistent"
        JOBS[job_id] = {"audio_file": Path("nonexistent.wav")}
        self.assertEqual(client.get(f"/api/audio/{job_id}").status_code, 404)
        del JOBS[job_id]

        # 200 when file exists
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"RIFFdummywavdata")
            tmp_path = Path(tmp.name)

        try:
            job_id = "test-audio-job-ok"
            JOBS[job_id] = {"audio_file": tmp_path}
            res = client.get(f"/api/audio/{job_id}")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.headers["content-type"], "audio/wav")
            del JOBS[job_id]
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_stream_job_not_found(self):
        res = client.get("/api/stream/nonexistent-job")
        self.assertEqual(res.status_code, 404)

    def test_stream_job_events(self):
        job_id = "test-stream-job"
        JOBS[job_id] = {
            "status": "completed",
            "events": [
                {"event": "step", "data": {"step": "download"}},
                {"event": "done", "data": {"finished": True}},
            ],
        }
        res = client.get(f"/api/stream/{job_id}")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/event-stream", res.headers["content-type"])
        body = res.text
        self.assertIn("event: step", body)
        self.assertIn('"step": "download"', body)
        self.assertIn("event: done", body)
        del JOBS[job_id]

    @patch("threading.Thread")
    def test_process_youtube_endpoint(self, mock_thread):
        req_data = {
            "url": "https://www.youtube.com/watch?v=rz5JCvMlCMs",
            "start_s": 0.0,
            "duration_s": 30.0,
            "asr_model": "openai/whisper-large-v3-turbo",
            "lang": "tr",
            "llm_model": "Qwen/Qwen3.5-4B",
        }
        res = client.post("/api/process-youtube", json=req_data)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("job_id", data)
        self.assertIn("meta", data)
        job_id = data["job_id"]
        self.assertIn(job_id, JOBS)
        self.assertEqual(JOBS[job_id]["status"], "queued")
        # Ensure worker thread was started
        mock_thread.assert_called_once()
        del JOBS[job_id]

    @patch("threading.Thread")
    def test_process_upload_endpoint(self, mock_thread):
        dummy_content = b"fake audio content"
        files = {"file": ("lecture.mp3", dummy_content, "audio/mpeg")}
        data = {
            "start_s": "5.0",
            "duration_s": "10.0",
            "asr_model": "openai/whisper-large-v3-turbo",
            "lang": "en",
            "llm_model": "Qwen/Qwen3.5-4B",
        }
        res = client.post("/api/process-upload", data=data, files=files)
        self.assertEqual(res.status_code, 200)
        resp_data = res.json()
        self.assertIn("job_id", resp_data)
        job_id = resp_data["job_id"]
        self.assertIn(job_id, JOBS)
        self.assertEqual(JOBS[job_id]["status"], "queued")
        self.assertEqual(JOBS[job_id]["meta"]["title"], "lecture.mp3")
        mock_thread.assert_called_once()
        temp_upload = Path(__file__).resolve().parent.parent / "gui" / "temp" / f"upload_{job_id}_lecture.mp3"
        if temp_upload.exists():
            temp_upload.unlink()
        del JOBS[job_id]

    @patch("threading.Thread")
    def test_process_upload_path_traversal(self, mock_thread):
        traversal_names = [
            "../../evil.wav",
            "..\\..\\evil.mp3",
            "C:\\Windows\\System32\\evil.m4a",
            "/etc/evil.flac",
        ]
        temp_dir = Path(__file__).resolve().parent.parent / "gui" / "temp"
        for evil_name in traversal_names:
            files = {"file": (evil_name, b"dummy audio content", "audio/wav")}
            res = client.post("/api/process-upload", files=files)
            self.assertEqual(res.status_code, 200, f"Failed for {evil_name}")
            data = res.json()
            job_id = data["job_id"]
            # Must be sanitized to strict basename
            expected_name = Path(evil_name.replace("\\", "/")).name
            self.assertEqual(data["meta"]["title"], expected_name)
            # Verify file on disk is strictly inside temp_dir
            staged_file = temp_dir / f"upload_{job_id}_{expected_name}"
            self.assertTrue(staged_file.exists())
            self.assertTrue(staged_file.resolve().is_relative_to(temp_dir.resolve()))
            staged_file.unlink()
            del JOBS[job_id]

    def test_process_upload_unsupported_type(self):
        bad_files = ["malware.exe", "exploit.sh", "notes.txt", "audio.py", "payload.bin"]
        for name in bad_files:
            files = {"file": (name, b"dummy content", "application/octet-stream")}
            res = client.post("/api/process-upload", files=files)
            self.assertEqual(res.status_code, 400, f"Did not reject unsupported type: {name}")
            self.assertIn("Unsupported file type", res.json()["detail"])

    def test_process_upload_empty_file(self):
        files = {"file": ("empty.wav", b"", "audio/wav")}
        res = client.post("/api/process-upload", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("empty", res.json()["detail"].lower())

    @patch("gui.server.MAX_UPLOAD_BYTES", 20)  # Cap at 20 bytes for test
    def test_process_upload_size_limit_exceeded(self):
        oversized = b"A" * 100  # 100 bytes > 20 bytes limit
        files = {"file": ("toolarge.wav", oversized, "audio/wav")}
        res = client.post("/api/process-upload", files=files)
        self.assertEqual(res.status_code, 413)
        self.assertIn("exceeds the maximum allowed limit", res.json()["detail"])
        # Ensure partial file was unlinked and not left on disk
        temp_dir = Path(__file__).resolve().parent.parent / "gui" / "temp"
        matching = list(temp_dir.glob("*toolarge.wav"))
        self.assertEqual(matching, [], "Partial oversized upload was not cleaned up!")

    def test_get_allowed_model_ids(self):
        asr_ids, llm_ids = get_allowed_model_ids()
        self.assertIn("openai/whisper-large-v3-turbo", asr_ids)
        self.assertIn("openai/whisper-large-v3", asr_ids)
        self.assertIn("Qwen/Qwen3.5-4B", llm_ids)
        self.assertIn("Qwen/Qwen3-8B", llm_ids)

    def test_process_youtube_disallowed_models(self):
        valid_url = "https://www.youtube.com/watch?v=rz5JCvMlCMs"
        # Disallowed ASR
        res = client.post("/api/process-youtube", json={
            "url": valid_url,
            "asr_model": "hostile-org/evil-asr",
            "llm_model": "Qwen/Qwen3.5-4B",
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn("Disallowed ASR model", res.json()["detail"])

        # Disallowed LLM
        res = client.post("/api/process-youtube", json={
            "url": valid_url,
            "asr_model": "openai/whisper-large-v3-turbo",
            "llm_model": "hostile-org/evil-llm",
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn("Disallowed LLM model", res.json()["detail"])

    def test_process_upload_disallowed_models(self):
        files = {"file": ("test.wav", b"RIFFfakeaudio", "audio/wav")}
        # Disallowed ASR
        res = client.post("/api/process-upload", files=files, data={"asr_model": "hostile/evil-asr"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Disallowed ASR model", res.json()["detail"])

        # Disallowed LLM
        res = client.post("/api/process-upload", files=files, data={"llm_model": "hostile/evil-llm"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Disallowed LLM model", res.json()["detail"])

    def test_cors_headers_not_present(self):
        # Verify CORS headers are dropped and not allowing arbitrary origins
        headers = {"Origin": "https://malicious-website.com"}
        res = client.get("/api/health", headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertNotIn("access-control-allow-origin", res.headers)

    @patch("threading.Thread")
    def test_single_running_job_limit(self, mock_thread):
        active_job_id = "test-active-job-running"
        JOBS[active_job_id] = {"status": "processing"}
        try:
            # Attempting to start a YouTube job while another is running -> 429
            res_yt = client.post("/api/process-youtube", json={"url": "https://www.youtube.com/watch?v=rz5JCvMlCMs"})
            self.assertEqual(res_yt.status_code, 429)
            self.assertIn("Another speech-to-terminology job is currently running", res_yt.json()["detail"])

            # Attempting to start an Upload job while another is running -> 429
            files = {"file": ("test.wav", b"RIFFfakeaudio", "audio/wav")}
            res_up = client.post("/api/process-upload", files=files)
            self.assertEqual(res_up.status_code, 429)
            self.assertIn("Another speech-to-terminology job is currently running", res_up.json()["detail"])
        finally:
            del JOBS[active_job_id]

    def test_auth_token_enforcement(self):
        test_token = "secret-token-xyz-123"
        with patch.dict("os.environ", {"EDUTERM_AUTH_TOKEN": test_token}):
            # Missing token -> 401
            res_unauth = client.get("/api/health")
            self.assertEqual(res_unauth.status_code, 401)
            self.assertIn("Unauthorized", res_unauth.json()["detail"])

            # Invalid token -> 401
            res_bad = client.get("/api/health", headers={"X-Auth-Token": "wrong-token"})
            self.assertEqual(res_bad.status_code, 401)

            # Valid token via X-Auth-Token -> 200
            res_x_auth = client.get("/api/health", headers={"X-Auth-Token": test_token})
            self.assertEqual(res_x_auth.status_code, 200)

            # Valid token via Authorization: Bearer -> 200
            res_bearer = client.get("/api/health", headers={"Authorization": f"Bearer {test_token}"})
            self.assertEqual(res_bearer.status_code, 200)

            # Valid token via ?token= query param -> 200
            res_query = client.get(f"/api/health?token={test_token}")
            self.assertEqual(res_query.status_code, 200)

            # Non-API route (index page) should still be accessible without token so GUI loads
            res_page = client.get("/")
            self.assertEqual(res_page.status_code, 200)

    def test_active_job_endpoints(self):
        # When no job is active
        res = client.get("/api/active-job")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["active"])
        self.assertIsNone(res.json()["job"])

        # When a job is active
        active_id = "test-active-query-job"
        JOBS[active_id] = {"status": "processing", "meta": {"title": "Running Lecture"}}
        try:
            res_active = client.get("/api/active-job")
            self.assertEqual(res_active.status_code, 200)
            data = res_active.json()
            self.assertTrue(data["active"])
            self.assertEqual(data["job"]["job_id"], active_id)
            self.assertEqual(data["job"]["status"], "processing")
        finally:
            del JOBS[active_id]

    @patch("gui.server.clear_gpu_memory")
    def test_cancel_job_no_active_job(self, mock_clear_gpu):
        res = client.post("/api/cancel-job")
        self.assertEqual(res.status_code, 404)

    @patch("gui.server.clear_gpu_memory")
    def test_cancel_job_endpoint(self, mock_clear_gpu):
        cancel_ev = threading.Event()
        job_id = "test-cancel-active-job"
        JOBS[job_id] = {
            "status": "processing",
            "events": [],
            "cancel_event": cancel_ev,
        }
        try:
            res = client.post("/api/cancel-job")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "cancelled")
            self.assertEqual(data["job_id"], job_id)
            self.assertTrue(cancel_ev.is_set())
            self.assertEqual(JOBS[job_id]["status"], "cancelled")
            event_types = [e["event"] for e in JOBS[job_id]["events"]]
            self.assertIn("cancelled", event_types)
            mock_clear_gpu.assert_called()
        finally:
            del JOBS[job_id]

    @patch("gui.server.clear_gpu_memory")
    def test_cancel_job_by_id_endpoint(self, mock_clear_gpu):
        cancel_ev = threading.Event()
        job_id = "test-cancel-by-id-123"
        JOBS[job_id] = {
            "status": "queued",
            "events": [],
            "cancel_event": cancel_ev,
        }
        try:
            res = client.post(f"/api/cancel/{job_id}")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["status"], "cancelled")
            self.assertTrue(cancel_ev.is_set())
            self.assertEqual(JOBS[job_id]["status"], "cancelled")
            mock_clear_gpu.assert_called()
        finally:
            del JOBS[job_id]

    @patch("gui.server.extract_terms")
    @patch("gui.server.transcribe_audio")
    def test_run_pipeline_job_skips_llm_on_empty_asr(self, mock_transcribe, mock_extract):
        mock_transcribe.return_value = {
            "text": "   ",
            "segments": [],
            "duration_s": 15.0,
            "latency_s": 0.2,
            "rtf": 0.01,
            "model": "whisper-test",
        }

        job_id = "test-empty-asr-1"
        JOBS[job_id] = {"status": "queued", "events": [], "result": None}

        _run_pipeline_job(
            job_id=job_id,
            audio_wav=Path("dummy.wav"),
            asr_model="whisper-test",
            lang="tr",
            llm_model="qwen-test",
        )

        self.assertEqual(JOBS[job_id]["status"], "completed")
        mock_extract.assert_not_called()
        res = JOBS[job_id]["result"]
        self.assertEqual(res["llm"]["terms"], [])
        self.assertEqual(res["llm"]["n_terms"], 0)
        del JOBS[job_id]

    @patch("gui.server.download_youtube_audio", return_value=600.0)
    @patch("gui.server._run_pipeline_job")
    def test_youtube_scope_capped_when_unconfirmed(self, mock_run, mock_dl):
        # Without confirmation, 1800s (30m) must be capped to 600.0s (10m)
        res = client.post(
            "/api/process-youtube",
            json={
                "url": "https://www.youtube.com/watch?v=rz5JCvMlCMs",
                "duration_s": 1800.0,
                "confirm_long": False,
            },
        )
        self.assertEqual(res.status_code, 200)
        job_id = res.json()["job_id"]
        # Wait briefly for thread to invoke download and pipeline
        import time
        for _ in range(40):
            if mock_dl.called and mock_run.called:
                break
            time.sleep(0.05)
        self.assertTrue(mock_dl.called)
        call_kwargs = mock_dl.call_args[1]
        self.assertEqual(call_kwargs.get("duration_s"), 600.0)
        del JOBS[job_id]

    @patch("gui.server.download_youtube_audio", return_value=1800.0)
    @patch("gui.server._run_pipeline_job")
    def test_youtube_scope_preserved_when_confirmed(self, mock_run, mock_dl):
        # With confirmation, requested 1800s is preserved
        res = client.post(
            "/api/process-youtube",
            json={
                "url": "https://www.youtube.com/watch?v=rz5JCvMlCMs",
                "duration_s": 1800.0,
                "confirm_long": True,
            },
        )
        self.assertEqual(res.status_code, 200)
        job_id = res.json()["job_id"]
        import time
        for _ in range(40):
            if mock_dl.called and mock_run.called:
                break
            time.sleep(0.05)
        self.assertTrue(mock_dl.called)
        call_kwargs = mock_dl.call_args[1]
        self.assertEqual(call_kwargs.get("duration_s"), 1800.0)
        del JOBS[job_id]

    @patch("gui.server.convert_local_audio", return_value=600.0)
    @patch("gui.server._run_pipeline_job")
    def test_upload_scope_capped_when_unconfirmed(self, mock_run, mock_conv):
        # Unconfirmed upload with duration > 600s must be capped to 600.0s
        test_content = b"fake audio content"
        files = {"file": ("test.wav", io.BytesIO(test_content), "audio/wav")}
        data = {
            "duration_s": "1200",
            "confirm_long": "false",
        }
        res = client.post("/api/process-upload", files=files, data=data)
        self.assertEqual(res.status_code, 200)
        job_id = res.json()["job_id"]
        import time
        for _ in range(40):
            if mock_conv.called and mock_run.called:
                break
            time.sleep(0.05)
        self.assertTrue(mock_conv.called)
        call_kwargs = mock_conv.call_args[1]
        self.assertEqual(call_kwargs.get("duration_s"), 600.0)
        del JOBS[job_id]

    def test_cleanup_temp_dir_purges_orphan_files(self):
        # Create a dummy orphan temp file
        test_file = TEMP_DIR / "test_orphan.wav"
        test_file.write_bytes(b"dummy audio")
        self.assertTrue(test_file.exists())

        removed = cleanup_temp_dir(max_age_s=0.0)
        self.assertGreaterEqual(removed, 1)
        self.assertFalse(test_file.exists())

    def test_cleanup_temp_dir_preserves_active_job_files(self):
        # Create an active job with an audio file in TEMP_DIR
        active_job_id = "test-active-job-cleanup"
        active_wav = TEMP_DIR / f"{active_job_id}.wav"
        active_wav.write_bytes(b"active audio")

        JOBS[active_job_id] = {
            "status": "processing",
            "audio_file": active_wav,
            "events": [],
            "created_at": 1000.0,
            "last_accessed": 1000.0,
        }

        try:
            cleanup_temp_dir(max_age_s=0.0)
            # The active job's file must NOT be deleted
            self.assertTrue(active_wav.exists())
        finally:
            del JOBS[active_job_id]
            if active_wav.exists():
                active_wav.unlink()

    def test_cleanup_expired_jobs_drops_old_jobs_and_wavs(self):
        import time
        now = time.time()
        old_job_id = "test-old-job-ttl"
        old_wav = TEMP_DIR / f"{old_job_id}.wav"
        old_wav.write_bytes(b"old audio")

        JOBS[old_job_id] = {
            "status": "completed",
            "audio_file": old_wav,
            "events": [],
            "created_at": now - 2000.0,
            "last_accessed": now - 2000.0,
        }

        recent_job_id = "test-recent-job-ttl"
        recent_wav = TEMP_DIR / f"{recent_job_id}.wav"
        recent_wav.write_bytes(b"recent audio")

        JOBS[recent_job_id] = {
            "status": "completed",
            "audio_file": recent_wav,
            "events": [],
            "created_at": now - 100.0,
            "last_accessed": now - 100.0,
        }

        try:
            removed_jobs = cleanup_expired_jobs(ttl_s=1800.0)
            self.assertGreaterEqual(removed_jobs, 1)

            # Old job removed and wav deleted
            self.assertNotIn(old_job_id, JOBS)
            self.assertFalse(old_wav.exists())

            # Recent job preserved and wav intact
            self.assertIn(recent_job_id, JOBS)
            self.assertTrue(recent_wav.exists())
        finally:
            JOBS.pop(old_job_id, None)
            JOBS.pop(recent_job_id, None)
            if old_wav.exists():
                old_wav.unlink()
            if recent_wav.exists():
                recent_wav.unlink()

    def test_endpoint_access_refreshes_last_accessed(self):
        import time
        now = time.time()
        job_id = "test-access-refresh-123"
        test_wav = TEMP_DIR / f"{job_id}.wav"
        test_wav.write_bytes(b"some wav data")

        initial_time = now - 500.0
        JOBS[job_id] = {
            "status": "completed",
            "audio_file": test_wav,
            "events": [],
            "result": {"llm": {"terms": []}, "asr": {"model": "m"}},
            "created_at": initial_time,
            "last_accessed": initial_time,
        }

        try:
            # Calling /api/job/{id} refreshes last_accessed
            res = client.get(f"/api/job/{job_id}")
            self.assertEqual(res.status_code, 200)
            refreshed = JOBS[job_id]["last_accessed"]
            self.assertGreater(refreshed, initial_time)
        finally:
            JOBS.pop(job_id, None)
            if test_wav.exists():
                test_wav.unlink()


class PracticeExamAndLearnerTests(unittest.TestCase):
    def setUp(self):
        # Reset profile before each test
        client.post("/api/learner/reset")

    def test_learner_profile_and_reset(self):
        res = client.get("/api/learner/profile")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("learner_id", data)
        self.assertIn("stats", data)
        self.assertIn("recent_exams", data)
        self.assertIn("analysis", data)
        self.assertEqual(data["stats"]["exams_taken"], 0)

        # Reset endpoint works
        reset_res = client.post("/api/learner/reset")
        self.assertEqual(reset_res.status_code, 200)
        self.assertTrue(reset_res.json()["ok"])

    def test_exam_generate_validation(self):
        # 404 for nonexistent job
        res = client.post("/api/exam/generate", json={"job_id": "nonexistent-job-xyz"})
        self.assertEqual(res.status_code, 404)

        # 400 for job without terms
        job_id = "test-empty-terms-job"
        JOBS[job_id] = {
            "status": "completed",
            "result": {"llm": {"terms": []}, "asr": {"text": "hello"}},
            "events": [],
        }
        try:
            res = client.post("/api/exam/generate", json={"job_id": job_id})
            self.assertEqual(res.status_code, 400)
        finally:
            JOBS.pop(job_id, None)

    def test_exam_full_lifecycle_and_adaptive_weighting(self):
        from gui.learner import build_learner_prompt_block, load_profile

        # 1. Setup mock completed job with lecture terms
        job_id = "test-lecture-exam-job"
        sample_terms = [
            {"term": "Neural Network", "explanation": "A series of algorithms that mimic human brain operations."},
            {"term": "Backpropagation", "explanation": "An algorithm for computing the gradient of the loss function."},
            {"term": "Activation Function", "explanation": "A function that decides whether a neuron should be activated."},
        ]
        JOBS[job_id] = {
            "status": "completed",
            "result": {
                "job_id": job_id,
                "video_meta": {"id": "AI-101", "title": "Intro to Deep Learning"},
                "lang": "en",
                "asr": {"text": "In this lecture we discuss Neural Network, Backpropagation, and Activation Function."},
                "llm": {"terms": sample_terms, "model": "Qwen/Qwen3.5-4B"},
            },
            "events": [],
        }

        try:
            # 2. Generate exam (deterministic fallback mode to ensure instantaneous test without model weights)
            gen_res = client.post("/api/exam/generate", json={"job_id": job_id, "use_llm": False})
            self.assertEqual(gen_res.status_code, 200)
            exam_data = gen_res.json()
            self.assertIn("exam_id", exam_data)
            exam_id = exam_data["exam_id"]
            questions = exam_data["questions"]
            self.assertGreaterEqual(len(questions), 3)

            # Check question structure (no internal scoring keys leaked)
            for q in questions:
                self.assertIn("id", q)
                self.assertIn("type", q)
                self.assertIn("term", q)
                self.assertNotIn("answer_index", q)  # Answer key must not be sent to browser

            # 3. Retrieve exam via GET /api/exam/{exam_id}
            get_res = client.get(f"/api/exam/{exam_id}")
            self.assertEqual(get_res.status_code, 200)
            self.assertEqual(get_res.json()["exam_id"], exam_id)

            # 4. Submit answers: answer Backpropagation WRONG, Neural Network CORRECT
            q_backprop = next(q for q in questions if "backpropagation" in q["term"].lower())
            q_neural = next(q for q in questions if "neural" in q["term"].lower())

            answers = [
                {"question_id": q_backprop["id"], "self_grade": "dont_know"},
                {"question_id": q_neural["id"], "self_grade": "know"},
            ]

            submit_res = client.post("/api/exam/submit", json={"exam_id": exam_id, "answers": answers})
            self.assertEqual(submit_res.status_code, 200)
            sub_data = submit_res.json()

            self.assertEqual(sub_data["total"], 2)
            self.assertEqual(sub_data["correct"], 1)
            self.assertEqual(sub_data["score"], 0.5)
            self.assertIn("Backpropagation", sub_data["weak_terms"])
            self.assertIn("Neural Network", sub_data["strong_terms"])
            self.assertIn("analysis_summary", sub_data)

            # 5. Verify learner profile is updated and reflects the exam
            prof_res = client.get("/api/learner/profile")
            self.assertEqual(prof_res.status_code, 200)
            prof_data = prof_res.json()
            self.assertEqual(prof_data["stats"]["exams_taken"], 1)
            self.assertEqual(prof_data["stats"]["avg_score"], 0.5)
            self.assertGreaterEqual(prof_data["stats"]["weak_count"], 1)

            # 6. Verify adaptive weighting: subsequent LLM analyses will prioritize Backpropagation
            active_profile = load_profile()
            learner_prompt = build_learner_prompt_block(active_profile)
            self.assertIn("Review-priority terms", learner_prompt)
            self.assertIn("Backpropagation", learner_prompt)

            # 7. Verify submitting again returns cached result without double-recording
            resubmit_res = client.post("/api/exam/submit", json={"exam_id": exam_id, "answers": answers})
            self.assertEqual(resubmit_res.status_code, 200)
            prof_after = load_profile()
            self.assertEqual(prof_after["stats"]["exams_taken"], 1)

        finally:
            JOBS.pop(job_id, None)


if __name__ == "__main__":
    unittest.main()

