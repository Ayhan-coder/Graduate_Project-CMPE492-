"""Unit tests for the speech and NLP processing pipeline in experiments/gui/pipeline.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gui.pipeline as pipeline


class ExtractYoutubeIdTests(unittest.TestCase):
    def test_standard_watch_urls(self):
        self.assertEqual(pipeline.extract_youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(pipeline.extract_youtube_id("http://youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(pipeline.extract_youtube_id("https://m.youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_urls_with_additional_query_params(self):
        self.assertEqual(pipeline.extract_youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&feature=share"), "dQw4w9WgXcQ")
        self.assertEqual(pipeline.extract_youtube_id("https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_short_urls(self):
        self.assertEqual(pipeline.extract_youtube_id("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(pipeline.extract_youtube_id("https://youtu.be/dQw4w9WgXcQ?t=10"), "dQw4w9WgXcQ")

    def test_shorts_and_embed_urls(self):
        self.assertEqual(pipeline.extract_youtube_id("https://www.youtube.com/shorts/2IK3DFHRFfw"), "2IK3DFHRFfw")
        self.assertEqual(pipeline.extract_youtube_id("https://www.youtube.com/embed/BaH04m_fS1c"), "BaH04m_fS1c")
        self.assertEqual(pipeline.extract_youtube_id("https://www.youtube.com/v/dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_raw_id_and_whitespace(self):
        self.assertEqual(pipeline.extract_youtube_id("dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(pipeline.extract_youtube_id("   dQw4w9WgXcQ   "), "dQw4w9WgXcQ")
        self.assertEqual(pipeline.extract_youtube_id("   https://youtu.be/dQw4w9WgXcQ   "), "dQw4w9WgXcQ")

    def test_invalid_urls(self):
        self.assertIsNone(pipeline.extract_youtube_id(None))
        self.assertIsNone(pipeline.extract_youtube_id(""))
        self.assertIsNone(pipeline.extract_youtube_id("     "))
        self.assertIsNone(pipeline.extract_youtube_id("not-a-valid-youtube-id"))
        self.assertIsNone(pipeline.extract_youtube_id("https://vimeo.com/12345678"))
        self.assertIsNone(pipeline.extract_youtube_id("https://google.com"))

    def test_ssrf_and_arbitrary_urls_rejected(self):
        ssrf_cases = [
            "http://127.0.0.1:8000/watch?v=dQw4w9WgXcQ",
            "http://169.254.169.254/latest/meta-data/?v=dQw4w9WgXcQ",
            "file:///etc/passwd",
            "http://attacker.com/embed/dQw4w9WgXcQ",
            "http://localhost:3000/shorts/dQw4w9WgXcQ",
        ]
        for url in ssrf_cases:
            self.assertIsNone(pipeline.extract_youtube_id(url), f"SSRF URL was not rejected: {url}")


class DetectLangTests(unittest.TestCase):
    def test_turkish_texts(self):
        self.assertEqual(pipeline.detect_lang("Bu bir Türkçe konuşma metnidir."), "tr")
        self.assertEqual(pipeline.detect_lang("Öğrenme algoritmaları ve veri yapıları"), "tr")
        self.assertEqual(pipeline.detect_lang("ŞAFAK VAKTİNDE GÖRÜŞÜRÜZ"), "tr")
        self.assertEqual(pipeline.detect_lang("çıktı doğrulaması"), "tr")

    def test_english_texts(self):
        self.assertEqual(pipeline.detect_lang("This is a standard English speech transcript."), "en")
        self.assertEqual(pipeline.detect_lang("Agile methodologies, sprint planning, and backlog items."), "en")

    def test_edge_cases(self):
        self.assertEqual(pipeline.detect_lang(""), "en")
        self.assertEqual(pipeline.detect_lang(None), "en")
        self.assertEqual(pipeline.detect_lang("1234567890 !@#$%^&*()"), "en")


class ParseTimestampSegmentsTests(unittest.TestCase):
    def test_whisper_token_parsing(self):
        raw = "<|0.00|> Agile is a set of practices. <|3.50|> Scrum is a framework. <|7.20|>"
        fallback = "Agile is a set of practices. Scrum is a framework."
        segs = pipeline.parse_timestamp_segments(raw, fallback)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["start"], 0.0)
        self.assertEqual(segs[0]["end"], 3.5)
        self.assertEqual(segs[0]["text"], "Agile is a set of practices.")
        self.assertEqual(segs[1]["start"], 3.5)
        self.assertEqual(segs[1]["end"], 7.2)
        self.assertEqual(segs[1]["text"], "Scrum is a framework.")

    def test_fallback_sentences(self):
        segs = pipeline.parse_timestamp_segments("", "Sentence one. Sentence two! Sentence three?")
        self.assertEqual(len(segs), 3)
        self.assertIsNone(segs[0]["start"])
        self.assertEqual(segs[0]["text"], "Sentence one.")
        self.assertIsNone(segs[1]["start"])
        self.assertEqual(segs[1]["text"], "Sentence two!")
        self.assertIsNone(segs[2]["start"])
        self.assertEqual(segs[2]["text"], "Sentence three?")

    def test_empty_input(self):
        segs = pipeline.parse_timestamp_segments("", "")
        self.assertEqual(segs, [])

    def test_malformed_tokens(self):
        raw = "<|invalid|> Some text"
        fallback = "Some text."
        segs = pipeline.parse_timestamp_segments(raw, fallback)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["text"], "Some text.")


class GetYoutubeInfoTests(unittest.TestCase):
    def test_local_cache_hit(self):
        info = pipeline.get_youtube_info("https://www.youtube.com/watch?v=rz5JCvMlCMs")
        self.assertEqual(info["id"], "rz5JCvMlCMs")
        self.assertIn("PROJE YÖNETİMİ", info["title"])
        self.assertGreater(info["duration"], 0)

    @patch("gui.pipeline._load_sources_cache", return_value={})
    @patch("urllib.request.urlopen")
    def test_oembed_path_when_ytdlp_fails(self, mock_urlopen, mock_sources):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "title": "Mocked Video Title",
            "author_name": "Mock Channel",
            "thumbnail_url": "https://example.com/thumb.jpg",
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        # Mock yt_dlp so it fails and execution falls back to oEmbed metadata
        mock_ytdl = MagicMock()
        mock_ytdl.YoutubeDL.side_effect = RuntimeError("yt-dlp disabled in test")
        with patch.dict("sys.modules", {"yt_dlp": mock_ytdl}):
            info = pipeline.get_youtube_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            self.assertEqual(info["id"], "dQw4w9WgXcQ")
            self.assertEqual(info["title"], "Mocked Video Title")
            self.assertEqual(info["channel"], "Mock Channel")

    @patch("gui.pipeline._load_sources_cache", return_value={})
    def test_ytdlp_enrichment_path(self, mock_sources):
        mock_ytdl_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = {
            "id": "dQw4w9WgXcQ",
            "title": "Enriched Video Title",
            "channel": "Enriched Channel",
            "duration": 212,
            "thumbnail": "https://example.com/enriched.jpg",
        }
        mock_ytdl_cls.return_value.__enter__.return_value = mock_instance

        mock_module = MagicMock(YoutubeDL=mock_ytdl_cls)
        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            info = pipeline.get_youtube_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            self.assertEqual(info["id"], "dQw4w9WgXcQ")
            self.assertEqual(info["title"], "Enriched Video Title")
            self.assertEqual(info["duration"], 212)

    @patch("gui.pipeline._load_sources_cache", return_value={})
    @patch("urllib.request.urlopen", side_effect=Exception("Network error"))
    def test_last_resort_fallback_when_valid_id(self, mock_urlopen, mock_sources):
        mock_ytdl = MagicMock()
        mock_ytdl.YoutubeDL.side_effect = RuntimeError("yt-dlp failed")
        with patch.dict("sys.modules", {"yt_dlp": mock_ytdl}):
            info = pipeline.get_youtube_info("https://www.youtube.com/watch?v=abc123xyz89")
            self.assertEqual(info["id"], "abc123xyz89")
            self.assertIn("abc123xyz89", info["title"])
            self.assertEqual(info["channel"], "YouTube")

    def test_invalid_url_raises_value_error(self):
        mock_ytdl = MagicMock()
        mock_ytdl.YoutubeDL.side_effect = RuntimeError("yt-dlp failed")
        with patch.dict("sys.modules", {"yt_dlp": mock_ytdl}):
            with self.assertRaises(ValueError):
                pipeline.get_youtube_info("https://example.com/not-youtube")


class AudioHelpersTests(unittest.TestCase):
    @patch("gui.pipeline._run_cmd")
    @patch("subprocess.check_output", return_value="14.5\n")
    def test_convert_local_audio(self, mock_check_output, mock_run_cmd):
        src = Path("test_input.mp3")
        out = Path("test_output.wav")
        dur = pipeline.convert_local_audio(src, out, start_s=2.5, duration_s=10.0)
        self.assertEqual(dur, 14.5)
        mock_run_cmd.assert_called_once()
        cmd_args = mock_run_cmd.call_args[0][0]
        self.assertIn("-ss", cmd_args)
        self.assertIn("2.5", cmd_args)
        self.assertIn("-t", cmd_args)
        self.assertIn("10.0", cmd_args)
        self.assertIn("16000", cmd_args)

    @patch("gui.pipeline._run_cmd")
    @patch("gui.pipeline._probe_duration", return_value=5.0)
    def test_download_youtube_audio_local_cache_hit(self, mock_probe, mock_run_cmd):
        out_wav = Path("out.wav")
        dur = pipeline.download_youtube_audio("https://www.youtube.com/watch?v=rz5JCvMlCMs", out_wav, start_s=1.0, duration_s=5.0)
        self.assertEqual(dur, 5.0)
        mock_run_cmd.assert_called_once()
        cmd_args = mock_run_cmd.call_args[0][0]
        self.assertEqual(cmd_args[0], "ffmpeg")

    @patch("gui.pipeline.extract_youtube_id", return_value="test_vid_123")
    @patch("gui.pipeline._load_sources_cache", return_value={})
    @patch("gui.pipeline._run_cmd")
    @patch("gui.pipeline._probe_duration", return_value=12.0)
    def test_download_youtube_audio_yt_dlp_path(self, mock_probe, mock_run_cmd, mock_sources, mock_extract):
        with tempfile.TemporaryDirectory() as td:
            out_wav = Path(td) / "out.wav"

            def fake_run_cmd(cmd):
                if "-o" in cmd:
                    out_pat = cmd[cmd.index("-o") + 1]
                    stem = out_pat.replace(".%(ext)s", "")
                    Path(f"{stem}.m4a").write_text("dummy audio")

            mock_run_cmd.side_effect = fake_run_cmd

            dur = pipeline.download_youtube_audio(
                "https://www.youtube.com/watch?v=test_vid_123",
                out_wav,
                start_s=0.0,
                duration_s=10.0,
            )
            self.assertEqual(dur, 12.0)
            self.assertEqual(mock_run_cmd.call_count, 2)

    def test_download_youtube_audio_rejects_ssrf(self):
        with self.assertRaises(ValueError):
            pipeline.download_youtube_audio("http://127.0.0.1:8000/bad", Path("test.wav"))

    @patch("gc.collect")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    def test_clear_gpu_memory(self, mock_empty_cache, mock_cuda_avail, mock_gc):
        pipeline._WHISPER_CACHE["test_model"] = ("proc", "model")
        pipeline.clear_gpu_memory()
        self.assertEqual(len(pipeline._WHISPER_CACHE), 0)
        self.assertGreaterEqual(mock_gc.call_count, 1)
        self.assertGreaterEqual(mock_empty_cache.call_count, 1)


class InferenceMocksTests(unittest.TestCase):
    @patch("gui.pipeline.clear_gpu_memory")
    @patch("gui.pipeline.generate_hf", return_value="1. **Sprint**: A fixed iteration.\n2. **Backlog**: Work items.")
    @patch("gui.pipeline.parse_terms", return_value=[
        {"term": "Sprint", "explanation": "A fixed iteration."},
        {"term": "Backlog", "explanation": "Work items."},
    ])
    def test_extract_terms(self, mock_parse, mock_generate, mock_clear_gpu):
        res = pipeline.extract_terms("Sprint planning happens first.", lang="en", model_id="test-llm")
        self.assertEqual(res["n_terms"], 2)
        self.assertEqual(len(res["terms"]), 2)
        self.assertEqual(res["terms"][0]["term"], "Sprint")
        self.assertEqual(res["model"], "test-llm")
        self.assertEqual(res["lang"], "en")
        mock_generate.assert_called_once()
        mock_parse.assert_called_once()

    @patch("gui.pipeline.clear_gpu_memory")
    def test_transcribe_audio_mocked(self, mock_clear_gpu):
        import numpy as np
        import torch

        mock_proc = MagicMock()
        mock_proc.return_value = {"input_features": torch.zeros((1, 80, 3000))}
        mock_proc.batch_decode.side_effect = [
            ["Eğitimde yapay zeka uygulamaları"],
            ["<|0.00|> Eğitimde yapay zeka uygulamaları <|2.00|>"],
        ]

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[50258, 50259]])
        mock_model.generation_config = MagicMock()
        mock_model.to.return_value = mock_model

        mock_transformers = MagicMock()
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_proc
        mock_transformers.AutoModelForSpeechSeq2Seq.from_pretrained.return_value = mock_model

        mock_sf = MagicMock()
        mock_sf.read.return_value = (np.zeros(32000, dtype=np.float32), 16000)

        with patch.dict("sys.modules", {"transformers": mock_transformers, "soundfile": mock_sf}):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(b"dummy")
                tmp_path = Path(tmp.name)

            try:
                pipeline._WHISPER_CACHE.clear()
                pipeline._ACTIVE_ASR_MODEL = None
                res = pipeline.transcribe_audio(tmp_path, model_id="test-whisper", lang="tr")
                self.assertEqual(res["text"], "Eğitimde yapay zeka uygulamaları")
                self.assertEqual(len(res["segments"]), 1)
                self.assertEqual(res["segments"][0]["start"], 0.0)
                self.assertEqual(res["segments"][0]["end"], 2.0)
                self.assertEqual(res["model"], "test-whisper")
                _, load_kwargs = mock_transformers.AutoModelForSpeechSeq2Seq.from_pretrained.call_args
                self.assertIn("device_map", load_kwargs)
                mock_model.to.assert_not_called()
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

    def test_clear_asr_cache(self):
        pipeline._WHISPER_CACHE["whisper-test"] = ("proc", "model")
        pipeline._ACTIVE_ASR_MODEL = "whisper-test"
        pipeline.clear_asr_cache()
        self.assertEqual(len(pipeline._WHISPER_CACHE), 0)
        self.assertIsNone(pipeline._ACTIVE_ASR_MODEL)

    @patch("gui.pipeline.clear_asr_cache")
    @patch("gui.pipeline.get_free_vram_mb", return_value=1500.0)
    @patch("gui.pipeline.generate_hf", return_value="[]")
    @patch("gui.pipeline.parse_terms", return_value=[])
    def test_extract_terms_evicts_asr_when_vram_tight(self, mock_parse, mock_gen, mock_vram, mock_clear_asr):
        pipeline.extract_terms("Some text", lang="en", model_id="Qwen/Qwen3.5-9B")
        mock_clear_asr.assert_called_once()

    @patch("soundfile.read", return_value=(__import__("numpy").zeros(16000, dtype=__import__("numpy").float32), 16000))
    @patch("gui.pipeline.clear_llm_cache")
    @patch("gui.pipeline.get_free_vram_mb", return_value=1500.0)
    def test_transcribe_audio_evicts_llm_when_vram_tight(self, mock_vram, mock_clear_llm, mock_sf_read):
        import torch

        mock_proc = MagicMock()
        mock_proc.return_value = {"input_features": torch.zeros((1, 80, 3000))}
        mock_proc.batch_decode.return_value = ["test transcript"]
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2]])
        mock_model.to.return_value = mock_model
        mock_transformers = MagicMock()
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_proc
        mock_transformers.AutoModelForSpeechSeq2Seq.from_pretrained.return_value = mock_model

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(b"wavdata")
                tmp_p = Path(tmp.name)
            try:
                pipeline._WHISPER_CACHE.clear()
                pipeline._ACTIVE_ASR_MODEL = None
                pipeline.transcribe_audio(tmp_p, model_id="new-whisper")
                mock_clear_llm.assert_called_once()
                self.assertIn("new-whisper", pipeline._WHISPER_CACHE)
            finally:
                if tmp_p.exists():
                    tmp_p.unlink()
                pipeline.clear_asr_cache()

    @patch("soundfile.read", return_value=(__import__("numpy").zeros(16000, dtype=__import__("numpy").float32), 16000))
    def test_transcribe_audio_reuses_resident_model(self, mock_sf_read):
        import torch

        mock_proc = MagicMock()
        mock_proc.return_value = {"input_features": torch.zeros((1, 80, 3000))}
        mock_proc.batch_decode.return_value = ["resident whisper"]
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2]])
        mock_model.to.return_value = mock_model

        pipeline._WHISPER_CACHE["resident-whisper"] = (mock_proc, mock_model)
        pipeline._ACTIVE_ASR_MODEL = "resident-whisper"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"wavdata")
            tmp_p = Path(tmp.name)
        try:
            res = pipeline.transcribe_audio(tmp_p, model_id="resident-whisper")
            self.assertEqual(res["text"], "resident whisper")
            self.assertIn("resident-whisper", pipeline._WHISPER_CACHE)
        finally:
            if tmp_p.exists():
                tmp_p.unlink()
            pipeline.clear_asr_cache()

    def test_extract_terms_empty_transcript(self):
        # Empty or whitespace transcript should immediately return without invoking LLM
        res = pipeline.extract_terms("", lang="en")
        self.assertEqual(res["terms"], [])
        self.assertEqual(res["n_terms"], 0)
        self.assertEqual(res["latency_s"], 0.0)

        res_spaces = pipeline.extract_terms("   \n\t  ", lang="tr")
        self.assertEqual(res_spaces["terms"], [])
        self.assertEqual(res_spaces["n_terms"], 0)

    @patch("gui.pipeline.clear_asr_cache")
    @patch("gui.pipeline.generate_hf")
    def test_extract_terms_windows_long_transcript(self, mock_gen, mock_clear):
        # Generate a 1200-word text to trigger windowing (>700 words)
        words = ["software", "engineering", "agile", "sprint", "kanban", "devops"] * 200
        long_text = " ".join(words)
        self.assertGreater(len(long_text.split()), 700)

        mock_gen.side_effect = [
            '[{"term": "agile", "explanation": "Agile is an iterative methodology."}]',
            '[{"term": "kanban", "explanation": "Kanban is a visual board system."}, {"term": "agile", "explanation": "Shorter."}]',
            '[{"term": "devops", "explanation": "Devops combines dev and ops."}]',
        ]

        res = pipeline.extract_terms(long_text, lang="en", model_id="Qwen/Qwen3.5-4B")
        self.assertGreater(mock_gen.call_count, 1)
        term_names = [t["term"] for t in res["terms"]]
        self.assertIn("agile", term_names)
        self.assertIn("kanban", term_names)
        self.assertIn("devops", term_names)
        # Deduplication check: agile should only appear once
        self.assertEqual(term_names.count("agile"), 1)
        # Should retain richest explanation
        agile_item = next(t for t in res["terms"] if t["term"] == "agile")
        self.assertEqual(agile_item["explanation"], "Agile is an iterative methodology.")

    @patch("soundfile.read")
    def test_transcribe_audio_chunks_long_audio(self, mock_sf_read):
        import numpy as np
        import torch

        # Simulate 65 seconds of audio at 16000 Hz = 1,040,000 samples
        # Expected chunks: 30s (0-30s), 30s (30-60s), 5s (60-65s) = 3 chunks
        sr = 16000
        samples = np.zeros(65 * sr, dtype=np.float32)
        mock_sf_read.return_value = (samples, sr)

        mock_proc = MagicMock()
        mock_proc.return_value = {"input_features": torch.zeros((1, 80, 3000))}
        call_count = [0]
        def fake_decode(ids, skip_special_tokens=True):
            idx = call_count[0] // 2
            call_count[0] += 1
            if idx == 0:
                return ["First thirty seconds."] if skip_special_tokens else ["<|0.00|> First thirty seconds. <|28.00|>"]
            elif idx == 1:
                return ["Second thirty seconds."] if skip_special_tokens else ["<|2.00|> Second thirty seconds. <|25.00|>"]
            else:
                return ["Final five seconds."] if skip_special_tokens else ["<|0.50|> Final five seconds. <|4.50|>"]

        mock_proc.batch_decode.side_effect = fake_decode

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2]])
        mock_model.to.return_value = mock_model

        pipeline._WHISPER_CACHE["chunk-whisper"] = (mock_proc, mock_model)
        pipeline._ACTIVE_ASR_MODEL = "chunk-whisper"

        progress_msgs = []
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake-wav")
            tmp_p = Path(tmp.name)
        try:
            res = pipeline.transcribe_audio(
                tmp_p,
                model_id="chunk-whisper",
                progress_cb=lambda m: progress_msgs.append(m),
            )
            self.assertEqual(mock_model.generate.call_count, 3)
            self.assertIn("First thirty seconds.", res["text"])
            self.assertIn("Second thirty seconds.", res["text"])
            self.assertIn("Final five seconds.", res["text"])

            # Verify chunk offset calculation
            segs = res["segments"]
            self.assertEqual(len(segs), 3)
            # Chunk 1: 0.0 + 0.0 = 0.0
            self.assertEqual(segs[0]["start"], 0.0)
            self.assertEqual(segs[0]["end"], 28.0)
            # Chunk 2: 30.0 + 2.0 = 32.0
            self.assertEqual(segs[1]["start"], 32.0)
            self.assertEqual(segs[1]["end"], 55.0)
            # Chunk 3: 60.0 + 0.5 = 60.5
            self.assertEqual(segs[2]["start"], 60.5)
            self.assertEqual(segs[2]["end"], 64.5)

            # Check chunk progress messages
            chunk_msgs = [m for m in progress_msgs if "chunk" in m.lower()]
            self.assertGreaterEqual(len(chunk_msgs), 3)
        finally:
            if tmp_p.exists():
                tmp_p.unlink()
            pipeline.clear_asr_cache()


if __name__ == "__main__":
    unittest.main()
