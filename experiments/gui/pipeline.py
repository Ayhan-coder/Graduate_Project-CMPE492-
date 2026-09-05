"""Core processing pipeline for Speech-to-Terminology Web GUI.

Handles:
1. YouTube metadata extraction and audio streaming via yt-dlp.
2. Audio conversion and slicing to 16kHz mono WAV via ffmpeg.
3. Fast Whisper ASR inference (cached on RTX 5060 GPU).
4. Terminology extraction & CEFR B1 explanation using Qwen3.5-4B (4-bit CUDA).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import torch

# Import project helper utilities
ROOT = Path(__file__).resolve().parent.parent
from fetch_models import resolve_hub_cache
from run_llm import PROMPT, _local_model_path, generate_hf, parse_terms

# Global in-memory Whisper cache: model_id -> (processor, model)
_WHISPER_CACHE: dict[str, tuple[object, object]] = {}

TEMP_DIR = ROOT / "gui" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def get_youtube_info(url: str) -> dict:
    """Fetch video metadata without downloading video stream."""
    import yt_dlp

    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info.get("id") or ""
        return {
            "id": video_id,
            "url": url,
            "title": info.get("title") or "Unknown Title",
            "channel": info.get("uploader") or info.get("channel") or "Unknown Channel",
            "duration": int(info.get("duration") or 0),
            "thumbnail": info.get("thumbnail") or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        }


def _run_cmd(cmd: list[str]) -> None:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr[-500:]}")


def download_youtube_audio(
    url: str,
    out_wav: Path,
    start_s: float = 0.0,
    duration_s: Optional[float] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> float:
    """Download audio stream via yt-dlp and convert to 16 kHz mono WAV."""
    if progress_cb:
        progress_cb("Downloading audio stream from YouTube...")

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    temp_stem = TEMP_DIR / f"yt_{int(time.time() * 1000)}"

    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--no-warnings",
        "--ignore-errors",
        "-o",
        f"{temp_stem}.%(ext)s",
        url,
    ]
    _run_cmd(cmd)

    # Locate downloaded file
    downloaded = None
    for ext in (".wav", ".webm", ".m4a", ".mp3", ".opus", ".ogg"):
        candidate = Path(f"{temp_stem}{ext}")
        if candidate.exists():
            downloaded = candidate
            break

    if not downloaded:
        raise FileNotFoundError(f"yt-dlp completed but output file not found for {url}")

    if progress_cb:
        progress_cb("Converting audio to 16 kHz mono WAV with ffmpeg...")

    # Convert to 16kHz mono WAV, with optional slice
    ffmpeg_cmd = ["ffmpeg", "-y"]
    if start_s > 0:
        ffmpeg_cmd.extend(["-ss", str(start_s)])
    ffmpeg_cmd.extend(["-i", str(downloaded)])
    if duration_s and duration_s > 0:
        ffmpeg_cmd.extend(["-t", str(duration_s)])
    ffmpeg_cmd.extend(["-ac", "1", "-ar", "16000", "-vn", str(out_wav)])

    _run_cmd(ffmpeg_cmd)

    # Cleanup temp source
    try:
        downloaded.unlink()
    except Exception:
        pass

    # Probe duration
    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(out_wav),
        ],
        text=True,
    ).strip()
    return float(probe)


def convert_local_audio(
    src_file: Path,
    out_wav: Path,
    start_s: float = 0.0,
    duration_s: Optional[float] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> float:
    """Convert uploaded local file to 16 kHz mono WAV."""
    if progress_cb:
        progress_cb("Converting uploaded file to 16 kHz mono WAV...")

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = ["ffmpeg", "-y"]
    if start_s > 0:
        ffmpeg_cmd.extend(["-ss", str(start_s)])
    ffmpeg_cmd.extend(["-i", str(src_file)])
    if duration_s and duration_s > 0:
        ffmpeg_cmd.extend(["-t", str(duration_s)])
    ffmpeg_cmd.extend(["-ac", "1", "-ar", "16000", "-vn", str(out_wav)])

    _run_cmd(ffmpeg_cmd)

    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(out_wav),
        ],
        text=True,
    ).strip()
    return float(probe)


def parse_timestamp_segments(raw_text: str, fallback_full_text: str) -> list[dict]:
    """Parse Whisper timestamp tokens <|0.00|> text <|5.20|> into structured segments."""
    segments = []
    # Pattern matching: <|timestamp|> text <|timestamp|>
    pattern = re.compile(r"<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>")
    matches = pattern.findall(raw_text)

    if matches:
        for start_s, text, end_s in matches:
            t = text.strip()
            if t:
                segments.append({
                    "start": round(float(start_s), 2),
                    "end": round(float(end_s), 2),
                    "text": t,
                })
    
    # Fallback: if no valid timestamps captured, split full text into sentence chunks
    if not segments and fallback_full_text:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", fallback_full_text) if s.strip()]
        for i, s in enumerate(sentences):
            segments.append({
                "start": None,
                "end": None,
                "text": s,
            })

    return segments


def transcribe_audio(
    audio_path: Path,
    model_id: str = "openai/whisper-large-v3-turbo",
    lang: str = "auto",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run Whisper ASR on audio file and return full text & timestamped segments."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import soundfile as sf
    import numpy as np

    if progress_cb:
        progress_cb(f"Loading Whisper model ({model_id}) on GPU...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    packed = _WHISPER_CACHE.get(model_id)
    if packed is None:
        cache = str(resolve_hub_cache(model_id))
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache)
        if getattr(model.generation_config, "max_length", None):
            model.generation_config.max_length = None
        packed = (processor, model)
        _WHISPER_CACHE[model_id] = packed

    processor, model = packed

    if progress_cb:
        progress_cb("Transcribing audio with Whisper...")

    t0 = time.perf_counter()
    audio, sr = sf.read(str(audio_path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)

    duration_s = len(audio) / sr

    # Ensure minimum 30s for Whisper's 3000 mel frame requirements
    if len(audio) < 30 * sr:
        audio = np.pad(audio, (0, 30 * sr - len(audio)))

    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(dtype=dtype)

    gen_kwargs = {
        "task": "transcribe",
        "return_timestamps": True,
        "condition_on_prev_tokens": False,
        "compression_ratio_threshold": 1.35,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "num_beams": 1,
        "max_new_tokens": 440,
    }
    if lang in {"tr", "en"}:
        gen_kwargs["language"] = lang

    with torch.inference_mode():
        ids = model.generate(**inputs, **gen_kwargs)

    elapsed = time.perf_counter() - t0
    rtf = elapsed / duration_s if duration_s > 0 else 0.0

    clean_text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    raw_text = processor.batch_decode(ids, skip_special_tokens=False)[0]

    segments = parse_timestamp_segments(raw_text, clean_text)

    return {
        "text": clean_text,
        "segments": segments,
        "duration_s": round(duration_s, 2),
        "latency_s": round(elapsed, 2),
        "rtf": round(rtf, 4),
        "model": model_id,
        "device": device,
    }


def extract_terms(
    transcript: str,
    lang: str = "tr",
    model_id: str = "Qwen/Qwen3.5-4B",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Extract domain terminology and CEFR B1 explanations using LLM."""
    if progress_cb:
        progress_cb(f"Extracting domain terms with {model_id} (4-bit GPU)...")

    lang_label = "Turkish" if lang.lower().startswith("tr") else "English"
    prompt = PROMPT.format(LANG=lang_label, TRANSCRIPT=transcript)

    t0 = time.perf_counter()
    raw = generate_hf(model_id, prompt)
    elapsed = time.perf_counter() - t0

    terms = parse_terms(raw)

    return {
        "terms": terms,
        "n_terms": len(terms),
        "latency_s": round(elapsed, 2),
        "raw": raw,
        "model": model_id,
        "lang": lang,
    }
