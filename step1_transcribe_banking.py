#!/usr/bin/env python3
"""
step1_transcribe_banking.py
Transcribes the 3 banking audio lectures using Whisper Large-v3-Turbo on NVIDIA A40.
Produces raw and cleaned transcripts in /users/ali.gunder/banking_project/transcripts/.
"""

import os
import sys
import json
import time
import logging
import torch
import torchaudio
from pathlib import Path
from transformers import AutoProcessor, WhisperForConditionalGeneration

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BankingTranscribe")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_PATH = "/users/ali.gunder/.cache/huggingface/hub/models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
AUDIO_DIR = Path("/users/ali.gunder/banking_project/audio")
OUTPUT_DIR = Path("/users/ali.gunder/banking_project/transcripts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_FILES = [
    {
        "id": "bank_01",
        "title": "TBB Finansal Okuryazarlık ve Bütçe Yönetimi",
        "filename": "bank_01.opus"
    },
    {
        "id": "bank_02",
        "title": "AÖF Bankacılık ve Sigortacılık - Mevduat ve Kredi Mekanizması",
        "filename": "bank_02.opus"
    },
    {
        "id": "bank_03",
        "title": "TCMB Para Politikası Araçları ve Zorunlu Karşılıklar",
        "filename": "bank_03.opus"
    }
]

def load_audio_chunked(file_path: Path, chunk_duration_sec: int = 30, target_sr: int = 16000):
    """Load and resample audio into 30s chunks for Whisper."""
    waveform, sample_rate = torchaudio.load(str(file_path))
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)

    total_samples = waveform.shape[1]
    chunk_samples = chunk_duration_sec * target_sr
    chunks = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, start:end]
        if chunk.shape[1] < chunk_samples:
            pad = torch.zeros((1, chunk_samples - chunk.shape[1]))
            chunk = torch.cat([chunk, pad], dim=1)
        chunks.append(chunk.squeeze(0).numpy())
    return chunks, (total_samples / target_sr)

def main():
    logger.info("Initializing Whisper Large-v3 Turbo on %s...", DEVICE)
    processor = AutoProcessor.from_pretrained(WHISPER_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_PATH,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()

    corpus_summary = []

    for item in AUDIO_FILES:
        audio_path = AUDIO_DIR / item["filename"]
        logger.info(f"\n--- Transcribing {item['id']}: {item['title']} ({audio_path}) ---")
        t0 = time.time()
        chunks, duration_sec = load_audio_chunked(audio_path)
        logger.info(f"Loaded {len(chunks)} chunks (Total duration: {duration_sec:.1f}s / {duration_sec/60:.2f} min)")

        transcribed_segments = []
        for idx, chunk in enumerate(chunks):
            input_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
            if DEVICE == "cuda":
                input_features = input_features.half()

            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    language="turkish",
                    task="transcribe",
                    max_new_tokens=128
                )
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            if text:
                transcribed_segments.append(text)

        full_transcript = " ".join(transcribed_segments)
        elapsed = time.time() - t0
        rtf = elapsed / max(1.0, duration_sec)
        word_count = len(full_transcript.split())

        logger.info(f"Finished {item['id']} in {elapsed:.1f}s | RTF: {rtf:.3f} | Words: {word_count}")

        output_file = OUTPUT_DIR / f"{item['id']}_transcript.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_transcript)

        meta_file = OUTPUT_DIR / f"{item['id']}_meta.json"
        meta = {
            "id": item["id"],
            "title": item["title"],
            "duration_sec": round(duration_sec, 2),
            "transcription_time_sec": round(elapsed, 2),
            "rtf": round(rtf, 3),
            "word_count": word_count,
            "transcript_file": str(output_file.name)
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        corpus_summary.append(meta)

    summary_file = OUTPUT_DIR / "banking_transcripts_manifest.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(corpus_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"All transcripts generated successfully and saved to {summary_file}")

if __name__ == "__main__":
    main()

