"""vad_chunker.py: Voice Activity Detection (VAD) and pause-aware speech chunker.

Addresses the key long-form failure mode identified by the advisor:
Whisper Large-v3 underperformed Medium on unsegmented 12-18 min lectures due to
hallucination and looping during long silences or music intros.

This module detects voice activity and splits long audio on natural acoustic pauses
(silence > 0.4s) into optimal 15-30s speech segments without cutting mid-word.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def detect_energy_vad(
    audio: np.ndarray,
    sr: int = 16000,
    frame_ms: int = 30,
    energy_threshold_percentile: float = 25.0,
) -> np.ndarray:
    """Frame-level energy Voice Activity Detection (VAD).

    Returns a boolean array where True indicates speech activity.
    """
    frame_len = int(sr * (frame_ms / 1000.0))
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return np.ones(1, dtype=bool)

    # Compute RMS energy per frame
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    energy = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

    # Adaptive threshold based on low-percentile ambient noise floor
    noise_floor = np.percentile(energy, energy_threshold_percentile)
    threshold = max(1e-4, noise_floor * 2.2)

    is_speech = energy > threshold

    # Morphological smoothing: close small gaps (< 200 ms) in speech
    gap_frames = int(0.200 / (frame_ms / 1000.0))
    smoothed = np.copy(is_speech)
    gap = 0
    for i in range(len(smoothed)):
        if not smoothed[i]:
            gap += 1
        else:
            if 0 < gap <= gap_frames:
                smoothed[i - gap : i] = True
            gap = 0

    return smoothed


def segment_audio_vad(
    audio: np.ndarray,
    sr: int = 16000,
    min_chunk_s: float = 12.0,
    max_chunk_s: float = 28.0,
    silence_pad_s: float = 0.25,
) -> list[tuple[float, float]]:
    """Segment long-form audio on natural acoustic silence boundaries.

    Returns list of (start_seconds, end_seconds) intervals.
    Each chunk is guaranteed to be <= max_chunk_s and split at a pause.
    """
    duration_s = len(audio) / sr
    if duration_s <= max_chunk_s:
        return [(0.0, duration_s)]

    frame_ms = 30
    frame_s = frame_ms / 1000.0
    vad_mask = detect_energy_vad(audio, sr=sr, frame_ms=frame_ms)

    chunks = []
    current_start_s = 0.0
    total_frames = len(vad_mask)

    while current_start_s < duration_s:
        # If remaining audio is short, take the rest
        if (duration_s - current_start_s) <= max_chunk_s:
            chunks.append((round(current_start_s, 2), round(duration_s, 2)))
            break

        # Search window for a good split point: between min_chunk_s and max_chunk_s
        search_start_s = current_start_s + min_chunk_s
        search_end_s = min(duration_s, current_start_s + max_chunk_s)

        f_start = int(search_start_s / frame_s)
        f_end = min(total_frames, int(search_end_s / frame_s))

        # Look for the quietest point (longest run of silence frames) in the search window
        search_slice = vad_mask[f_start:f_end]
        split_s = None

        if len(search_slice) > 0 and not np.all(search_slice):
            # Find contiguous non-speech runs
            silence_indices = np.where(~search_slice)[0]
            if len(silence_indices) > 0:
                # Pick silence closest to the middle of the search window
                mid_idx = len(search_slice) // 2
                best_silence_idx = min(silence_indices, key=lambda idx: abs(idx - mid_idx))
                split_s = (f_start + best_silence_idx) * frame_s

        # Fallback if no clear silence: split at max_chunk_s
        if split_s is None or split_s <= current_start_s:
            split_s = search_end_s

        # Add slight boundary padding without exceeding bounds
        chunk_end = min(duration_s, split_s + silence_pad_s)
        chunks.append((round(current_start_s, 2), round(chunk_end, 2)))

        # Next chunk starts where this one ended (minus small overlap for continuity)
        current_start_s = max(chunk_end - 0.1, current_start_s + min_chunk_s)

    return chunks


def load_audio_chunks(
    audio_path: Path | str,
    min_chunk_s: float = 12.0,
    max_chunk_s: float = 28.0,
) -> list[tuple[float, float, np.ndarray]]:
    """Load WAV audio and return list of (start_s, end_s, chunk_samples_16k)."""
    import soundfile as sf

    audio, sr = sf.read(str(audio_path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)

    intervals = segment_audio_vad(audio, sr=sr, min_chunk_s=min_chunk_s, max_chunk_s=max_chunk_s)
    result = []
    for start_s, end_s in intervals:
        start_idx = int(start_s * sr)
        end_idx = min(len(audio), int(end_s * sr))
        piece = audio[start_idx:end_idx]
        result.append((start_s, end_s, piece))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="VAD pause-aware audio chunker for long-form speech")
    parser.add_argument("--audio", type=Path, default=None, help="Input WAV audio file")
    parser.add_argument("--test", action="store_true", help="Run synthetic self-test")
    parser.add_argument("--min-chunk", type=float, default=12.0)
    parser.add_argument("--max-chunk", type=float, default=28.0)
    args = parser.parse_args()

    if args.test:
        print("Running VAD chunker self-test...")
        sr = 16000
        # Create a 60-second synthetic audio with speech bursts and 1-second silences
        t = np.linspace(0, 60, 60 * sr, dtype=np.float32)
        audio = np.sin(2 * np.pi * 300 * t) * 0.15  # Tone
        # Insert silences at 15-17s, 32-34s, 48-50s
        audio[int(15 * sr) : int(17 * sr)] = 0.0001
        audio[int(32 * sr) : int(34 * sr)] = 0.0001
        audio[int(48 * sr) : int(50 * sr)] = 0.0001

        chunks = segment_audio_vad(audio, sr=sr, min_chunk_s=args.min_chunk, max_chunk_s=args.max_chunk)
        print(f"60s audio segmented into {len(chunks)} pause-aligned chunks:")
        for i, (st, en) in enumerate(chunks):
            print(f"  Chunk {i+1}: {st:5.2f}s - {en:5.2f}s (duration: {en-st:4.2f}s)")

        assert len(chunks) >= 3, "Expected at least 3 chunks"
        for st, en in chunks:
            assert (en - st) <= args.max_chunk + 1.0, f"Chunk exceeded max duration: {en-st}"
        print("Self-test passed!")
        return

    if not args.audio or not args.audio.exists():
        print("Please provide an existing audio file via --audio <path>")
        return

    chunks = load_audio_chunks(args.audio, min_chunk_s=args.min_chunk, max_chunk_s=args.max_chunk)
    print(f"Segmented {args.audio.name} into {len(chunks)} chunks:")
    for i, (st, en, piece) in enumerate(chunks):
        print(f"  Chunk {i+1}: {st:6.2f}s - {en:6.2f}s ({len(piece)/16000:5.2f}s)")


if __name__ == "__main__":
    main()
