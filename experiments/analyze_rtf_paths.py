"""Reconcile the RTF numbers that live on different measurement paths.

Two RTF families circulate in the report and they are not comparable as written:

1. `outputs/asr-20260903-turbo.jsonl` — 7 gold pilot clips, per-language aggregates
   (this is the source of the "0.025-0.044" range in Section 6.1).
2. `outputs/transcripts/machine_only/*.json` — full-length lecture audio written by
   `data/run_machine_transcripts.py` (same model, same sequential-generate decoding).
3. `outputs/srm_asr_measured.json` — tiny/base/small + Qwen3-ASR-0.6B over 6 gold clips,
   where `mean_rtf` is an unweighted per-clip mean and each model's first clip is cold.

The difference is mostly *when the clock starts*: `run_asr.transcribe_whisper()` caches
weights per process, so the first clip of every run also pays the weights load, CUDA
context init and the HF hub revision check. On a 4-7 minute clip that fixed cost is
several times the inference itself, which is why single-clip smoke runs look slow.

This script fits `latency = alpha + beta * audio_seconds` over the full-audio records,
reports the distribution over clips long enough that the fixed cost is negligible, and
writes all three families side by side to `outputs/rtf_path_audit.json`. GPU-free: it only
reads artifacts already on disk.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TRANSCRIPTS = OUT / "transcripts" / "machine_only"
PILOT_RUN = OUT / "asr-20260903-turbo.jsonl"
SRM_RUN = OUT / "srm_asr_measured.json"
LOGS_DIR = ROOT / "logs"

CLIP_HEADER = re.compile(r"^\[\d+/\d+\] ([A-Z]{2}-\d+[A-Za-z0-9\-]*):")


def cold_start_ids(extra: list[str] | None = None) -> set[str]:
    """Clip ids whose timing includes the model load of their process.

    Union across every logged run: if a clip was the first clip of any process, the
    record it left on disk may carry that cost, so it is excluded from the warm set.
    Single-clip smoke runs print to the console instead of a log, so they are passed
    in through `extra`.
    """
    cold: set[str] = set(extra or [])
    for log in sorted(LOGS_DIR.glob("*.log")):
        current = None
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = CLIP_HEADER.match(line)
            if m:
                current = m.group(1)
            elif current and "loading whisper" in line:
                cold.add(current)
    return cold


def ols(pairs: list[tuple[float, float]]) -> tuple[float, float, float]:
    n = len(pairs)
    sx = sum(x for x, _ in pairs)
    sy = sum(y for _, y in pairs)
    sxx = sum(x * x for x, _ in pairs)
    sxy = sum(x * y for x, y in pairs)
    beta = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    alpha = (sy - beta * sx) / n
    ybar = sy / n
    ss_res = sum((y - (alpha + beta * x)) ** 2 for x, y in pairs)
    ss_tot = sum((y - ybar) ** 2 for x, y in pairs)
    return alpha, beta, 1 - ss_res / ss_tot


def describe(rtfs: list[float], lat: list[float], dur: list[float]) -> dict:
    srt = sorted(rtfs)
    mid = srt[len(srt) // 2] if len(srt) % 2 else (srt[len(srt) // 2 - 1] + srt[len(srt) // 2]) / 2
    return {
        "n": len(rtfs),
        "min": round(min(rtfs), 4),
        "median": round(mid, 4),
        "max": round(max(rtfs), 4),
        "pooled_sum_latency_over_sum_audio": round(sum(lat) / sum(dur), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="RTF measurement-path audit (reads artifacts, no GPU).")
    ap.add_argument("--cold", nargs="*", default=["TR-63"],
                    help="extra clip ids whose record was the first clip of its process")
    args = ap.parse_args()

    cold_ids = cold_start_ids(args.cold)
    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(TRANSCRIPTS.glob("*.json"))]
    rows = [r for r in records if r.get("rtf") and r.get("duration_s")]
    cold_rows = [r for r in rows if r["id"] in cold_ids]

    alpha, beta, r2 = ols([(r["duration_s"], r["latency_s"]) for r in rows])
    # The headline statistic is a duration band, not an exclusion list: some transcripts
    # predate the surviving logs, so "was this the first clip of its process?" cannot
    # always be answered from disk. Over 600 s of audio the fixed cost moves RTF by
    # less than 0.01, so that band is the model speed.
    long_clips = [r for r in rows if r["duration_s"] >= 600]
    short_clips = [r for r in rows if r["duration_s"] < 420]
    full_audio = {
        "path": "data/run_machine_transcripts.py -> outputs/transcripts/machine_only/*.json",
        "model": "openai/whisper-large-v3-turbo",
        "decoding": "sequential generate, num_beams=1, temperature=0.0, chunk 30 s, no VAD",
        "records": len(rows),
        "first_clip_of_logged_process_ids": sorted(r["id"] for r in cold_rows),
        "first_clip_ids_from_cli": sorted(args.cold),
        "fit_latency_equals_alpha_plus_beta_times_audio": {
            "alpha_fixed_cost_s": round(alpha, 2),
            "beta_steady_state_rtf": round(beta, 4),
            "r_squared": round(r2, 4),
            "n": len(rows),
        },
        "clips_over_600s": describe([r["rtf"] for r in long_clips],
                                    [r["latency_s"] for r in long_clips],
                                    [r["duration_s"] for r in long_clips]),
        "clips_over_600s_by_lang": {
            lang: describe([r["rtf"] for r in long_clips if r["lang"] == lang],
                           [r["latency_s"] for r in long_clips if r["lang"] == lang],
                           [r["duration_s"] for r in long_clips if r["lang"] == lang])
            for lang in ("tr", "en")
        },
        "clips_under_420s": describe([r["rtf"] for r in short_clips],
                                     [r["latency_s"] for r in short_clips],
                                     [r["duration_s"] for r in short_clips]),
        "first_clip_of_process": describe([r["rtf"] for r in cold_rows],
                                          [r["latency_s"] for r in cold_rows],
                                          [r["duration_s"] for r in cold_rows]) if cold_rows else None,
        "reading": "per-clip RTF on short audio is the fixed cost divided by a small number, "
                   "not model speed; cite the fit or the over-600 s band",
    }

    pilot = []
    if PILOT_RUN.exists():
        for line in PILOT_RUN.read_text(encoding="utf-8").splitlines():
            if line.strip():
                pilot.append(json.loads(line))
    pilot_rows = []
    for r in pilot:
        dur = r["latency_s"] / r["rtf"]  # run_asr does not store duration_s; derived
        pilot_rows.append({"id": r["id"], "lang": r.get("lang"),
                           "latency_s": round(r["latency_s"], 1),
                           "duration_s_derived": round(dur, 1),
                           "rtf": round(r["rtf"], 4), "is_first_row_of_run": r is pilot[0]})
    by_lang = {}
    by_lang_excl_first = {}
    for lang in ("tr", "en"):
        sub = [p for p in pilot_rows if p["lang"] == lang]
        by_lang[lang] = describe([p["rtf"] for p in sub],
                                 [p["latency_s"] for p in sub],
                                 [p["duration_s_derived"] for p in sub])
        warm_sub = [p for p in sub if not p["is_first_row_of_run"]]
        by_lang_excl_first[lang] = describe([p["rtf"] for p in warm_sub],
                                            [p["latency_s"] for p in warm_sub],
                                            [p["duration_s_derived"] for p in warm_sub])
    pilot_block = {
        "path": "run_asr.py -> outputs/asr-20260903-turbo.jsonl (source of report Section 6.1)",
        "clip_set": "7 gold pilot clips (TR-1..3, EN-1..4)",
        "note": "aggregates include the run's first clip, which carries the weights load; "
                "duration_s is derived as latency/rtf because the run file does not store it",
        "per_clip": pilot_rows,
        "per_lang": by_lang,
        "per_lang_excluding_first_row": by_lang_excl_first,
        "range_cited_in_report": [min(p["rtf"] for p in pilot_rows),
                                  max(p["rtf"] for p in pilot_rows)],
    }

    # Third path: the SRM catalog rows (whisper-tiny/base/small + Qwen3-ASR-0.6B over 6 gold
    # clips, `outputs/srm_asr_measured.json`). Its mean_rtf is an unweighted per-clip mean and
    # each model is loaded inside its own loop section, so every model's first clip is cold.
    srm_block = None
    if SRM_RUN.exists():
        srm = json.loads(SRM_RUN.read_text(encoding="utf-8"))
        per_model = {}
        for res in srm.get("results", []):
            rtfs = {p["id"]: p["rtf"] for p in res.get("per_clip", []) if p.get("rtf")}
            per_model[res["key"]] = {
                "model": res["model"],
                "params_m": res.get("params_m"),
                "mean_rtf_stored_unweighted_per_clip": res.get("mean_rtf"),
                "pooled_wall_over_audio": (round(res["wall_seconds"] / res["audio_seconds"], 4)
                                           if res.get("wall_seconds") and res.get("audio_seconds") else None),
                "clips": rtfs,
            }
        inv = None
        if "tiny" in per_model and "small" in per_model:
            t, s = per_model["tiny"]["clips"], per_model["small"]["clips"]
            shared = sorted(set(t) & set(s))
            inv = {
                "clip_ids": shared,
                "tiny_slower_on": sum(1 for c in shared if t[c] > s[c]),
                "of": len(shared),
                "tiny_mean": per_model["tiny"]["mean_rtf_stored_unweighted_per_clip"],
                "small_mean": per_model["small"]["mean_rtf_stored_unweighted_per_clip"],
                "reading": "39M tiny vs 244M small: RTF is not monotone in parameter count at "
                           "n=6 clips. Generated token counts were not logged, so the cause "
                           "(repetition/hallucination loops fill the 440-token budget) is "
                           "unverified; these rows are not a speed ranking.",
            }
        srm_block = {
            "path": "benchmark_srm_asr_local.py -> outputs/srm_asr_measured.json (report catalog Section 8.x)",
            "clip_set": f"{len(srm.get('clip_ids', []))} gold clips: {srm.get('clip_ids')}",
            "metadata_caveat": "the stored decoding block records chunk_length_s=30.0, but "
                               "run_asr.transcribe_whisper() never reads args.chunk_s; the value "
                               "is descriptive metadata, not an applied setting",
            "per_model": per_model,
            "size_speed_inversion": inv,
        }

    payload = {
        "full_audio_path": full_audio,
        "pilot_gold_path": pilot_block,
        "srm_catalog_path": srm_block,
        "conclusion": {
            "turbo_rtf_full_audio_over_600s": full_audio["clips_over_600s"]["pooled_sum_latency_over_sum_audio"],
            "turbo_rtf_steady_state_fit": round(beta, 4),
            "fixed_cost_per_call_s": round(alpha, 2),
            "pilot_gold_per_lang": {k: v["pooled_sum_latency_over_sum_audio"] for k, v in by_lang.items()},
            "pilot_gold_per_lang_excluding_first_row": {k: v["pooled_sum_latency_over_sum_audio"] for k, v in by_lang_excl_first.items()},
        },
    }
    out = OUT / "rtf_path_audit.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)
    print(f"full-audio: n={full_audio['records']} over-600s pooled={full_audio['clips_over_600s']['pooled_sum_latency_over_sum_audio']} "
          f"fit alpha={alpha:.1f}s beta={beta:.4f} R2={r2:.3f} first-clip={full_audio['first_clip_of_logged_process_ids']}")
    for lang, blk in by_lang.items():
        print(f"pilot {lang}: pooled={blk['pooled_sum_latency_over_sum_audio']} min={blk['min']} max={blk['max']} "
              f"| excl first row={by_lang_excl_first[lang]['pooled_sum_latency_over_sum_audio']}")
    if srm_block and srm_block["size_speed_inversion"]:
        i = srm_block["size_speed_inversion"]
        print(f"srm: tiny mean={i['tiny_mean']} small mean={i['small_mean']} "
              f"tiny slower on {i['tiny_slower_on']}/{i['of']} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
