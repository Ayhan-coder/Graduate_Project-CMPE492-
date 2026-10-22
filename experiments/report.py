"""Aggregate ASR / LLM / ablation JSONL into tables. Does not invent numbers.

Reads completed run files and writes:
    outputs/asr_by_lang.json
    outputs/asr_2026.json
    outputs/ablation.json
    outputs/tables.md
    ../paper/tables/*.tex   (optional, --write-tex)

Usage:
    python report.py --asr outputs/asr-20260828.jsonl
    python report.py --asr outputs/asr-medium.jsonl outputs/asr-large-v3.jsonl
    python report.py --asr outputs/asr.jsonl --ablation outputs/ablation.jsonl --write-tex
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
TEX_DIR = ROOT.parent / "paper" / "tables"

METRIC_KEYS = ("wer_raw", "cer_raw", "wer_norm", "cer_norm", "latency_s", "rtf")
LANGS = ("tr", "en")

# Display names used in the paper tables.
DISPLAY = {
    "openai/whisper-medium": "Whisper Medium",
    "openai/whisper-large-v3": "Whisper Large-v3",
    "openai/whisper-large-v3-turbo": "Whisper Turbo",
    "facebook/wav2vec2-large-xlsr-53": "Wav2Vec2",
    "speechbrain/asr-crdnn-rnnlm-librispeech": "SpeechBrain",
    "Qwen/Qwen3-ASR-1.7B-hf": "Qwen3-ASR 1.7B",
    "Qwen/Qwen3-ASR-0.6B-hf": "Qwen3-ASR 0.6B",
    "selimc/whisper-large-v3-turbo-turkish": "Whisper Turbo TR",
    "nvidia/nemotron-3.5-asr-streaming-0.6b": "Nemotron 3.5 ASR",
    "gpt-4.1": "GPT-4.1",
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "Qwen/Qwen3.5-9B": "Qwen3.5-9B",
    "Qwen/Qwen3.5-4B": "Qwen3.5-4B",
    "google/gemma-4-E4B-it": "Gemma 4 8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "meta-llama/Llama-3.3-8B-Instruct": "Llama 3.3 8B",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B",
}

PILOT_ASR = (
    "openai/whisper-medium",
    "openai/whisper-large-v3",
    "facebook/wav2vec2-large-xlsr-53",
    "speechbrain/asr-crdnn-rnnlm-librispeech",
)
REFRESH_ASR = (
    "openai/whisper-large-v3-turbo",
    "Qwen/Qwen3-ASR-1.7B-hf",
    "Qwen/Qwen3-ASR-0.6B-hf",
    "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "selimc/whisper-large-v3-turbo-turkish",
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _lang(row: dict) -> str:
    lang = (row.get("lang") or "").strip().lower()
    if lang.startswith("tr"):
        return "tr"
    if lang.startswith("en"):
        return "en"
    vid = str(row.get("id") or row.get("video_id") or "")
    if vid.upper().startswith("TR"):
        return "tr"
    if vid.upper().startswith("EN"):
        return "en"
    return "unk"


def mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "---"
    return f"{value:.{digits}f}"


def load_clip_info(manifest_path: Path | None = None) -> dict[str, dict]:
    """Load word counts, duration, and language per clip from manifest."""
    path = manifest_path or (ROOT / "data" / "manifest.jsonl")
    if not path.exists():
        return {}
    info = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cid = row.get("id")
        ref = row.get("reference") or ""
        dur = float(row.get("duration_s", 0.0))
        info[cid] = {
            "words": len(ref.split()),
            "duration": dur,
            "lang": row.get("lang"),
        }
    return info


def bootstrap_ci(values: list[float], n_boot: int = 10000, seed: int = 42) -> tuple[float, float] | None:
    """Compute 95% bootstrap confidence interval for the mean (10,000 resamples)."""
    if not values:
        return None
    if len(values) == 1:
        return (values[0], values[0])
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    n = len(arr)
    boot_means = [rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)]
    low = float(np.percentile(boot_means, 2.5))
    high = float(np.percentile(boot_means, 97.5))
    return (low, high)


def _scored_asr_row(row: dict) -> bool:
    """A row that can enter a WER/CER mean. Missing-reference clips keep the key as null."""
    if row.get("skipped") or row.get("error"):
        return False
    return row.get("wer_norm") is not None


def summarize_asr(rows: list[dict], clip_info: dict | None = None, manifest_path: Path | None = None, n_boot: int = 10000) -> dict:
    if clip_info is None:
        clip_info = load_clip_info(manifest_path)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if not _scored_asr_row(row):
            continue
        groups[(row.get("model", "?"), _lang(row))].append(row)

    by_model: dict[str, dict] = {}
    for (model, lang), items in groups.items():
        rec = by_model.setdefault(model, {"model": model, "langs": {}, "n_clips": 0})
        block = {k: mean([float(x[k]) for x in items if x.get(k) is not None]) for k in METRIC_KEYS}
        block["n"] = len(items)
        block["ids"] = [x.get("id") for x in items]

        # Confidence interval on macro WER
        wers = [float(x["wer_norm"]) for x in items if x.get("wer_norm") is not None]
        block["wer_ci"] = bootstrap_ci(wers, n_boot=n_boot)
        block["wer_macro"] = block.get("wer_norm")

        # Micro (word-weighted) WER and duration-weighted WER
        if clip_info:
            valid = [x for x in items if x.get("id") in clip_info]
            tot_words = sum(clip_info[x["id"]]["words"] for x in valid)
            tot_dur = sum(clip_info[x["id"]]["duration"] for x in valid)
            if tot_words > 0:
                block["wer_micro"] = sum(
                    float(x["wer_norm"]) * clip_info[x["id"]]["words"]
                    for x in valid if x.get("wer_norm") is not None
                ) / tot_words
            else:
                block["wer_micro"] = block.get("wer_norm")
            if tot_dur > 0:
                block["wer_dur"] = sum(
                    float(x["wer_norm"]) * clip_info[x["id"]]["duration"]
                    for x in valid if x.get("wer_norm") is not None
                ) / tot_dur
            else:
                block["wer_dur"] = block.get("wer_norm")
        else:
            block["wer_micro"] = block.get("wer_norm")
            block["wer_dur"] = block.get("wer_norm")

        rec["langs"][lang] = block
        rec["n_clips"] += len(items)

    for rec in by_model.values():
        clips = [r for r in rows if r.get("model") == rec["model"] and _scored_asr_row(r)]
        rec["pooled"] = {k: mean([float(x[k]) for x in clips if x.get(k) is not None]) for k in METRIC_KEYS}
        rec["pooled"]["n"] = len(clips)
        wers_pooled = [float(x["wer_norm"]) for x in clips if x.get("wer_norm") is not None]
        rec["pooled"]["wer_ci"] = bootstrap_ci(wers_pooled)
        rec["pooled"]["wer_macro"] = rec["pooled"].get("wer_norm")
        if clip_info:
            valid_pooled = [x for x in clips if x.get("id") in clip_info]
            tw = sum(clip_info[x["id"]]["words"] for x in valid_pooled)
            if tw > 0:
                rec["pooled"]["wer_micro"] = sum(
                    float(x["wer_norm"]) * clip_info[x["id"]]["words"]
                    for x in valid_pooled if x.get("wer_norm") is not None
                ) / tw
            else:
                rec["pooled"]["wer_micro"] = rec["pooled"].get("wer_norm")
        else:
            rec["pooled"]["wer_micro"] = rec["pooled"].get("wer_norm")
    return {"models": by_model, "n_rows": len(rows)}


def attach_labels(rows: list[dict], labels_csv: Path) -> list[dict]:
    """Add R1/R0/D1/D0 counts to each ablation row from an annotation CSV."""
    from iaa import load_labels

    labels = load_labels(labels_csv)
    by_clip: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in labels.values():
        by_clip[(rec.get("video_id", ""), rec.get("model", ""))].append(rec)

    out = []
    for row in rows:
        rec = dict(row)
        tag = rec.get("asr_model") or ""
        llm = rec.get("model") or ""
        key_a = (rec.get("id", ""), f"{tag}|{llm}" if tag else llm)
        key_b = (rec.get("id", ""), llm)
        items = by_clip.get(key_a) or by_clip.get(key_b) or []
        if items:
            rec["R1"] = sum(x["R"] == 1 for x in items)
            rec["R0"] = sum(x["R"] == 0 for x in items)
            rec["D1"] = sum(x["D"] == 1 for x in items)
            rec["D0"] = sum(x["D"] == 0 for x in items)
        out.append(rec)
    return out


def summarize_ablation(rows: list[dict]) -> dict:
    """Group by ASR model x language. Expects fields from run_ablation.py."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        asr = row.get("asr_model") or row.get("source_asr") or "?"
        groups[(asr, _lang(row))].append(row)

    models: dict[str, dict] = {}
    for (asr, lang), items in groups.items():
        rec = models.setdefault(asr, {"asr_model": asr, "langs": {}})
        n_terms = sum(int(x.get("n_terms") or 0) for x in items)
        r1 = sum(int(x.get("R1") or 0) for x in items)
        r0 = sum(int(x.get("R0") or 0) for x in items)
        d1 = sum(int(x.get("D1") or 0) for x in items)
        d0 = sum(int(x.get("D0") or 0) for x in items)
        rec["langs"][lang] = {
            "n_clips": len(items),
            "n_terms": n_terms,
            "R1": r1,
            "R0": r0,
            "D1": d1,
            "D0": d0,
            "R1_pct": (100.0 * r1 / (r1 + r0)) if (r1 + r0) else None,
            "D1_pct": (100.0 * d1 / (d1 + d0)) if (d1 + d0) else None,
            "mean_latency_s": mean([float(x["latency_s"]) for x in items if x.get("latency_s") is not None]),
        }
    return {"models": models, "n_rows": len(rows), "llm": rows[0].get("model") if rows else None}


def asr_lang_tex(summary: dict, model_ids: tuple[str, ...], caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{TR (Macro / Micro)} & \textbf{TR 95\% CI} & \textbf{EN (Macro / Micro)} & \textbf{EN 95\% CI} \\",
        r"\midrule",
    ]
    models = summary.get("models", {})
    for mid in model_ids:
        rec = models.get(mid, {})
        langs = rec.get("langs", {})
        name = DISPLAY.get(mid, mid)
        tr, en = langs.get("tr", {}), langs.get("en", {})
        tr_macro = fmt(tr.get("wer_norm"))
        tr_micro = fmt(tr.get("wer_micro"))
        tr_ci = f"[{tr['wer_ci'][0]:.3f}, {tr['wer_ci'][1]:.3f}]" if tr.get("wer_ci") else "---"
        en_macro = fmt(en.get("wer_norm"))
        en_micro = fmt(en.get("wer_micro"))
        en_ci = f"[{en['wer_ci'][0]:.3f}, {en['wer_ci'][1]:.3f}]" if en.get("wer_ci") else "---"
        tr_val = f"{tr_macro} / {tr_micro}" if tr.get("wer_norm") is not None else "---"
        en_val = f"{en_macro} / {en_micro}" if en.get("wer_norm") is not None else "---"
        lines.append(
            f"{name} & {tr_val} & {tr_ci} & {en_val} & {en_ci} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def ablation_tex(summary: dict) -> str:
    llm = summary.get("llm") or "LLM"
    lines = [
        r"\begin{table}[t]",
        r"\caption{Cross-layer ablation: the same open-weight LLM (\texttt{Qwen3.5-4B}) on ASR hypotheses. "
        r"Whisper rows reflect human Annotator~1 ratings; extended checkpoints reflect validated LLM-as-a-judge scoring calibrated against human rubrics.}",
        r"\label{tab:ablation}",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{ASR} & \textbf{Lang.} & \textbf{Terms} & \textbf{R1\%} & \textbf{D1\%} \\",
        r"\midrule",
    ]
    models = summary.get("models", {})
    order = list(PILOT_ASR) + [m for m in models if m not in PILOT_ASR]
    any_row = False
    for mid in order:
        rec = models.get(mid)
        if not rec:
            continue
        for lang in LANGS:
            block = rec.get("langs", {}).get(lang)
            if not block:
                continue
            any_row = True
            r1 = fmt(block.get("R1_pct"), 1) if block.get("R1_pct") is not None else "---"
            d1 = fmt(block.get("D1_pct"), 1) if block.get("D1_pct") is not None else "---"
            terms = block.get("n_terms", "---")
            lines.append(
                f"{DISPLAY.get(mid, mid)} & {lang.upper()} & {terms} & {r1} & {d1} \\\\"
            )
    if not any_row:
        lines.append(r"\multicolumn{5}{c}{\textit{pending --- run run\_ablation.py}} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def pending_asr_lang_tex() -> str:
    return asr_lang_tex({"models": {}}, PILOT_ASR,
                        "Normalized WER/CER by language (unweighted mean over clips). "
                        r"Filled by \texttt{report.py} after \texttt{run\_asr.py}; "
                        "pooled values in Table~\\ref{tab:asr} are unchanged.",
                        "tab:asr-lang")


def pending_asr_2026_tex() -> str:
    return asr_lang_tex({"models": {}}, REFRESH_ASR,
                        "2026 refresh, normalized WER/CER by language. "
                        "Empty until the seven videos are re-run.",
                        "tab:asr-2026")


def pending_ablation_tex() -> str:
    return ablation_tex({"models": {}, "llm": "GPT-4o-mini"})


def paper_blockers(manifest_path: Path | None = None) -> list[str]:
    """Reasons the IEEE tables must stay empty. Caption drafts are not gold."""
    path = manifest_path or (ROOT / "data" / "manifest.jsonl")
    if not path.exists():
        return [f"manifest missing: {path}"]
    blockers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row.get("id", "?")
        if not row.get("confirmed"):
            blockers.append(f"{cid}: confirmed=false")
        src = row.get("reference_source") or "missing"
        if src != "gold":
            blockers.append(f"{cid}: reference_source={src} (need gold)")
        if src == "gold":
            from gold_io import resolve_reference
            if not resolve_reference(row):
                blockers.append(f"{cid}: gold text is empty")
    return blockers


def write_clip_markdown(rows: list[dict]) -> str:
    """One row per clip. Used as a lab diagnostic, not an IEEE table."""
    lines = [
        "## Per-clip (lab diagnostic)",
        "",
        "| id | model | lang | ref | confirmed | WER_n | CER_n | RTF |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("skipped"):
            continue
        mid = DISPLAY.get(row.get("model", "?"), row.get("model", "?"))
        err = row.get("error")
        wer = "ERR" if err else fmt(row.get("wer_norm"))
        cer = "ERR" if err else fmt(row.get("cer_norm"))
        rtf = "ERR" if err else fmt(row.get("rtf"))
        lines.append(
            f"| {row.get('id', '?')} | {mid} | {_lang(row)} | "
            f"{row.get('reference_source') or 'missing'} | "
            f"{bool(row.get('confirmed'))} | {wer} | {cer} | {rtf} |"
        )
    return "\n".join(lines) + "\n"


def write_markdown(asr: dict | None, ablation: dict | None, asr_rows: list[dict] | None = None) -> str:
    lines = ["# Generated tables", "", "Numbers appear only if a JSONL run was passed in.", ""]
    if asr:
        lines.append("## ASR by language (normalized)")
        lines.append("")
        lines.append("| Model | Lang | n | Macro WER | 95% CI (Bootstrap) | Micro WER | CER | latency (s) | RTF |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for mid, rec in asr["models"].items():
            for lang, block in rec.get("langs", {}).items():
                ci = block.get("wer_ci")
                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "---"
                lines.append(
                    f"| {DISPLAY.get(mid, mid)} | {lang} | {block.get('n', 0)} | "
                    f"{fmt(block.get('wer_norm'))} | {ci_str} | {fmt(block.get('wer_micro'))} | "
                    f"{fmt(block.get('cer_norm'))} | {fmt(block.get('latency_s'), 1)} | {fmt(block.get('rtf'))} |"
                )
        lines.append("")
        if asr_rows:
            lines.append(write_clip_markdown(asr_rows))
    else:
        lines.append("No `--asr` file given.")
        lines.append("")
    if ablation:
        lines.append("## Ablation")
        lines.append("")
        lines.append(f"LLM: {ablation.get('llm')}")
        lines.append("")
        lines.append("| ASR | Lang | Terms | R1% | D1% |")
        lines.append("|---|---|---:|---:|---:|")
        for mid, rec in ablation["models"].items():
            for lang, block in rec.get("langs", {}).items():
                lines.append(
                    f"| {DISPLAY.get(mid, mid)} | {lang} | {block.get('n_terms', 0)} | "
                    f"{fmt(block.get('R1_pct'), 1)} | {fmt(block.get('D1_pct'), 1)} |"
                )
    else:
        lines.append("No `--ablation` file given.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr", type=Path, nargs="+",
                        help="one or more run_asr.py JSONL files")
    parser.add_argument("--ablation", type=Path)
    parser.add_argument("--labels", type=Path, help="annotation CSV to attach R/D to ablation rows")
    parser.add_argument("--write-tex", action="store_true",
                        help="write paper/tables/*.tex (refused unless clips are confirmed gold)")
    parser.add_argument("--allow-unconfirmed", action="store_true",
                        help="override: write TeX even if confirmed=false or reference is not gold")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.jsonl")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    asr_rows = None
    if args.asr:
        asr_rows = []
        for path in args.asr:
            asr_rows.extend(load_jsonl(path))
    asr_sum = summarize_asr(asr_rows) if asr_rows is not None else None
    abl_sum = None
    if args.ablation:
        abl_rows = load_jsonl(args.ablation)
        if args.labels:
            abl_rows = attach_labels(abl_rows, args.labels)
        abl_sum = summarize_ablation(abl_rows)

    if asr_sum:
        (OUT_DIR / "asr_by_lang.json").write_text(json.dumps(asr_sum, indent=2), encoding="utf-8")
    if abl_sum:
        (OUT_DIR / "ablation.json").write_text(json.dumps(abl_sum, indent=2), encoding="utf-8")

    md = write_markdown(asr_sum, abl_sum, asr_rows)
    blockers = paper_blockers(args.manifest)
    if blockers:
        banner = (
            "CANDIDATE / NOT FOR THE IEEE PAPER\n"
            "Clips are unconfirmed or the reference is not gold. "
            "Do not paste these cells into tab:asr-lang, tab:asr-2026, or tab:ablation.\n"
            + "\n".join(f"- {b}" for b in blockers)
            + "\n\n"
        )
        md = banner + md
    (OUT_DIR / "tables.md").write_text(md, encoding="utf-8")
    print(md)

    if args.write_tex:
        if blockers and not args.allow_unconfirmed:
            raise SystemExit(
                "refusing --write-tex: clips are unconfirmed or reference is not gold.\n"
                + "\n".join(blockers)
                + "\nPass --allow-unconfirmed only for a local scratch copy, never the paper."
            )
        TEX_DIR.mkdir(parents=True, exist_ok=True)
        (TEX_DIR / "asr_bylang.tex").write_text(
            asr_lang_tex(asr_sum or {"models": {}}, PILOT_ASR,
                         r"Normalized WER by language: Macro mean / Micro (word-weighted) mean, with 95\% bootstrap confidence intervals in brackets ($N=10{,}000$ resamples). GPU gold-rescored refresh; pooled CPU-pilot values in Table~\ref{tab:asr} are unchanged.",
                         "tab:asr-lang"),
            encoding="utf-8",
        )
        (TEX_DIR / "asr_2026.tex").write_text(
            asr_lang_tex(asr_sum or {"models": {}}, REFRESH_ASR,
                         r"2026 refresh, normalized WER by language: Macro mean / Micro (word-weighted) mean with 95\% bootstrap confidence intervals in brackets. TR-matched Turbo is scored on TR only.",
                         "tab:asr-2026"),
            encoding="utf-8",
        )
        (TEX_DIR / "ablation.tex").write_text(
            ablation_tex(abl_sum or {"models": {}, "llm": "GPT-4o-mini"}),
            encoding="utf-8",
        )
        print("wrote", TEX_DIR)


if __name__ == "__main__":
    main()
