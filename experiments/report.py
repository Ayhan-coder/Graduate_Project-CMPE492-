"""Aggregate ASR / LLM / ablation JSONL into tables. Does not invent numbers.

Reads completed run files and writes:
    outputs/asr_by_lang.json
    outputs/asr_2026.json
    outputs/ablation.json
    outputs/tables.md
    ../paper/tables/*.tex   (optional, --write-tex)

Usage:
    python report.py --asr outputs/asr-20260828.jsonl
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
    "gpt-4.1": "GPT-4.1",
    "gpt-4o-mini": "GPT-4o-mini",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
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


def summarize_asr(rows: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("skipped"):
            continue
        if "wer_norm" not in row:
            continue
        groups[(row.get("model", "?"), _lang(row))].append(row)

    by_model: dict[str, dict] = {}
    for (model, lang), items in groups.items():
        rec = by_model.setdefault(model, {"model": model, "langs": {}, "n_clips": 0})
        block = {k: mean([float(x[k]) for x in items if x.get(k) is not None]) for k in METRIC_KEYS}
        block["n"] = len(items)
        block["ids"] = [x.get("id") for x in items]
        rec["langs"][lang] = block
        rec["n_clips"] += len(items)

    for rec in by_model.values():
        pooled = {k: [] for k in METRIC_KEYS}
        for block in rec["langs"].values():
            for k in METRIC_KEYS:
                if block.get(k) is not None:
                    # unweighted over clips, not over languages
                    pass
        # recompute pooled over all clips of this model
        clips = [r for r in rows if r.get("model") == rec["model"] and not r.get("skipped") and "wer_norm" in r]
        rec["pooled"] = {k: mean([float(x[k]) for x in clips if x.get(k) is not None]) for k in METRIC_KEYS}
        rec["pooled"]["n"] = len(clips)
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
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{TR WER} & \textbf{TR CER} & \textbf{EN WER} & \textbf{EN CER} \\",
        r"\midrule",
    ]
    models = summary.get("models", {})
    for mid in model_ids:
        rec = models.get(mid, {})
        langs = rec.get("langs", {})
        name = DISPLAY.get(mid, mid)
        tr, en = langs.get("tr", {}), langs.get("en", {})
        lines.append(
            f"{name} & {fmt(tr.get('wer_norm'))} & {fmt(tr.get('cer_norm'))} "
            f"& {fmt(en.get('wer_norm'))} & {fmt(en.get('cer_norm'))} \\\\"
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
        r"\caption{Cross-layer ablation: the same LLM on transcripts from each ASR system. "
        r"R1\% and D1\% require human labels; term count is automatic.}",
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


def write_markdown(asr: dict | None, ablation: dict | None) -> str:
    lines = ["# Generated tables", "", "Numbers appear only if a JSONL run was passed in.", ""]
    if asr:
        lines.append("## ASR by language (normalized)")
        lines.append("")
        lines.append("| Model | Lang | n | WER | CER | latency (s) | RTF |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for mid, rec in asr["models"].items():
            for lang, block in rec.get("langs", {}).items():
                lines.append(
                    f"| {DISPLAY.get(mid, mid)} | {lang} | {block.get('n', 0)} | "
                    f"{fmt(block.get('wer_norm'))} | {fmt(block.get('cer_norm'))} | "
                    f"{fmt(block.get('latency_s'), 1)} | {fmt(block.get('rtf'))} |"
                )
        lines.append("")
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
    parser.add_argument("--asr", type=Path)
    parser.add_argument("--ablation", type=Path)
    parser.add_argument("--labels", type=Path, help="annotation CSV to attach R/D to ablation rows")
    parser.add_argument("--write-tex", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    asr_sum = summarize_asr(load_jsonl(args.asr)) if args.asr else None
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

    md = write_markdown(asr_sum, abl_sum)
    (OUT_DIR / "tables.md").write_text(md, encoding="utf-8")
    print(md)

    if args.write_tex:
        TEX_DIR.mkdir(parents=True, exist_ok=True)
        (TEX_DIR / "asr_bylang.tex").write_text(
            asr_lang_tex(asr_sum or {"models": {}}, PILOT_ASR,
                         "Normalized WER/CER by language (unweighted mean over clips).",
                         "tab:asr-lang"),
            encoding="utf-8",
        )
        (TEX_DIR / "asr_2026.tex").write_text(
            asr_lang_tex(asr_sum or {"models": {}}, REFRESH_ASR,
                         "2026 refresh, normalized WER/CER by language.",
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
