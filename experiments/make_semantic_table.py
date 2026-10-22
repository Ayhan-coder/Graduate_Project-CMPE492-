"""make_semantic_table.py: regenerate paper/tables/sem.tex (Table ref{tab:sem}) from artifacts.

Column policy -- a cell is printed only if a file on disk produces it:

  Terms    : structured run artifacts (JSONL with a ``terms`` array) counted per clip, or,
             for the historical commercial-API pilot, numbered/bold headings parsed out of
             ``outputs/pilot_baseline/semantic_outputs/<lang>/<model>/*.txt``.
  R1/D1    : a per-item label sheet. Human annotator 1 -> annotation/labels_a1.csv.
             Real rubric judge -> labels_ablation_real_judge.csv / labels_evren_judge.csv.
             Where no sheet exists the cell is printed as an em dash, never estimated.

Two columns that the hand-written table carried are dropped outright:
  * B1+ ratio -- no code in this repository computes it and no artifact stores it
    (grep for b1_plus/b1_ratio/b1plus returns only prose files).
  * BLEU     -- only ``outputs/pilot_baseline/semantic_bleu_en.csv`` exists, it covers a
    single model and 24 of its 56 English terms, and sacreBLEU corpus scores in it are
    1.08--5.58, i.e. 0.011--0.056 as a fraction, not the 0.57/0.61 previously printed.
    ``make_bleu_table.py`` (not this script) is the place to publish it.

Run:  python experiments/make_semantic_table.py
"""

from __future__ import annotations

import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLES = ROOT.parent / "paper" / "tables"
OUT = TABLES / "sem.tex"

PILOT = ROOT / "outputs" / "pilot_baseline" / "semantic_outputs"
EVREN_QWEN = ROOT / "outputs" / "llm-20261010-214422.jsonl"
EVREN_DEEPSEEK = ROOT / "outputs" / "llm-20261010-214130.jsonl"
ABLATION = ROOT / "outputs" / "ablation-all.jsonl"
NINE_B = ROOT / "outputs" / "llm-20260913-210556.jsonl"
HUMAN = ROOT / "annotation" / "labels_a1.csv"
JUDGE_EVREN = ROOT / "annotation" / "labels_evren_judge.csv"

# Pilot markdown appears in three different list layouts.
ITEM_PATTERNS = [
    re.compile(r"(?m)^[ \t]*\*\*[ \t]*\d+\.?"),
    re.compile(r"(?m)^[ \t]*\d+\.[ \t]*\*\*"),
    re.compile(r"(?m)^[ \t]*#{2,4}[ \t]*\d+\.[ \t]*\*\*"),
]

DASH = r"--"


def count_pilot_terms(lang: str, model: str) -> int:
    files = sorted(glob.glob(str(PILOT / lang / model / "*.txt")))
    if not files:
        raise SystemExit(f"no pilot output files for {lang}/{model}")
    total = 0
    for p in files:
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        total += max(len(pat.findall(text)) for pat in ITEM_PATTERNS)
    return total


def count_jsonl_terms(path: Path, lang_pred) -> int:
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if lang_pred(rec):
            total += len(rec.get("terms", []))
    return total


def lang_is_tr(rec: dict) -> bool:
    v = str(rec.get("lang", "")).lower()
    return v.startswith("tr") or v.startswith("turk")


def lang_is_en(rec: dict) -> bool:
    return not lang_is_tr(rec) and (str(rec.get("lang", "")).lower().startswith("en")
                                    or str(rec.get("lang", "")).lower().startswith("engl"))


def load_sheet(path: Path, split_model: bool) -> dict[tuple[str, str], list[int]]:
    """Return {(model_key, lang): [n, R1, D1]} from a label CSV."""
    agg: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            model = row["model"].split("|")[0] if split_model else row["model"]
            lang = "TR" if str(row["lang"]).lower()[:2] == "tr" else "EN"
            try:
                r, d = int(row["R"]), int(row["D"])
            except (KeyError, TypeError, ValueError):
                continue
            a = agg[(model, lang)]
            a[0] += 1
            a[1] += r
            a[2] += d
    return agg


def fmt(agg, key, expected_terms):
    if key not in agg:
        return DASH, DASH, None
    n, r1, d1 = agg[key]
    if expected_terms is not None and n != expected_terms:
        raise SystemExit(f"{key}: sheet has {n} pairs, artifact emitted {expected_terms} terms")
    return (
        f"{r1}/{n - r1} ({100.0 * r1 / n:.1f}\\%)",
        f"{d1}/{n - d1} ({100.0 * d1 / n:.1f}\\%)",
        n,
    )


def main() -> None:
    # labels_a1.csv stores the pair as 'asr|llm'; keys below use that exact string.
    human = load_sheet(HUMAN, split_model=False)
    evren = load_sheet(JUDGE_EVREN, split_model=False)

    q4_tr = count_jsonl_terms(ABLATION, lambda r: r.get("asr_model") == "openai/whisper-large-v3" and lang_is_tr(r))
    q4_en = count_jsonl_terms(ABLATION, lambda r: r.get("asr_model") == "openai/whisper-large-v3" and lang_is_en(r))
    q9_tr = count_jsonl_terms(NINE_B, lang_is_tr)
    eq_tr = count_jsonl_terms(EVREN_QWEN, lang_is_tr)
    eq_en = count_jsonl_terms(EVREN_QWEN, lang_is_en)
    ed_tr = count_jsonl_terms(EVREN_DEEPSEEK, lang_is_tr)
    ed_en = count_jsonl_terms(EVREN_DEEPSEEK, lang_is_en)

    rows = []

    def add(model, lang, terms, rcell, dcell, source):
        rows.append((model, lang, terms, rcell, dcell, source))

    # Commercial API pilot: term counts reproducible from stored markdown, no R/D sheet.
    add("GPT-4.1", "EN", count_pilot_terms("en", "gpt-4.1"), DASH, DASH, "Pilot terms")
    add("GPT-4o-mini", "EN", count_pilot_terms("en", "gpt-4o-mini"), DASH, DASH, "Pilot terms")
    add("GPT-4.1", "TR", count_pilot_terms("tr", "gpt-4.1"), DASH, DASH, "Pilot terms")
    add("GPT-4o-mini", "TR", count_pilot_terms("tr", "gpt-4o-mini"), DASH, DASH, "Pilot terms")

    # Local open-weight runs.
    r, d, _ = fmt(human, ("openai/whisper-large-v3|Qwen/Qwen3.5-4B", "EN"), q4_en)
    add("Qwen3.5-4B", "EN", q4_en, r, d, "Human A1")
    r, d, _ = fmt(human, ("openai/whisper-large-v3|Qwen/Qwen3.5-4B", "TR"), q4_tr)
    add("Qwen3.5-4B", "TR", q4_tr, r, d, "Human A1")
    add("Qwen3.5-9B", "TR", q9_tr, DASH, DASH, "No sheet")

    # EVREN runs, judged by real closed-form rubric calls.
    for model, terms, key in (
        ("EVREN Qwen 3.8 Flash", eq_en, ("evren/qwen3.8-flash-next", "EN")),
        ("EVREN Qwen 3.8 Flash", eq_tr, ("evren/qwen3.8-flash-next", "TR")),
        ("EVREN DeepSeek V4.1", ed_en, ("evren/deepseek-v4.1-flash", "EN")),
        ("EVREN DeepSeek V4.1", ed_tr, ("evren/deepseek-v4.1-flash", "TR")),
    ):
        r, d, _ = fmt(evren, key, terms)
        add(model, key[1], terms, r, d, "LLM judge")

    lines = [
        "% GENERATED by experiments/make_semantic_table.py -- do not edit by hand.",
        "% Terms recomputed from run artifacts / pilot markdown; R1/D1 only where a label sheet exists.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\small",
        r"\begin{tabular}{llccll}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Lang.} & \textbf{Terms} & \textbf{$R1/R0$ ($R1\%$)} & \textbf{$D1/D0$ ($D1\%$)} & \textbf{Labels} \\",
        r"\midrule",
    ]
    for model, lang, terms, rc, dc, src in rows:
        lines.append(f"{model} & {lang} & {terms} & {rc} & {dc} & {src} \\\\")
    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{p{0.95\textwidth}}{\footnotesize $R1\% = R1/(R1{+}R0)$ and $D1\% = D1/(D1{+}D0)$",
        r"are \emph{precision over emitted terms}: an em dash means no per-item rating sheet exists for",
        r"that cell, not a zero score. Human A1 is \texttt{labels\_a1.csv}; the \texttt{Qwen3.5-4B} rows",
        r"shown here are its Whisper Large-v3 cells, while Table~\ref{tab:ablation} reports all six",
        r"human-rated Whisper cells. LLM judge is closed-form rubric output from the local",
        r"\texttt{Qwen3.5-4B} judge (\texttt{labels\_evren\_judge.csv}), whose unquantified leniency makes",
        r"these cells upper bounds (Table~\ref{tab:llm_judge} and \S\ref{sec:llm-judge}). The",
        r"\texttt{Qwen3.5-9B} row covers only the TR-1/TR-2 subset. Pilot term counts for the",
        r"commercial-API rows are recovered by parsing the stored Markdown lists, which reproduces",
        r"$105$/$56$/$78$/$36$ exactly; no rating sheet was ever produced for them, so their semantic",
        r"accuracy is unreported rather than carried over from an earlier draft. Two columns present",
        r"in the previous version of this table are removed because nothing on disk supports them:",
        r"the B1$+$ ratio has no implementation or artifact anywhere in the repository, and the BLEU",
        r"figures were inconsistent with \texttt{pilot\_baseline/semantic\_bleu\_en.csv}, whose",
        r"sacreBLEU corpus scores correspond to $0.011$--$0.056$, not $0.57$/$0.61$.}",
        r"\end{tabular}",
        r"\caption{Semantic analysis: terminology emitted per system and, where a rating sheet",
        r"exists, context-consistency and explanation-accuracy rates.}",
        r"% AUTOGEN-END",
        r"\label{tab:sem}",
        r"\end{table*}",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")
    for model, lang, terms, rc, dc, src in rows:
        print(f"  {model:<22}{lang:<4}{terms:>5}  R={rc:<18} D={dc:<18} {src}")


if __name__ == "__main__":
    main()
