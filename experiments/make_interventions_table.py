"""Render paper/tables/asr_interventions.tex from outputs/interventions_summary.json.

Every cell is recomputed from the JSONL artifacts by summarize_interventions.py; the
table is generated so that no intervention number can drift from its evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIASTED = ("TR-1", "EN-1")  # only these two clips own a closed gold list


def pct(x: float) -> str:
    return f"{x * 100:.1f}"


def rel(prev: float, cur: float) -> str:
    if not prev:
        return "---"
    d = (cur - prev) / prev * 100.0
    return f"{d:+.1f}\\%"


def stage_by_name(stages: list[dict], prefix: str) -> dict | None:
    for s in stages:
        if s["stage"].startswith(prefix):
            return s
    return None


def clip_mean(stage: dict, lang: str, key: str, ids: tuple[str, ...] | None = None) -> float | None:
    vals = [row[key] for cid, row in stage["per_clip"].items()
            if row.get(key) is not None and (ids is None or cid in ids)]
    return (sum(vals) / len(vals)) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=ROOT / "outputs" / "interventions_summary.json")
    ap.add_argument("--out", type=Path, default=ROOT.parent / "paper" / "tables" / "asr_interventions.tex")
    args = ap.parse_args()

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    stages = payload["stages"]
    got = {name: stage_by_name(stages, name) for name in
           ("(1) baseline", "(2) + energy VAD", "(3) + VAD + glossary", "(4) + LLM post")}

    rows = []
    prev = {}
    for label, key in (("Raw Baseline (Whisper Large-v3)", "(1) baseline"),
                       ("+ Energy VAD Chunking", "(2) + energy VAD"),
                       ("+ VAD + Glossary Biasing", "(3) + VAD + glossary"),
                       ("+ LLM Post-Correction", "(4) + LLM post")):
        st = got[key]
        if st is None or st.get("missing"):
            rows.append((label, "---", "---", "not measured yet", "---"))
            continue
        tr, en = st["per_lang"].get("tr"), st["per_lang"].get("en")
        tr_wer = tr["wer_norm_macro"] if tr else None
        en_wer = en["wer_norm_macro"] if en else None
        dtr = rel(prev.get("tr"), tr_wer) if tr_wer is not None else "---"
        den = rel(prev.get("en"), en_wer) if en_wer is not None else "---"
        prev = {"tr": tr_wer, "en": en_wer}
        cov = {lang: st["per_lang"].get(lang, {}).get("gold_mention_coverage") for lang in ("tr", "en")}
        cov_txt = " / ".join("---" if cov[l] is None else f"{cov[l] * 100:.1f}\\%" for l in ("tr", "en"))
        rows.append((label, f"{tr_wer:.4f}" if tr_wer is not None else "---",
                     f"{en_wer:.4f}" if en_wer is not None else "---",
                     f"{dtr} / {den}", cov_txt))

    body = "\n".join(
        f"{lab} & {tr} & {en} & {delta} & {cov} \\\\"
        for lab, tr, en, delta, cov in rows)

    base_st = got["(1) baseline"]
    worse, better = [], []
    for key in ("(2) + energy VAD", "(3) + VAD + glossary", "(4) + LLM post"):
        st = got[key]
        if st is None or st.get("missing") or base_st is None:
            continue
        for lang in ("tr", "en"):
            b = base_st["per_lang"].get(lang, {}).get("wer_norm_macro")
            c = st["per_lang"].get(lang, {}).get("wer_norm_macro")
            if b is None or c is None:
                continue
            tag = f"{key.strip('(0123456789) ')} in {lang.upper()}"
            (better if c < b else worse).append(tag)
    if not worse and not better:
        verdict = "Only the baseline stage has been measured so far."
    elif not better:
        verdict = ("No measured stage lowers macro WER: every intervention cell sits above the "
                   "baseline in both languages.")
    else:
        verdict = (f"{len(better)} of {len(better) + len(worse)} measured stage/language cells "
                   f"fall below the baseline ({'; '.join(better)}); the rest raise it.")

    # Latency cost of the biasing prompt, on the two clips that actually receive a glossary.
    vad_st, bias_st = got["(2) + energy VAD"], got["(3) + VAD + glossary"]
    cost = ""
    if vad_st and bias_st:
        pairs = []
        for cid in BIASTED:
            a = vad_st["per_clip"].get(cid, {}).get("latency_s")
            b = bias_st["per_clip"].get(cid, {}).get("latency_s")
            if a and b:
                pairs.append(f"${b / a:.1f}\\times$ on {cid} ({a:.0f}\\,s to {b:.0f}\\,s)")
        if pairs:
            cost = (" Biasing is not free: on the two clips that receive a glossary it multiplies "
                    f"wall clock ({'; '.join(pairs)}). The five clips with no gold list are "
                    "unaffected by the prompt and reproduce the previous stage's hypotheses "
                    "character for character, which doubles as a determinism control.")

    # Repetition: the published claim was that the ladder removes loops. Report what it does.
    rep_txt = ""
    b_tr = base_st["per_lang"].get("tr", {}) if base_st else {}
    p_tr = (got["(4) + LLM post"] or {}).get("per_lang", {}).get("tr", {})
    r0, r3 = b_tr.get("repetition_rate"), p_tr.get("repetition_rate")
    r0_en = (base_st["per_lang"].get("en", {}) if base_st else {}).get("repetition_rate")
    if r0 is not None and r0_en is not None:
        rep_txt = (f"No stage removes a repetition loop: the GPU baseline has almost none to "
                   f"remove ($8$-gram repetition ${r0:.4f}$ TR / ${r0_en:.4f}$ EN)")
        if r3 is not None:
            rep_txt += f", and the correction stage raises it to ${r3:.4f}$ on Turkish"
        rep_txt += "."

    tex = f"""% GENERATED by experiments/make_interventions_table.py from outputs/interventions_summary.json
% Kaynak JSONL: asr-whisper-combined.rescored.jsonl, asr-vad-large-v3.jsonl,
%                asr-vad-bias-large-v3.jsonl, post_correction.jsonl -- elle düzenlemeyin.
\\begin{{table}}[t]
\\caption{{Measured acoustic-intervention ladder on the same seven clips and the same
\\texttt{{openai/whisper-large-v3}} checkpoint. WER is the unweighted per-clip mean of
normalised WER (macro). {verdict} The terminology column is
\\emph{{mention coverage}} over the closed gold list, not extraction $F_1$.}}
\\label{{tab:interventions}}
\\centering
\\renewcommand{{\\arraystretch}}{{1.15}}
\\setlength{{\\tabcolsep}}{{3.0pt}}
\\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\\toprule
\\textbf{{Pipeline Configuration}} & \\textbf{{TR WER}} & \\textbf{{EN WER}} &
\\textbf{{Rel. $\\Delta$ vs previous (TR / EN)}} & \\textbf{{Gold coverage TR / EN}} \\\\
\\midrule
{body}
\\bottomrule
\\multicolumn{{5}}{{p{{0.98\\columnwidth}}}}{{\\footnotesize
Energy VAD = \\texttt{{vad\\_chunker.py}} ($30$\\,ms frames, cut line
$\\max(10^{{-4}}, 2.2\\times$ the $25$th-percentile noise floor), gaps $\\le 200$\\,ms
closed, boundary search on pauses $>0.4$\\,s, $12$--$28$\\,s windows. It is
\\emph{{not}} a learned VAD and is \\emph{{not}} called by the shipped GUI, which slices
fixed $30$\\,s windows. Biasing injects the first $25$ canonical terms of
\\texttt{{data/gold\\_terms.json}} through \\texttt{{processor.get\\_prompt\\_ids}}, so it
only applies to {', '.join(BIASTED)} and its coverage figure is \\emph{{gold-list leakage}};
only its WER is interpretable. The previously published '+Domain Biasing' cell could not
have been produced by this code path: the prompt plus \\texttt{{max\\_new\\_tokens=440}}
overflowed Whisper's $448$ decoder positions and \\texttt{{generate()}} raised on every
clip (archived under \\texttt{{outputs/*.positions-crash-attempt1}}). {rep_txt}{cost}}}
\\end{{tabular}}
\\end{{table}}
"""
    args.out.write_text(tex, encoding="utf-8")
    print(f"Written: {args.out}")

    # Per-clip grid: the aggregate row above hides the whole finding, which is that the
    # ladder moves one clip by an order of magnitude and leaves five untouched.
    order = sorted(payload["clips"], key=lambda c: (c.startswith("EN"), c))
    keys = ("(1) baseline", "(2) + energy VAD", "(3) + VAD + glossary", "(4) + LLM post")
    cells, improved, degraded, vs_prev_up, vs_prev_dn = [], 0, 0, 0, 0
    for cid in order:
        vals = []
        for key in keys:
            st = got[key]
            v = None
            if st and not st.get("missing"):
                v = st["per_clip"].get(cid, {}).get("wer_norm")
            vals.append("---" if v is None else f"{v:.4f}")
        b0, b3 = vals[0], vals[3]
        if b0 != "---" and b3 != "---":
            if float(b3) < float(b0):
                improved += 1
            elif float(b3) > float(b0):
                degraded += 1
            if b3 != "---" and vals[2] != "---":
                if float(b3) < float(vals[2]):
                    vs_prev_dn += 1
                elif float(b3) > float(vals[2]):
                    vs_prev_up += 1
        mark = "$^{\\dagger}$" if cid in BIASTED else ""
        cells.append(f"{cid}{mark} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\")
    perclip = f"""% GENERATED by experiments/make_interventions_table.py -- elle duzenmeyin.
% Normalised WER per clip across the four measured stages. $^{{\\dagger}}$ marks the two
% clips that own a closed gold list and therefore receive the biasing prompt.
\\begin{{table}}[t]
\\caption{{Clip-level view of Table~\\ref{{tab:interventions}}. Normalised WER. The ladder
does not behave uniformly: on the five clips with no gold list the stage-3 hypothesis is
byte-identical to stage-2, so those columns are blank in effect. Post-correction lowers WER
on {vs_prev_dn} of {len(order)} clips relative to its own biased input, yet {improved} of
{len(order)} clips finish stage-4 below the raw baseline ({degraded} finish above it, the
rest equal). The clip the ladder exists to help (TR-1) is the one it damages hardest, and the
damage compounds through the correction stage.}}
\\label{{tab:interventions_perclip}}
\\centering
\\renewcommand{{\\arraystretch}}{{1.1}}
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\\toprule
\\textbf{{Clip}} & \\textbf{{(1) Raw}} & \\textbf{{(2) +VAD}} &
\\textbf{{(3) +biasing}} & \\textbf{{(4) +post-corr}} \\\\
\\midrule
{chr(10).join(c for c in cells)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    pc_out = args.out.parent / "asr_interventions_perclip.tex"
    pc_out.write_text(perclip, encoding="utf-8")
    print(f"Written: {pc_out}")


if __name__ == "__main__":
    main()
