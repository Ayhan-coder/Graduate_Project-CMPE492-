"""Agreement + triage statistics for the machine-only judge panel.

Reads two or more judge arms produced by llm_judge_protocol.py (and optionally
the earlier A40 7B arm, which used a different rubric) and answers three
questions, none of which need a human:

1. model-model agreement  — how much do differently-origin judges agree?
   Reported as Po / Pe / Cohen kappa / PABAK. This is NOT inter-annotator
   agreement: no human is involved, and agreement between models says nothing
   about correctness.
2. human-LLM agreement    — each arm scored against the real human labels
   (labels_a1.csv, labels_a2.csv) carried on labels_pilot_ctx.csv. The
   human-human value recomputed on the same rows is the reference line an arm
   has to approach to be usable as a screen.
3. triage                 — on the machine_only candidates, split items into
   accept / reject / disagreement (prefix "consensus_" when two or more arms
   scored the item, "solo_" when only one did) and emit a stratified human
   queue (disagreement first, then arm-flagged zeros) so human labour is spent
   where the models disagree or object.

Usage
-----
python experiments/panel_agreement.py \
    --arms experiments/annotation/labels_panel_gemma.csv \
           experiments/annotation/labels_panel_smollm.csv \
           experiments/annotation/labels_7b_judge.csv
Output: experiments/outputs/panel_agreement.json
        experiments/annotation/labels_todo_human_queue.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "experiments" / "annotation"
OUT_JSON = ROOT / "experiments" / "outputs" / "panel_agreement.json"
QUEUE_CSV = ANN / "labels_todo_human_queue.csv"

QUEUE_FIELDS = [
    "item_id", "video_id", "model", "lang", "term", "explanation", "context",
    "R", "D", "comment", "gold_how", "queue_reason",
]


def agreement(pairs: list[tuple[dict, dict]], key: str) -> dict:
    """Cohen's kappa on binary label pairs (as dicts), plus PABAK."""
    n = len(pairs)
    if not n:
        return {"n": 0}
    ab = [(a[key], b[key]) for a, b in pairs]
    po = sum(1 for x, y in ab if x == y) / n
    p1a = sum(x for x, _ in ab) / n
    p1b = sum(y for _, y in ab) / n
    pe = p1a * p1b + (1 - p1a) * (1 - p1b)
    kappa = 0.0 if pe == 1 else (po - pe) / (1 - pe)
    return {
        "n": n,
        "agreement_pct": round(100 * po, 2),
        "po": round(po, 4),
        "pe": round(pe, 4),
        "kappa": round(kappa, 4),
        "pabak": round(2 * po - 1, 4),
        "confusion_matrix": {
            "tp": sum(1 for x, y in ab if x == 1 and y == 1),
            "tn": sum(1 for x, y in ab if x == 0 and y == 0),
            "fp": sum(1 for x, y in ab if x == 0 and y == 1),
            "fn": sum(1 for x, y in ab if x == 1 and y == 0),
        },
    }


def arm_tag(path: Path) -> str:
    return "panel_" + Path(path).stem.removeprefix("labels_").removeprefix("panel_")


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_arm(path: Path) -> dict[str, dict]:
    """Normalise an arm CSV to item_id -> {R, D, why, model, rubric}."""
    rows = read_rows(path)
    if not rows:
        return {}
    cols = set(rows[0])
    if "judge_7b_R" in cols:
        rk, dk, wk = "judge_7b_R", "judge_7b_D", "judge_reasoning"
        rubric = "a40-zeroshot-b1 (farkli rubrik; human-LLM kiyasinda ayrica not edilmez)"
        model = "Qwen2.5-7B-Instruct (A40)"
    elif "judge_R" in cols:
        rk, dk, wk = "judge_R", "judge_D", "judge_why"
        rubric = rows[0].get("prompt_version", "protocol-v1")
        model = rows[0].get("judge_model", path.stem)
    else:
        raise SystemExit(f"{path}: judge_R / judge_7b_R sütunu yok")
    out = {}
    for r in rows:
        try:
            r_v, d_v = int(r[rk]), int(r[dk])
        except (TypeError, ValueError):
            continue
        out[r["item_id"]] = {"R": r_v, "D": d_v, "why": (r.get(wk) or "")[:300],
                             "model": model, "rubric": rubric}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=Path, nargs="+", required=True)
    ap.add_argument("--candidates", type=Path, default=ANN / "labels_todo_32clip_machine.csv")
    ap.add_argument("--pilot", type=Path, default=ANN / "labels_pilot_ctx.csv")
    ap.add_argument("--human-candidates", type=Path, default=None,
                    help="real single-annotator labels over the candidate sheet (export format: "
                         "item_id,R,D,comment). There is no second human arm on these rows, so the "
                         "result is human-LLM agreement and is never an IAA value.")
    ap.add_argument("--queue", type=int, default=80)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    arms = {p.name: load_arm(p) for p in args.arms if p.exists()}
    skipped = [p.name for p in args.arms if not p.exists()]
    if not arms:
        raise SystemExit("hiç okunabilen hakem kolu yok")

    pilot = {r["item_id"]: r for r in read_rows(args.pilot)}
    cands = read_rows(args.candidates)
    cand_ids = [c["item_id"] for c in cands]
    cand_set = set(cand_ids)

    human_cand = {}
    if args.human_candidates and args.human_candidates.exists():
        for r in read_rows(args.human_candidates):
            try:
                human_cand[r["item_id"]] = {"R": int(r["R"]), "D": int(r["D"])}
            except (KeyError, TypeError, ValueError):
                continue

    # ---- per-arm coverage + verdict distribution -----------------------------
    per_arm = {}
    for name, arm in arms.items():
        scored = list(arm.values())
        per_arm[name] = {
            "model": scored[0]["model"],
            "rubric": scored[0]["rubric"],
            "judged_items": len(arm),
            "pilot_overlap": sum(1 for i in arm if i in pilot),
            "candidate_overlap": sum(1 for i in arm if i in cand_set),
            "R1_pct": round(100 * sum(v["R"] for v in scored) / len(scored), 2),
            "D1_pct": round(100 * sum(v["D"] for v in scored) / len(scored), 2),
            "both1_pct": round(100 * sum(1 for v in scored if v["R"] and v["D"]) / len(scored), 2),
        }

    # ---- model-model agreement ------------------------------------------------
    # union of judged ids: arms may cover different slices (e.g. an A40 arm that only
    # scored pilot items vs a local arm that also scored the candidates)
    all_ids = sorted({i for arm in arms.values() for i in arm})
    model_model = {}
    for a, b in combinations(sorted(arms), 2):
        common = [i for i in all_ids if i in arms[a] and i in arms[b]]
        if not common:
            continue
        pairs = [(arms[a][i], arms[b][i]) for i in common]
        cand_common = [i for i in common if i in cand_set]
        entry = {
            "scope": "pilot+aday",
            "n_pilot": sum(1 for i in common if i in pilot),
            "n_candidate": len(cand_common),
            "R": agreement(pairs, "R"),
            "D": agreement(pairs, "D"),
            "full_agreement_pct": round(
                100 * sum(1 for x, y in pairs if x["R"] == y["R"] and x["D"] == y["D"]) / len(pairs), 2
            ),
            "rubrics": [arms[a][common[0]]["rubric"], arms[b][common[0]]["rubric"]],
        }
        if cand_common:
            cpairs = [(arms[a][i], arms[b][i]) for i in cand_common]
            entry["candidates_only"] = {"R": agreement(cpairs, "R"), "D": agreement(cpairs, "D")}
        model_model[f"{a} vs {b}"] = entry

    # ---- human-LLM agreement on the pilot rows -------------------------------
    human_llm = {}
    for name, arm in arms.items():
        for human in ("human_a1", "human_a2"):
            pairs = []
            for i in arm:
                p = pilot.get(i)
                if not p or not str(p.get(f"{human}_R", "")).strip() or not str(p.get(f"{human}_D", "")).strip():
                    continue
                pairs.append((arm[i], {"R": int(p[f"{human}_R"]), "D": int(p[f"{human}_D"])}))
            if pairs:
                human_llm[f"{name} vs {human}"] = {"R": agreement(pairs, "R"),
                                                  "D": agreement(pairs, "D")}

    # same comparison restricted to the candidate sheet, where the only human arm is a1
    if human_cand:
        for name, arm in arms.items():
            pairs = [(arm[i], human_cand[i]) for i in arm if i in human_cand]
            if pairs:
                entry = {"R": agreement(pairs, "R"), "D": agreement(pairs, "D"),
                         "n": len(pairs),
                         "note": "tek insan kolu (a1); bu deger insan-LLM uyumudur, IAA degildir"}
                only = [i for i in arm if i in cand_set and i in human_cand]
                if only:
                    opairs = [(arm[i], human_cand[i]) for i in only]
                    entry["candidates_only"] = {"R": agreement(opairs, "R"),
                                                "D": agreement(opairs, "D"), "n": len(opairs)}
                human_llm[f"{name} vs a1_candidates"] = entry

    # ---- human-human reference line on the same rows -------------------------
    hh_pairs = [
        ({"R": int(pilot[i]["human_a1_R"]), "D": int(pilot[i]["human_a1_D"])},
         {"R": int(pilot[i]["human_a2_R"]), "D": int(pilot[i]["human_a2_D"])})
        for i in pilot
        if str(pilot[i].get("human_a1_R", "")).strip() and str(pilot[i].get("human_a2_R", "")).strip()
    ]
    human_human = {"labels_a1 vs labels_a2": {"R": agreement(hh_pairs, "R"),
                                              "D": agreement(hh_pairs, "D")}} if hh_pairs else {}

    # ---- triage on the candidates --------------------------------------------
    # "consensus" is only used when at least two arms scored the item; with a single
    # covering arm the honest word is "solo".
    def verdict(item_id: str) -> str:
        vs = [arms[a][item_id] for a in arms if item_id in arms[a]]
        if not vs:
            return "unjudged"
        pref = "consensus" if len(vs) >= 2 else "solo"
        if all(v["R"] == 1 and v["D"] == 1 for v in vs):
            return f"{pref}_accept"
        if all(v["R"] == 0 or v["D"] == 0 for v in vs):
            return f"{pref}_reject"
        return "disagreement"

    triage = {c["item_id"]: verdict(c["item_id"]) for c in cands}
    triage_counts = Counter(triage.values())
    by_lang = defaultdict(Counter)
    for c in cands:
        by_lang[c["lang"]][triage[c["item_id"]]] += 1

    # ---- stratified human queue ---------------------------------------------
    rng = random.Random(args.seed)
    # items a1 already labeled by hand must not be queued again
    unlabeled = [c for c in cands if c["item_id"] not in human_cand]
    disagree = [c for c in unlabeled if triage[c["item_id"]] == "disagreement"]
    flagged = [c for c in unlabeled
               if triage[c["item_id"]] != "disagreement"
               and any(v["R"] == 0 or v["D"] == 0 for a in arms for v in [arms[a].get(c["item_id"])] if v)]
    rest = [c for c in unlabeled
            if triage[c["item_id"]] != "disagreement" and c not in flagged]
    queue: list[tuple[dict, str]] = [(c, "panel_disagreement") for c in disagree]
    queue += [(c, "panel_flagged_zero") for c in flagged]
    if len(queue) < args.queue and rest:
        byclip: dict[str, list[dict]] = defaultdict(list)
        for c in rest:
            byclip[c["video_id"]].append(c)
        for lst in byclip.values():
            rng.shuffle(lst)
        order = sorted(byclip, key=lambda k: -len(byclip[k]))
        while len(queue) < args.queue and any(byclip.values()):
            for k in order:
                if not byclip[k]:
                    continue
                c = byclip[k].pop()
                queue.append((c, f"stratified_{triage[c['item_id']]}"))
                if len(queue) >= args.queue:
                    break
    queue = queue[: args.queue]

    extra_cols = []
    for a in arms:
        stem = arm_tag(a)
        extra_cols += [f"{stem}_R", f"{stem}_D"]

    with QUEUE_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, QUEUE_FIELDS + extra_cols)
        w.writeheader()
        for c, reason in queue:
            row = {k: c.get(k, "") for k in QUEUE_FIELDS if k in c}
            row.update({"R": "", "D": "", "gold_how": "machine_only", "queue_reason": reason,
                        "comment": "panel sirasi; R/D insan etiketi bekliyor (machine_only altin degil)"})
            for a in arms:
                v = arms[a].get(c["item_id"])
                stem = arm_tag(a)
                row[f"{stem}_R"] = v["R"] if v else ""
                row[f"{stem}_D"] = v["D"] if v else ""
            w.writerow(row)

    report = {
        "framing": {
            "model_model": "model-model uyumu; IAA DEGIL (isina karismadi)",
            "human_llm": "insan-LLM uyumu; a1/a2 gerçek insan etiketleri",
            "human_human": "ayni satırlarda yeniden hesaplanan insan-insan çizgisi",
            "gold_status": "panel etiketleri gold_how=machine_only olarak kalır",
        },
        "arms_read": sorted(arms),
        "arms_missing": skipped,
        "per_arm": per_arm,
        "model_model_agreement": model_model,
        "human_llm_agreement": human_llm,
        "human_human_reference": human_human,
        "human_candidate_labels": {
            "source": str(args.human_candidates.resolve().relative_to(ROOT)) if args.human_candidates else None,
            "labeled": len(human_cand),
            "of_candidates": len(cands),
            "R0": sum(1 for v in human_cand.values() if v["R"] == 0),
            "D0": sum(1 for v in human_cand.values() if v["D"] == 0),
            "both_arms": False,
            "note": "tek etiketleyici (a1); ikinci insan kolu yok -> IAA hesaplanamaz",
        },
        "candidate_triage": {
            "total": len(cands),
            "counts": dict(triage_counts),
            "by_lang": {k: dict(v) for k, v in sorted(by_lang.items())},
            "reject_pct": round(
                100 * sum(v for k, v in triage_counts.items() if k.endswith("_reject")) / len(cands), 2
            ) if cands else None,
        },
        "human_queue": {
            "path": str(QUEUE_CSV.relative_to(ROOT)),
            "size": len(queue),
            "reasons": dict(Counter(r for _, r in queue)),
            "seed": args.seed,
            "panel_columns": extra_cols,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
