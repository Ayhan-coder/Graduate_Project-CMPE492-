"""Protocol-aligned LLM judge arm (R/D as defined in annotation/PROTOCOL.md).

This is a *machine arm*: it can never stand in for a human annotator. The report
must say "human–LLM agreement", never "inter-annotator agreement". Default judge
is Qwen3-8B, i.e. a different model from the Qwen3.5-4B extractor, so the judge is
not grading its own output.

Rubric (verbatim from PROTOCOL.md, not the B1 rubric used by llm_judge.py):
  R1  term is consistent with the transcript context
  R0  wrong language, not in the lecture, or not a domain term
  D1  explanation is correct relative to TDK (Turkish) / OED (English)
  D0  it is not

Usage
-----
python experiments/llm_judge_protocol.py --sheet annotation/labels_todo_15clip.csv --limit 5
python experiments/llm_judge_protocol.py --sheet annotation/labels_todo_15clip.csv
python experiments/llm_judge_protocol.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --sheet annotation/labels_todo_32clip_machine.csv --out annotation/labels_panel_smollm.csv
Output: annotation/labels_llmjudge.csv (resumable; rerun to continue)

Cross-origin panels: pass a judge model from a different family than the
Qwen3.5-4B extractor and a separate --out per arm. Panel output is machine
labelling; it never replaces a human arm and is never reported as IAA.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

ANN = ROOT / "experiments" / "annotation"
OUT = ANN / "labels_llmjudge.csv"
JUDGE_MODEL = "Qwen/Qwen3-8B"
PROMPT_VERSION = "protocol-v1"

PROMPT_TR = """Bağımsız bir hakemsin. Aşağıdaki terim ve açıklamayı, ders transkriptinden alınan bağlama göre iki ikili ölçütle değerlendir.

Transkript bağlamı: "{CONTEXT}"
Terim: "{TERM}"
Açıklama: "{EXPLANATION}"

R (Bağlam tutarlılığı):
- 1: Terim derste geçen, alana özgü gerçek bir terimdir ve bağlamla çelişmez.
- 0: Terim derste geçmiyor, yanlış dilde, günlük bir kelime ya da alan dışı.

D (Tanım doğruluğu, TDK ölçütü):
- 1: Açıklama terimin TDK sözlüğündeki ya da alanın standart tanımındaki anlamıyla uyuşur.
- 0: Açıklama yanlış, uydurma, döngüsel ya da terimin anlamıyla çelişiyor.

Sadece şu JSON'u döndür: {{"R": 0 veya 1, "D": 0 veya 1, "why": "kısa gerekçe"}}"""

PROMPT_EN = """You are an independent judge. Evaluate the term and explanation against the transcript context using two binary criteria.

Transcript context: "{CONTEXT}"
Term: "{TERM}"
Explanation: "{EXPLANATION}"

R (Context consistency):
- 1: the term occurs in the lecture, belongs to the domain, and does not contradict the context.
- 0: not in the lecture, wrong language, an everyday word, or out of domain.

D (Definition correctness, OED criterion):
- 1: the explanation matches the term's OED / standard domain definition.
- 0: the explanation is wrong, invented, circular, or contradicts the term's meaning.

Return only this JSON: {{"R": 0 or 1, "D": 0 or 1, "why": "short reason"}}"""

# Same rubric, terse key-value wording. The 1.7B/1B local judges echo the long JSON
# prompt back instead of answering (~57% unparseable on protocol-v1), so cross-origin
# small arms use this style and are recorded as prompt_version=protocol-v1-keyvalue.
PROMPT_KV_TR = """Sen bağımsız bir hakemsin. Bir ders transkriptinden alınan terimi ve açıklamayı değerlendir.

Bağlam: "{CONTEXT}"
Terim: "{TERM}"
Açıklama: "{EXPLANATION}"

R: terim derste geçiyor ve alana özgü gerçek bir terim mi? (yanlış dil, günlük kelime ya da alan dışı ise 0)
D: açıklama terimin alan/TDK tanımısıyla uyuşuyor mu? (yanlış, uydurma veya döngüsel ise 0)

Yalnızca şu üç satırı yaz:
R=<0 veya 1>
D=<0 veya 1>
NEDEN=<en çok 12 kelime>"""

PROMPT_KV_EN = """You are an independent judge. Rate a term and its explanation taken from a lecture transcript.

Context: "{CONTEXT}"
Term: "{TERM}"
Explanation: "{EXPLANATION}"

R: does the term occur in the lecture and name a real domain concept? (everyday word, wrong language or out of domain -> 0)
D: does the explanation match the term's standard domain definition? (wrong, invented or circular -> 0)

Write only these three lines:
R=<0 or 1>
D=<0 or 1>
REASON=<at most 12 words>"""


def parse(raw: str) -> dict | None:
    m = re.search(r"\{[^{}]*\}", raw, flags=re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            d = None
        if isinstance(d, dict):
            try:
                r, dd = int(d["R"]), int(d["D"])
            except (KeyError, TypeError, ValueError):
                r = dd = None
            if r in (0, 1) and dd in (0, 1):
                return {"R": r, "D": dd, "why": str(d.get("why", ""))[:300]}
    # key-value style: R=1 / D: 0, also tolerating both verdicts on one line
    kv: dict[str, int] = {}
    for k, v in re.findall(r"\b([RD])\s*[:=]\s*([01])\b", raw):
        kv.setdefault(k.lower(), int(v))
    if "r" in kv and "d" in kv:
        why = re.search(r"(?im)(?:NEDEN|REASON)\s*[:=]\s*(.+)", raw)
        return {"R": kv["r"], "D": kv["d"], "why": (why.group(1).strip() if why else "")[:300]}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=Path, default=ANN / "labels_todo_15clip.csv")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--model", default=JUDGE_MODEL,
                    help="judge model id; use a family other than the Qwen3.5-4B extractor")
    ap.add_argument("--prompt-style", choices=["json", "keyvalue"], default="json",
                    help="keyvalue for small judges that echo the JSON rubric back")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    version = PROMPT_VERSION if args.prompt_style == "json" else f"{PROMPT_VERSION}-keyvalue"

    from run_llm import generate_hf

    rows = []
    with args.sheet.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if args.video and r["video_id"] != args.video:
                continue
            rows.append(r)

    done = {}
    if args.out.exists():
        with args.out.open(encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("judge_R", "") != "":
                    done[r["item_id"]] = r

    todo = [r for r in rows if r["item_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows)} madde, {len(done)} zaten hakemlendi, {len(todo)} kaldı", flush=True)

    fields = ["item_id", "video_id", "lang", "term", "explanation", "context",
              "judge_R", "judge_D", "judge_why", "judge_model", "prompt_version"]
    t0 = time.time()
    for n, r in enumerate(todo, 1):
        ctx = r.get("context", "") or ""
        if args.prompt_style == "keyvalue":
            tmpl = PROMPT_KV_TR if r["lang"] == "tr" else PROMPT_KV_EN
        else:
            tmpl = PROMPT_TR if r["lang"] == "tr" else PROMPT_EN
        prompt = tmpl.format(CONTEXT=ctx[:600], TERM=r["term"], EXPLANATION=r["explanation"])
        verdict = None
        for attempt in range(2):
            if args.prompt_style == "keyvalue":
                cont = generate_hf(args.model, prompt, max_new_tokens=80, prefill="R=")
                raw = "R=" + cont
            else:
                raw = generate_hf(args.model, prompt, max_new_tokens=320)
            verdict = parse(raw)
            if verdict:
                break
        if verdict is None:
            print(f"[{n}] {r['item_id']}: ayrıştırılamadı, atlandı", flush=True)
            continue
        done[r["item_id"]] = {
            "item_id": r["item_id"],
            "video_id": r["video_id"],
            "lang": r["lang"],
            "term": r["term"],
            "explanation": r["explanation"],
            "context": ctx,
            "judge_R": verdict["R"],
            "judge_D": verdict["D"],
            "judge_why": verdict["why"],
            "judge_model": args.model,
            "prompt_version": version,
        }
        elapsed = time.time() - t0
        print(f"[{n}/{len(todo)}] {r['item_id']} R={verdict['R']} D={verdict['D']} ({elapsed:.0f}s)", flush=True)
        if n % 10 == 0:
            write(args.out, fields, done, rows)

    write(args.out, fields, done, rows)
    print(f"yazıldı: {args.out} ({len(done)} madde)")


def write(path: Path, fields: list[str], done: dict, order: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in order:
            if r["item_id"] in done:
                w.writerow(done[r["item_id"]])


if __name__ == "__main__":
    main()
