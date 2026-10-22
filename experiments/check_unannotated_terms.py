import csv
import json
import re
import sys
from pathlib import Path

# Configure UTF-8 for console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent

def norm_term(t: str, lang: str = "tr") -> str:
    cleaned = (t or "").strip()
    if lang.startswith("tr"):
        cleaned = cleaned.replace("I", "ı").replace("İ", "i")
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^\w\sçğıöşü]", "", cleaned)
    return " ".join(cleaned.split())

# Load known human annotations from labels_a1.csv
a1_map = {}
with open(ROOT / "annotation" / "labels_a1.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        lang = "tr" if row["video_id"].startswith("TR") else "en"
        k = (row["video_id"], norm_term(row["term"], lang))
        a1_map[k] = (int(row["R"]), int(row["D"]), row.get("comment", ""))

with open(ROOT / "outputs" / "ablation-all.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

target_configs = [
    ("selimc/whisper-large-v3-turbo-turkish", "tr"),
    ("Qwen/Qwen3-ASR-0.6B-hf", "tr"),
    ("Qwen/Qwen3-ASR-0.6B-hf", "en"),
    ("nvidia/nemotron-3.5-asr-streaming-0.6b", "tr"),
    ("nvidia/nemotron-3.5-asr-streaming-0.6b", "en"),
]

for model, lang in target_configs:
    print(f"\n==================== {model} ({lang.upper()}) ====================")
    matched = 0
    total = 0
    items = []
    for r in rows:
        if r["asr_model"] == model and r["lang"] == lang:
            vid = r["id"]
            for t in r["terms"]:
                total += 1
                term = t["term"].strip()
                expl = t["explanation"].strip()
                n_t = norm_term(term, lang)
                key = (vid, n_t)
                
                # Direct match
                if key in a1_map:
                    matched += 1
                    r_val, d_val, c = a1_map[key]
                    items.append((vid, term, expl, r_val, d_val, f"Human match: R={r_val}, D={d_val} ({c})"))
                else:
                    # Fuzzy / plural / compound match
                    found = None
                    for (v, cand_t), (r_v, d_v, c) in a1_map.items():
                        if v == vid and (cand_t == n_t.rstrip("s") or cand_t.rstrip("s") == n_t or n_t in cand_t or cand_t in n_t):
                            found = (r_v, d_v, f"Human close match ({cand_t}): R={r_v}, D={d_v}")
                            break
                    if found:
                        matched += 1
                        items.append((vid, term, expl, found[0], found[1], found[2]))
                    else:
                        items.append((vid, term, expl, None, None, "Needs judge"))
                        
    print(f"Total terms: {total}, Human matches from labels_a1: {matched}/{total}")
    for vid, term, expl, r_val, d_val, note in items:
        status = f"[R={r_val}, D={d_val}]" if r_val is not None else "[PENDING]"
        print(f"  {vid} | {term:30s} {status} | {note}")
