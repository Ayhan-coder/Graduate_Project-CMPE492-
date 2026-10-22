import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

manifest_file = ROOT / "data" / "manifest.jsonl"
rows = [json.loads(line) for line in manifest_file.read_text(encoding="utf-8").splitlines() if line.strip()]

total_words = 0
total_dur = 0.0

# Load terms per video
terms_csv = ROOT / "annotation" / "ablation_terms_qwen3.5_4b.csv"
import csv
clip_terms = {}
if terms_csv.exists():
    with terms_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", "")
            term = row.get("term", "").strip()
            if vid and term:
                clip_terms.setdefault(vid, set()).add(term)

print(f"{'ID':<6} {'Lang':<5} {'Duration':<10} {'Words':<7} {'WPM':<7} {'Unique Terms':<14} {'Density (T/1k)':<15} {'Gold Source':<25}")
print("-" * 95)

for r in rows:
    dur_s = float(r.get("duration_s", 0))
    ref = r.get("reference", "")
    words = len(ref.split())
    total_words += words
    total_dur += dur_s
    wpm = (words / (dur_s / 60.0)) if dur_s > 0 else 0
    m, s = int(dur_s // 60), int(dur_s % 60)
    time_str = f"{m:02d}:{s:02d}"
    gold = r.get("gold_how", "unknown")
    
    n_terms = len(clip_terms.get(r['id'], []))
    density = (n_terms / words * 1000.0) if words > 0 else 0.0
    print(f"{r['id']:<6} {r['lang']:<5} {time_str:<10} {words:<7} {wpm:<7.1f} {n_terms:<14} {density:<15.2f} {gold:<25}")

print("-" * 95)
tot_m, tot_s = int(total_dur // 60), int(total_dur % 60)
all_terms_count = sum(len(v) for v in clip_terms.values())
overall_density = (all_terms_count / total_words * 1000.0) if total_words > 0 else 0.0
print(f"TOTAL: {len(rows)} clips, {tot_m:02d}:{tot_s:02d} ({total_dur:.1f}s), {total_words} words, {all_terms_count} terms, avg density: {overall_density:.2f} terms/1k words")

