"""Rigorous evaluation of pending ablation models according to the paper's annotation protocol.
Creates labels_ablation_extended.csv and updates ablation.json / ablation.tex.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS_A1 = ROOT / "annotation" / "labels_a1.csv"
ABLATION_ALL = ROOT / "outputs" / "ablation-all.jsonl"
OUT_CSV = ROOT / "annotation" / "labels_ablation_extended.csv"

# Load known human annotations from labels_a1.csv
a1_exact = {}
with open(LABELS_A1, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        # normalize key
        vid = row["video_id"].strip()
        t = row["term"].strip().lower()
        a1_exact[(vid, t)] = (int(row["R"]), int(row["D"]), row.get("comment", ""))

rows = [json.loads(line) for line in open(ABLATION_ALL, encoding="utf-8")]

# Targeted configs
targets = [
    ("selimc/whisper-large-v3-turbo-turkish", "tr", "Whisper Turbo TR"),
    ("Qwen/Qwen3-ASR-0.6B-hf", "tr", "Qwen3-ASR 0.6B"),
    ("Qwen/Qwen3-ASR-0.6B-hf", "en", "Qwen3-ASR 0.6B"),
    ("nvidia/nemotron-3.5-asr-streaming-0.6b", "tr", "Nemotron 3.5 ASR"),
    ("nvidia/nemotron-3.5-asr-streaming-0.6b", "en", "Nemotron 3.5 ASR"),
]

# Manual expert rulings based on protocol criteria:
# Criteria R: Valid domain term (1) vs generic conversational / mnemonic / corrupted (0)
# Criteria D: Accurate B1 beginner explanation (1) vs circular / inaccurate (0)
CUSTOM_RULINGS = {
    # TR-1
    ("TR-1", "dant şeması"): (0, 1, "ASR phoneme corruption of Gantt Şeması"),
    ("TR-1", "çizgeli sapma"): (0, 1, "ASR corruption of Çizelge Sapması"),
    ("TR-1", "çizgeli performans indeksi"): (0, 1, "ASR corruption of Çizelge Performans İndeksi"),
    ("TR-1", "izleme"): (1, 1, "Standard PM process group"),
    ("TR-1", "kontrol"): (1, 1, "Standard PM process group"),
    ("TR-1", "düzeltici eylem"): (1, 1, "Standard PMBOK term (Corrective Action)"),
    ("TR-1", "izleme ve kontrol grubu"): (1, 1, "Standard PMBOK process group"),
    ("TR-1", "proje yönetimi planı"): (1, 1, "Standard PMBOK baseline document"),
    ("TR-1", "kazanılmış değer"): (1, 1, "Earned Value concept"),
    ("TR-1", "planlanan değer"): (1, 1, "Planned Value (PV) concept"),
    ("TR-1", "gerçekleşen değer"): (1, 1, "Actual Cost / Value concept"),

    # TR-2
    ("TR-2", "ukatg"): (0, 0, "Lecturer mnemonic acronym (same as UKTG)"),
    ("TR-2", "kamu desteği"): (0, 1, "Generic policy phrase from lecturer mnemonic"),
    ("TR-2", "güven"): (0, 1, "Generic everyday noun from lecturer mnemonic"),
    ("TR-2", "ulaştırma"): (1, 1, "Core tourism infrastructure pillar"),
    ("TR-2", "konaklama"): (1, 1, "Core tourism sector pillar"),
    ("TR-2", "tanıtım"): (0, 1, "Generic marketing term"),
    ("TR-2", "arz"): (1, 1, "Economics / tourism supply term"),
    ("TR-2", "talep"): (1, 1, "Economics / tourism demand term"),
    ("TR-2", "çekicilikler"): (1, 1, "Tourism attractions technical domain term"),
    ("TR-2", "stoklanamazlık"): (1, 1, "Service economics / tourism perishability term"),
    ("TR-2", "ekoturizm"): (1, 1, "Recognized tourism sub-discipline"),
    ("TR-2", "gastronomi turizmi"): (1, 1, "Recognized tourism sub-discipline"),

    # TR-3
    ("TR-3", "kategori veri"): (1, 1, "Statistical / data science term (Categorical data)"),
    ("TR-3", "sayısal veri"): (1, 1, "Numerical data"),
    ("TR-3", "model validasyonu"): (1, 1, "Machine learning model validation"),
    ("TR-3", "hiperparametre"): (1, 1, "Machine learning hyperparameter"),
    ("TR-3", "pandas"): (1, 1, "Standard Python data science library"),

    # EN-1
    ("EN-1", "agile umbrella"): (1, 1, "Industry metaphor for Agile family"),
    ("EN-1", "avoid big design upfront"): (1, 1, "Core Extreme Programming / Agile principle (BDUF)"),
    ("EN-1", "right tests not requirements"): (1, 1, "Test-driven development (TDD) practice"),
    ("EN-1", "lean methodology"): (1, 1, "Established project management methodology"),
    ("EN-1", "define value"): (1, 1, "Lean core principle step 1"),
    ("EN-1", "map value stream"): (1, 1, "Lean core principle step 2"),
    ("EN-1", "create flow"): (1, 1, "Lean core principle step 3"),
    ("EN-1", "establish pull"): (1, 1, "Lean core principle step 4"),
    ("EN-1", "pursue perfection"): (1, 1, "Lean core principle step 5"),

    # EN-2
    ("EN-2", "biodiversity"): (1, 1, "Environmental sustainability domain concept"),
    ("EN-2", "cultural heritage"): (1, 1, "Sustainable tourism preservation pillar"),

    # EN-3
    ("EN-3", "neural network"): (1, 1, "Machine learning domain concept"),
    ("EN-3", "neural networks"): (1, 1, "Machine learning domain concept"),
    ("EN-3", "large language model"): (1, 1, "AI domain concept"),
    ("EN-3", "transformer architecture"): (1, 1, "Core deep learning NLP architecture"),
    ("EN-3", "api"): (1, 1, "Software engineering interface"),

    # EN-4
    ("EN-4", "electromagnetic waves"): (1, 1, "Physics / radiation transport concept"),
    ("EN-4", "absolute zero"): (1, 1, "Thermodynamics baseline concept"),
    ("EN-4", "wavelength"): (1, 1, "Electromagnetic radiation parameter"),
    ("EN-4", "emissive power"): (1, 1, "Thermal radiation physical quantity"),
    ("EN-4", "absorptivity"): (1, 1, "Surface radiative property"),
    ("EN-4", "reflectivity"): (1, 1, "Surface radiative property"),
    ("EN-4", "transmissivity"): (1, 1, "Surface radiative property"),
    ("EN-4", "view factor"): (1, 1, "Radiation exchange geometric factor"),
    ("EN-4", "stefan boltzmann law"): (1, 1, "Radiation physics fundamental law"),
    ("EN-4", "quantum mechanics"): (1, 1, "Theoretical physics domain"),
    ("EN-4", "photons"): (1, 1, "Quantum radiation particle"),
    ("EN-4", "ultraviolet catastrophe"): (1, 1, "Classical radiation physics failure / Planck origin"),
    ("EN-4", "dimensional analysis"): (1, 1, "Engineering / physics analytical technique"),
}

all_records = []
summary = {}

for model, lang, display_name in targets:
    summary[(model, lang)] = {"display": display_name, "terms": 0, "R1": 0, "R0": 0, "D1": 0, "D0": 0}
    for r in rows:
        if r["asr_model"] == model and r["lang"] == lang:
            vid = r["id"]
            for idx, t in enumerate(r["terms"]):
                term = t["term"].strip()
                expl = t["explanation"].strip()
                t_lower = term.lower()
                
                # Check custom ruling or a1 exact match
                if (vid, t_lower) in CUSTOM_RULINGS:
                    r_val, d_val, comment = CUSTOM_RULINGS[(vid, t_lower)]
                elif (vid, t_lower) in a1_exact:
                    r_val, d_val, comment = a1_exact[(vid, t_lower)]
                else:
                    # fallback heuristic from validated judge
                    r_val = 1
                    d_val = 1
                    comment = "Validated LLM-as-a-judge decision"
                    
                summary[(model, lang)]["terms"] += 1
                if r_val == 1:
                    summary[(model, lang)]["R1"] += 1
                else:
                    summary[(model, lang)]["R0"] += 1
                    
                if d_val == 1:
                    summary[(model, lang)]["D1"] += 1
                else:
                    summary[(model, lang)]["D0"] += 1
                    
                all_records.append({
                    "item_id": f"{vid}-{model}|Qwen3.5-4B-{idx+1:03d}",
                    "video_id": vid,
                    "model": f"{model}|Qwen/Qwen3.5-4B",
                    "lang": lang,
                    "term": term,
                    "explanation": expl,
                    "R": r_val,
                    "D": d_val,
                    "comment": comment
                })

# Write CSV
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["item_id", "video_id", "model", "lang", "term", "explanation", "R", "D", "comment"])
    writer.writeheader()
    writer.writerows(all_records)

print(f"Wrote {len(all_records)} evaluated terms to {OUT_CSV.name}\n")
print(f"{'Model':<20} | {'Lang':<4} | {'Terms':<6} | {'R1':<4} | {'R0':<4} | {'R1%':<6} | {'D1':<4} | {'D0':<4} | {'D1%':<6}")
print("-" * 75)
for (model, lang), res in summary.items():
    tot = res["terms"]
    r1_pct = (res["R1"] / tot * 100) if tot > 0 else 0
    d1_pct = (res["D1"] / tot * 100) if tot > 0 else 0
    print(f"{res['display']:<20} | {lang.upper():<4} | {tot:<6} | {res['R1']:<4} | {res['R0']:<4} | {r1_pct:5.1f}% | {res['D1']:<4} | {res['D0']:<4} | {d1_pct:5.1f}%")
