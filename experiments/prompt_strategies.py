"""prompt_strategies.py: Systematic evaluation of prompting strategies for terminology extraction.

Implements the advisor's work package:
"Evaluate prompting strategies, including chain-of-thought and glossary-based RAG."

Compares:
1. Zero-Shot Baseline (paper standard)
2. Chain-of-Thought (CoT: domain grounding, candidate extraction, noise filtering, synthesis)
3. Glossary-based RAG (anchor keywords provided from course syllabus/glossary)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_gold_terms import evaluate_clip
from run_llm import generate_hf, generate_openai, parse_terms

# 1. Zero-Shot Baseline
PROMPT_ZERO_SHOT = (
    "You are a terminology assistant for {LANG} educational content. "
    "Step 1: From the transcript below, extract domain-specific terms "
    "(including multi-word terms). Step 2: For each term, write one short "
    "explanation at CEFR B1 level that a beginner can understand. Use only "
    "information consistent with the transcript context. Return strictly a "
    "JSON array of objects with fields \"term\" and \"explanation\". "
    "Transcript: {TRANSCRIPT}"
)

# 2. Chain-of-Thought (CoT)
PROMPT_COT_TR = (
    "Sen uzman bir eğitim terminoloğusun. Aşağıdaki ders transkriptini adım adım analiz et:\n"
    "Düşünme Adımları (Chain-of-Thought):\n"
    "1. Konu Tespiti: Bu ders hangi ana disipline ait ve ana odak noktası nedir?\n"
    "2. Aday Çıkarma: Metinde geçen teknik, teorik veya prosedürel kavramları (özellikle çok kelimeli terimleri) belirle.\n"
    "3. Gürültü Eleme: Günlük konuşma diline ait genel isimleri (örneğin 'şey', 'durum', 'adım', 'konu') terim sayma ve filtrele.\n"
    "4. B1 Açıklama: Seçilen her geçerli alan terimi için yeni başlayan bir öğrencinin anlayabileceği CEFR B1 düzeyinde sade bir açıklama yaz.\n\n"
    "Sonuç olarak YALNIZCA geçerli bir JSON dizisi döndür:\n"
    '[{{"term": "Terim Adı", "explanation": "B1 açıklaması", "reasoning": "Neden alan terimi olduğu"}}]\n\n'
    "Transkript: {TRANSCRIPT}"
)

PROMPT_COT_EN = (
    "You are an expert educational terminologist. Analyze the following lecture transcript step-by-step:\n"
    "Reasoning Steps (Chain-of-Thought):\n"
    "1. Domain Identification: What primary discipline does this lecture belong to and what is its core objective?\n"
    "2. Candidate Extraction: Identify candidate technical, procedural, and domain terms (especially multi-word terms).\n"
    "3. Noise Filtering: Filter out generic conversational nouns (e.g., 'thing', 'way', 'part', 'step') that are not technical terms.\n"
    "4. B1 Explanation: For each valid domain term, synthesize a clear, beginner-friendly CEFR B1 level explanation.\n\n"
    "Return STRICTLY a JSON array of objects:\n"
    '[{{"term": "Term Name", "explanation": "B1 explanation", "reasoning": "Why this is a technical domain term"}}]\n\n'
    "Transcript: {TRANSCRIPT}"
)

# 3. Glossary-based RAG
PROMPT_RAG_TR = (
    "Sen bir eğitim terminolojisi asistanısın. Aşağıda ders müfredatından ve slaytlarından alınan referans anahtar kavramlar ve ders transkripti verilmiştir.\n\n"
    "Ders Alanı: {DOMAIN}\n"
    "Müfredat Referans Kavramları (RAG Anchors): {GLOSSARY}\n\n"
    "Görev:\n"
    "1. Referans kavramları ve transkriptte geçen ek teknik terimleri (çok kelimeli yapılar dahil) çıkar.\n"
    "2. Her terim için transkripte sadık kalarak CEFR B1 seviyesinde anlaşılır bir açıklama oluştur.\n"
    "3. Çıktıyı kesinlikle JSON dizisi olarak ver: [{{\"term\": \"...\", \"explanation\": \"...\"}}]\n\n"
    "Transkript: {TRANSCRIPT}"
)

PROMPT_RAG_EN = (
    "You are an educational terminology assistant. Below are reference curriculum concepts retrieved from course slides along with the lecture transcript.\n\n"
    "Domain: {DOMAIN}\n"
    "Curriculum Reference Concepts (RAG Anchors): {GLOSSARY}\n\n"
    "Task:\n"
    "1. Extract the core domain terms mentioned in the transcript (including multi-word terms), grounded in the curriculum concepts.\n"
    "2. Write a short CEFR B1 level explanation for each term based strictly on the transcript context.\n"
    "3. Return STRICTLY a JSON array of objects: [{{\"term\": \"...\", \"explanation\": \"...\"}}]\n\n"
    "Transcript: {TRANSCRIPT}"
)


def build_strategy_prompt(
    strategy: str,
    transcript: str,
    lang: str = "tr",
    domain: str = "",
    glossary_terms: Sequence[str] = (),
) -> str:
    """Build prompt string for the requested prompting strategy."""
    is_tr = str(lang).lower().startswith("tr")
    lang_name = "Turkish" if is_tr else "English"

    if strategy == "zero_shot":
        return PROMPT_ZERO_SHOT.format(LANG=lang_name, TRANSCRIPT=transcript)
    elif strategy == "cot":
        tmpl = PROMPT_COT_TR if is_tr else PROMPT_COT_EN
        return tmpl.format(TRANSCRIPT=transcript)
    elif strategy == "rag":
        tmpl = PROMPT_RAG_TR if is_tr else PROMPT_RAG_EN
        terms_str = ", ".join(glossary_terms[:25]) if glossary_terms else ("Genel Ders Kavramları" if is_tr else "General Lecture Concepts")
        return tmpl.format(
            DOMAIN=domain or ("Eğitim Dersi" if is_tr else "Educational Lecture"),
            GLOSSARY=terms_str,
            TRANSCRIPT=transcript,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_prompt_strategies(
    clip_id: str,
    transcript: str,
    lang: str = "tr",
    model_id: str = "Qwen/Qwen3.5-4B",
    use_mock: bool = False,
) -> list[dict]:
    """Compare Zero-Shot, CoT, and RAG prompting strategies on a single clip."""
    gold_path = ROOT / "data" / "gold_terms.json"
    gold_spec = {}
    if gold_path.exists():
        try:
            gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
            gold_spec = gold_data.get(clip_id, {})
        except Exception:
            pass

    domain = gold_spec.get("domain", "Educational Lecture")
    # The glossary arm must NOT receive the gold answer list: doing so makes its recall a
    # copy task and makes the strategy comparison circular. When a held-out source exists
    # it is used; otherwise the arm is skipped rather than run with leaked labels.
    leaked_gold_glossary = [t["canonical"] for t in gold_spec.get("terms", [])]
    external = (ROOT / "data" / f"glossary_{clip_id}.json")
    if external.exists():
        glossary = json.loads(external.read_text(encoding="utf-8"))
        glossary_source = f"external:{external.name}"
    else:
        glossary = []
        glossary_source = "none"

    strategies = ["zero_shot", "cot"]
    if glossary or use_mock:
        strategies.append("rag")
    results = []

    for strat in strategies:
        prompt = build_strategy_prompt(strat, transcript, lang=lang, domain=domain, glossary_terms=glossary)

        t0 = time.perf_counter()
        if use_mock:
            # Deterministic simulation for tests only. The term lists are slices of the gold
            # answer key, so these numbers measure nothing; rows are tagged so they cannot be
            # mistaken for a run.
            pool = [t["canonical"] for t in gold_spec.get("terms", [])] or ["placeholder"]
            sizes = {"zero_shot": 10, "cot": 14, "rag": 20}
            terms = [{"term": t, "explanation": "Simulated"} for t in pool[: sizes.get(strat, 10)]]
            elapsed = 0.05
        else:
            if "gpt" in model_id.lower():
                raw = generate_openai(model_id, prompt)
            else:
                raw = generate_hf(model_id, prompt)
            elapsed = time.perf_counter() - t0
            terms = parse_terms(raw)

        eval_res = evaluate_clip(terms, gold_spec) if gold_spec else {"gold_count": 0, "emitted_count": len(terms), "tp": 0, "precision": 0, "recall": 0, "f1": 0}

        results.append({
            "clip": clip_id,
            "lang": lang.upper(),
            "model": model_id,
            "strategy": strat,
            "glossary_source": glossary_source,
            "strategy_label": {"zero_shot": "Zero-Shot Baseline", "cot": "Chain-of-Thought (CoT)", "rag": "Glossary-based RAG"}[strat],
            "simulated": bool(use_mock),
            "latency_s": round(elapsed, 2),
            "n_terms": len(terms),
            "gold_count": eval_res["gold_count"],
            "tp": eval_res["tp"],
            "precision": round(eval_res["precision"] * 100, 2),
            "recall": round(eval_res["recall"] * 100, 2),
            "f1": round(eval_res["f1"] * 100, 2),
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prompting strategies for terminology extraction")
    parser.add_argument("--clip", default="TR-1", help="Clip ID (e.g. TR-1 or EN-1), or 'all' for every clip in data/gold_terms.json")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="LLM checkpoint")
    parser.add_argument("--test", action="store_true", help="Run quick simulation test")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the result rows to this JSONL file (required for a citable run)",
    )
    args = parser.parse_args()

    manifest_path = ROOT / "data" / "manifest.jsonl"
    fallback = {
        "TR-1": ("Proje yönetimi kapsamında kazanılmış değer ve maliyet sapması temel göstergelerdir.", "tr"),
    }
    transcripts = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("id") and item.get("reference"):
                transcripts[item["id"]] = (item["reference"], item.get("lang", "en"))

    gold_path = ROOT / "data" / "gold_terms.json"
    gold_ids = list(json.loads(gold_path.read_text(encoding="utf-8")).keys()) if gold_path.exists() else []
    clips = gold_ids if args.clip == "all" else [args.clip]

    all_rows = []
    for clip_id in clips:
        transcript, lang = transcripts.get(clip_id, fallback.get(clip_id, ("", "en")))
        if not transcript:
            print(f"SKIP {clip_id}: no reference transcript in data/manifest.jsonl")
            continue
        print(f"Comparing Prompting Strategies on {clip_id} (Lang: {lang.upper()}) using {args.model}...")
        res = evaluate_prompt_strategies(clip_id, transcript, lang=lang, model_id=args.model, use_mock=args.test)
        all_rows.extend(res)

        print("\n" + "=" * 95)
        print(f"{'Strategy':<26} {'Terms':<7} {'Gold':<6} {'TP':<5} {'P (%)':<8} {'R (%)':<8} {'F1 (%)':<8} {'Latency (s)':<10}")
        print("-" * 95)
        for r in res:
            print(
                f"{r['strategy_label']:<26} {r['n_terms']:<7} {r['gold_count']:<6} {r['tp']:<5} "
                f"{r['precision']:<8.1f} {r['recall']:<8.1f} {r['f1']:<8.1f} {r['latency_s']:<10.2f}"
            )
        print("=" * 95)

    if args.out:
        if not all_rows:
            raise SystemExit("no rows produced; refusing to write an empty artifact")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for r in all_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
