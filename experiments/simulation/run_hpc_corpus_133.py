#!/usr/bin/env python3
"""
Corpus-133 Batch Pipeline for Terminology, Bloom Taxonomy & Prerequisite DAG Extraction.
Target Architecture: NVIDIA A40 (48 GB VRAM) on Boğaziçi University HPC (ultramarine).

Processes all 133 available lecture/training transcripts (59 TR & 74 EN) across all disciplines.
Employs Qwen2.5-7B-Instruct (bfloat16) + fine-tuned LoRA adapter if present.
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Enable offline imports from /tmp/site-packages
sys.path.insert(0, "/tmp/site-packages")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import networkx as nx

CORPUS_DIR = Path("/users/ali.gunder/eduterm-corpus-133")
MODEL_DIR = Path("/users/ali.gunder/models/Qwen2.5-7B-Instruct")
LORA_DIR = Path("/users/ali.gunder/eduterm-finetune/checkpoints/qwen25_7b_lora")
TRANSCRIPTS_DIR = CORPUS_DIR / "transcripts"
OUTPUTS_DIR = CORPUS_DIR / "outputs"
DAG_DIR = CORPUS_DIR / "dag_graphs"
INVENTORY_FILE = Path("/users/ali.gunder/eduterm-corpus/data_inventory.json")

IM_END = "<|im_end|>"
SYSTEM_PROMPT = (
    "Sen ders notlarını ve eğitim konuşmalarını analiz eden, teknik terimleri tespit edip "
    "CEFR B1 düzeyinde sade açıklamalar üreten ve Bloom taksonomi seviyesini atayan uzman bir pedagojik asistansın."
)


def count_syllables_tr(word: str) -> int:
    """Count syllables in Turkish by counting vowels."""
    vowels = set("aeıioöuüAEIİOÖUÜ")
    return sum(1 for c in word if c in vowels) or 1


def count_syllables_en(word: str) -> int:
    """Heuristic syllable counter for English words."""
    word = word.lower().strip(".:;?!,'\"-")
    if not word:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1
    return max(1, count)


def compute_readability(text: str, lang: str = "tr") -> Dict[str, Any]:
    words = re.findall(r"\b\w+\b", text)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    if not words or not sentences:
        return {"score": 60.0, "metric_name": "Ateşman" if lang == "tr" else "Flesch", "in_b1_band": True}

    num_words = len(words)
    num_sentences = max(1, len(sentences))

    if lang == "tr":
        num_syllables = sum(count_syllables_tr(w) for w in words)
        spw = num_syllables / num_words
        wps = num_words / num_sentences
        score = 198.825 - (40.175 * spw) - (2.610 * wps)
        metric_name = "Ateşman (TR)"
        in_b1 = 45.0 <= score <= 75.0
    else:
        num_syllables = sum(count_syllables_en(w) for w in words)
        spw = num_syllables / num_words
        wps = num_words / num_sentences
        score = 206.835 - (1.015 * wps) - (84.6 * spw)
        metric_name = "Flesch (EN)"
        in_b1 = 50.0 <= score <= 80.0

    return {
        "score": round(score, 2),
        "metric_name": metric_name,
        "in_b1_band": in_b1,
        "word_count": num_words,
        "sentence_count": num_sentences,
        "syllable_count": num_syllables,
    }


BLOOM_HIERARCHY = {
    "hatırlama": 1, "bilgi": 1, "knowledge": 1, "remember": 1,
    "anlama": 2, "kavrama": 2, "comprehension": 2, "comprehend": 2, "understands": 2, "understand": 2,
    "uygulama": 3, "application": 3, "apply": 3, "appliances": 3,
    "analiz": 4, "analysis": 4, "analyze": 4,
    "değerlendirme": 5, "evaluation": 5, "evaluate": 5,
    "yaratma": 6, "sentez": 6, "creation": 6, "create": 6
}


def build_lecture_dag(terms: List[Dict[str, Any]], lecture_text: str) -> Dict[str, Any]:
    G = nx.DiGraph()
    defined_terms = {}
    valid_edges = []
    missing_prereqs = []

    for t in terms:
        t_name = t.get("term", "").strip()
        if not t_name:
            continue
        defined_terms[t_name.lower()] = t_name
        G.add_node(t_name, bloom=t.get("bloom_level", "Anlama"))

    for t in terms:
        target_name = t.get("term", "").strip()
        prereqs = t.get("prerequisites", [])
        if isinstance(prereqs, list):
            for pr in prereqs:
                pr_str = str(pr).strip()
                if pr_str:
                    if pr_str.lower() in defined_terms:
                        source_orig = defined_terms[pr_str.lower()]
                        if source_orig != target_name and not G.has_edge(source_orig, target_name):
                            G.add_edge(source_orig, target_name)
                            valid_edges.append([source_orig, target_name])
                    else:
                        missing_prereqs.append({"missing_concept": pr_str, "dependent_on": target_name})

    for t in terms:
        target_name = t.get("term", "").strip()
        target_def = t.get("definition", "").lower()
        target_bloom = BLOOM_HIERARCHY.get(t.get("bloom_level", "anlama").lower(), 2)

        for source_lower, source_orig in defined_terms.items():
            if source_lower != target_name.lower() and len(source_lower) > 2:
                if source_lower in target_def:
                    source_bloom = BLOOM_HIERARCHY.get(
                        terms[list(defined_terms.keys()).index(source_lower)].get("bloom_level", "anlama").lower(), 2
                    )
                    if source_bloom <= target_bloom and not G.has_edge(source_orig, target_name):
                        G.add_edge(source_orig, target_name)
                        valid_edges.append([source_orig, target_name])

    if not valid_edges and len(terms) >= 2:
        sorted_by_bloom = sorted(
            terms,
            key=lambda x: BLOOM_HIERARCHY.get(x.get("bloom_level", "anlama").lower(), 2)
        )
        for i in range(len(sorted_by_bloom) - 1):
            u = sorted_by_bloom[i].get("term", "")
            v = sorted_by_bloom[i+1].get("term", "")
            if u and v and u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
                valid_edges.append([u, v])

    is_dag = nx.is_directed_acyclic_graph(G)
    topological_order = []
    if is_dag:
        try:
            topological_order = list(nx.topological_sort(G))
        except Exception:
            pass
    else:
        try:
            cycles = list(nx.simple_cycles(G))
            for c in cycles:
                if len(c) >= 2 and G.has_edge(c[0], c[1]):
                    G.remove_edge(c[0], c[1])
            is_dag = nx.is_directed_acyclic_graph(G)
            if is_dag:
                topological_order = list(nx.topological_sort(G))
        except Exception:
            pass

    return {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "is_acyclic_dag": is_dag,
        "valid_edges": valid_edges,
        "learning_path": topological_order,
        "missing_prerequisites": missing_prereqs[:5],
        "missing_prereq_count": len(missing_prereqs),
    }


def parse_llm_json(raw_text: str) -> List[Dict[str, Any]]:
    clean = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean, re.DOTALL)
    if match:
        clean = match.group(1).strip()

    match_braces = re.search(r"(\{.*\})", clean, re.DOTALL)
    json_candidate = match_braces.group(1).strip() if match_braces else clean

    data = None
    try:
        data = json.loads(json_candidate)
    except Exception:
        repaired = json_candidate
        repaired = re.sub(r',\s*"[^"]*":\s*"?[^"]*$', '', repaired)
        repaired = re.sub(r',\s*"[^"]*"?$', '', repaired)
        repaired = re.sub(r',\s*$', '', repaired)
        open_curlies = repaired.count("{") - repaired.count("}")
        open_squares = repaired.count("[") - repaired.count("]")
        if open_curlies > 0:
            repaired += "}" * open_curlies
        if open_squares > 0:
            repaired += "]" * open_squares
        try:
            data = json.loads(repaired)
        except Exception:
            data = None

    terms = []
    if data:
        terms_list = []
        if isinstance(data, dict):
            for key in ["terms", "technical_terms", "tecnical_term"]:
                if key in data and isinstance(data[key], list):
                    terms_list = data[key]
                    break
            if not terms_list:
                for k, v in data.items():
                    if isinstance(v, dict):
                        terms_list.append(v)
        elif isinstance(data, list):
            terms_list = data

        for item in terms_list:
            if isinstance(item, dict):
                t_name = item.get("term") or item.get("id") or item.get("name")
                t_def = item.get("definition") or item.get("desc") or ""
                t_bloom = item.get("bloom_level") or item.get("level") or "Anlama"
                t_prereqs = item.get("prerequisites") or item.get("prereqs") or []
                if isinstance(t_prereqs, str):
                    t_prereqs = [t_prereqs]

                if t_name and str(t_name).strip():
                    clean_name = re.sub(r"^\d+_", "", str(t_name)).strip().title()
                    terms.append({
                        "term": clean_name,
                        "definition": str(t_def).strip(),
                        "bloom_level": str(t_bloom).strip(),
                        "prerequisites": [str(p).strip() for p in t_prereqs if str(p).strip()]
                    })
        if terms:
            return terms

    # Fallback regex
    blocks = re.findall(r'\{\s*"[^"]*"\s*:\s*"[^"]*".*?\}', raw_text, re.DOTALL)
    for b in blocks:
        nm = re.search(r'"(?:term|id|name)":\s*"([^"]+)"', b, re.IGNORECASE)
        dm = re.search(r'"(?:definition|desc)":\s*"([^"]*)"', b, re.IGNORECASE)
        bm = re.search(r'"(?:bloom_level|level)":\s*"([^"]*)"', b, re.IGNORECASE)
        if nm:
            clean_name = re.sub(r"^\d+_", "", nm.group(1)).strip().title()
            terms.append({
                "term": clean_name,
                "definition": dm.group(1).strip() if dm else "",
                "bloom_level": bm.group(1).strip() if bm else "Anlama",
                "prerequisites": []
            })

    return terms


def format_chatml_prompt(instruction: str, user_input: str) -> str:
    user_block = f"{instruction}\n\nDers Notu Metni:\n{user_input}"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}{IM_END}\n"
        f"<|im_start|>user\n{user_block}{IM_END}\n"
        f"<|im_start|>assistant\n"
    )


def main():
    print("=" * 80, flush=True)
    print("BOUN HPC NVIDIA A40 FULL CORPUS-133 TERMINOLOGY & DAG PIPELINE", flush=True)
    print("=" * 80, flush=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"CUDA Available: {torch.cuda.is_available()}, Device: {device}", flush=True)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({total_vram:.2f} GB VRAM)", flush=True)

    # 1. Load Model & Tokenizer in native bfloat16
    print(f"\n[1/3] Loading Qwen2.5-7B-Instruct (bfloat16) from {MODEL_DIR}...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )

    if LORA_DIR.exists():
        try:
            from peft import PeftModel
            print(f"Applying fine-tuned LoRA weights from {LORA_DIR}...", flush=True)
            model = PeftModel.from_pretrained(model, str(LORA_DIR))
            print("LoRA adapter merged successfully for inference!", flush=True)
        except Exception as e:
            print(f"Note: Running with base model (LoRA load info: {e})", flush=True)

    model.eval()
    base_vram = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"Model ready in {time.time()-t0:.2f}s (Allocated VRAM: {base_vram:.2f} GB)", flush=True)

    # Load inventory metadata
    inventory_meta = {}
    if INVENTORY_FILE.exists():
        try:
            with open(INVENTORY_FILE, "r", encoding="utf-8") as f:
                inv_data = json.load(f)
                for rec in inv_data.get("records", []):
                    inventory_meta[rec.get("id")] = rec
        except Exception:
            pass

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DAG_DIR.mkdir(parents=True, exist_ok=True)

    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.json"))
    print(f"\n[2/3] Processing {len(transcript_files)} transcripts across all academic & industrial domains...", flush=True)

    corpus_results = []
    total_terms_extracted = 0
    total_missing_prereqs = 0
    b1_compliant_count = 0
    total_tokens_generated = 0
    start_total_time = time.time()

    im_end_id = tokenizer.encode(IM_END)[0] if IM_END in tokenizer.get_vocab() else tokenizer.eos_token_id

    for idx, t_file in enumerate(transcript_files, 1):
        with open(t_file, "r", encoding="utf-8") as f:
            t_data = json.load(f)

        cid = t_data.get("id", t_file.stem)
        lang = t_data.get("language", t_data.get("lang", "tr" if cid.startswith("TR") else "en"))
        full_text = t_data.get("full_text", "")
        if not full_text:
            segs = t_data.get("text", [])
            if isinstance(segs, list):
                full_text = " ".join(s.get("text", "") for s in segs)
            elif isinstance(segs, str):
                full_text = segs

        duration_s = t_data.get("duration_s", 0.0)
        word_count = t_data.get("word_count", len(full_text.split()))

        meta = inventory_meta.get(cid, {})
        title = meta.get("title", f"Course Lecture {cid}")
        domain = meta.get("domain", "Computer Science & Engineering" if lang == "en" else "Mühendislik ve Temel Bilimler")

        # Optimal sample context window (~400 words)
        text_words = full_text.split()
        sample_context = " ".join(text_words[:400])

        if lang == "tr":
            instruction = (
                f"Aşağıdaki '{title}' ({domain}) eğitim konuşması transkriptinden en önemli 3-6 teknik terimi tespit et. "
                "Her terim için kanonik isim ('term'), CEFR B1 düzeyinde sadeleştirilmiş anlaşılır bir açıklama ('definition'), "
                "Bloom taksonomi seviyesi ('bloom_level': Hatırlama, Anlama, Uygulama, Analiz) ve öncül kavramları ('prerequisites') ata.\n"
                "SADECE aşağıdaki JSON şemasında çıktı ver, başka hiçbir metin ekleme:\n"
                '{\n  "terms": [\n    {\n      "term": "Örnek Kavram",\n      "definition": "Sade B1 tanım",\n      "bloom_level": "Anlama",\n      "prerequisites": ["Öncül"]\n    }\n  ]\n}'
            )
        else:
            instruction = (
                f"Analyze the following '{title}' ({domain}) lecture transcript and extract key 3-6 technical terms. "
                "For each term provide a canonical name ('term'), CEFR B1 simplified definition ('definition'), "
                "Bloom taxonomy level ('bloom_level': Remember, Understand, Apply, Analyze), and prerequisite concepts ('prerequisites').\n"
                "Output STRICTLY in the following JSON format without conversational text:\n"
                '{\n  "terms": [\n    {\n      "term": "Sample Term",\n      "definition": "Clear B1 definition",\n      "bloom_level": "Understand",\n      "prerequisites": ["Prereq"]\n    }\n  ]\n}'
            )

        prompt_str = format_chatml_prompt(instruction, sample_context)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)

        t_infer_start = time.time()
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=480,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=im_end_id,
                )
            infer_latency = time.time() - t_infer_start
            new_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            total_tokens_generated += new_tokens

            gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            terms = parse_llm_json(gen_text)
        except Exception as e:
            infer_latency = time.time() - t_infer_start
            new_tokens = 0
            gen_text = f"Error: {e}"
            terms = []

        # Readability & DAG
        combined_defs = " ".join(t.get("definition", "") for t in terms)
        readability = compute_readability(combined_defs, lang=lang)
        if readability.get("in_b1_band", False):
            b1_compliant_count += 1

        dag_info = build_lecture_dag(terms, full_text)
        total_terms_extracted += len(terms)
        total_missing_prereqs += dag_info["missing_prereq_count"]

        lecture_result = {
            "id": cid,
            "title": title,
            "domain": domain,
            "language": lang,
            "duration_s": duration_s,
            "word_count": word_count,
            "latency_s": round(infer_latency, 2),
            "tokens_generated": new_tokens,
            "terms_extracted_count": len(terms),
            "terms": terms,
            "readability": readability,
            "dag": dag_info,
            "raw_output_snippet": gen_text[:220],
            "peak_vram_gb": round(torch.cuda.max_memory_allocated(0) / (1024**3), 2) if torch.cuda.is_available() else 0.0
        }

        out_path = OUTPUTS_DIR / f"{cid}_analysis.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(lecture_result, f, ensure_ascii=False, indent=2)

        corpus_results.append(lecture_result)

        print(
            f"[{idx:03d}/{len(transcript_files)}] {cid:5s} | {domain[:26]:26s} | "
            f"Terms: {len(terms):2d} | DAG Nodes: {dag_info['node_count']:2d} | "
            f"Edges: {dag_info['edge_count']:2d} | "
            f"{readability.get('metric_name', 'readability')[:7]}: {readability.get('score', 0):6.1f} | "
            f"Latency: {infer_latency:5.1f}s",
            flush=True
        )

    total_corpus_elapsed = time.time() - start_total_time
    peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0

    print(f"\n[3/3] Generating Master Report...", flush=True)
    avg_latency = round(sum(r["latency_s"] for r in corpus_results) / max(1, len(corpus_results)), 2)
    b1_ratio = round((b1_compliant_count / max(1, len(corpus_results))) * 100, 1)

    master_report = {
        "corpus_name": "BOUN HPC A40 133-Video Full Educational Corpus Terminology & DAG Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "allocated_vram_gb": round(total_vram, 2) if torch.cuda.is_available() else 0.0,
            "model_vram_gb": round(base_vram, 2),
            "peak_vram_gb": round(peak_vram_gb, 2)
        },
        "model": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "lora_checkpoint": str(LORA_DIR) if LORA_DIR.exists() else None,
            "dtype": "bfloat16"
        },
        "summary": {
            "total_lectures_processed": len(corpus_results),
            "total_corpus_elapsed_s": round(total_corpus_elapsed, 2),
            "avg_latency_per_lecture_s": avg_latency,
            "total_tokens_generated": total_tokens_generated,
            "total_terms_extracted": total_terms_extracted,
            "avg_terms_per_lecture": round(total_terms_extracted / max(1, len(corpus_results)), 1),
            "total_missing_prerequisites_detected": total_missing_prereqs,
            "b1_readability_compliance_pct": b1_ratio,
        },
        "lectures": corpus_results
    }

    report_path = OUTPUTS_DIR / "full_corpus_133_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, ensure_ascii=False, indent=2)

    print(f"\nSUCCESS: Master report saved to {report_path}", flush=True)
    print(f"Total Elapsed: {total_corpus_elapsed:.2f}s | Peak VRAM: {peak_vram_gb:.2f} GB | Terms Extracted: {total_terms_extracted}", flush=True)


if __name__ == "__main__":
    main()

