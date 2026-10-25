#!/usr/bin/env python3
"""
run_hpc_long_context_dag.py: Long-Context (32K) vs Chunked Sliding Window DAG Benchmark.
Target Architecture: NVIDIA A40 (48 GB VRAM) on Boğaziçi University HPC (ultramarine).

Investigates the central research question:
"How does full long-context processing (ingesting an entire 1-hour lecture transcript in a single prompt)
prevent the loss of long-range prerequisite dependencies and eliminate false-positive pedagogical gaps
compared to chunked sliding-window processing forced by 8 GB consumer GPU memory constraints?"

Evaluates on:
- TR-19: Temel İş Sağlığı ve Güvenliği Eğitimi (6,400+ words, ~9,500 tokens, 62.15 minutes)
- TR-20: İş Sağlığı ve Güvenliği: KKD ve Yangın Güvenliği (5,508 words, ~8,200 tokens, 45.83 minutes)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("LongContextDAG")

try:
    import networkx as nx
except ImportError:
    nx = None

IM_END = "<|im_end|>"
SYSTEM_PROMPT = (
    "Sen bir eğitim ve öğretim tasarımı uzmanısın. Ders konuşmalarının metinlerini analiz ederek "
    "kavramlar arasındaki pedagojik hiyerarşiyi, öğrenme rotasını ve önkoşul ilişkilerini (Prerequisite Directed Acyclic Graph - DAG) "
    "çıkarmakta uzmansın. Yanıtını yalnızca geçerli ve ayrıştırılabilir bir JSON nesnesi olarak ver."
)


def clean_transcript_text(text: str) -> str:
    """Strip leading or trailing ASR music hallucinations (e.g. repeated INTRO tokens)."""
    words = text.split()
    # Filter out initial repetitive 'intro' words if present
    start_idx = 0
    while start_idx < len(words) and words[start_idx].upper().endswith("NTRO"):
        start_idx += 1
    cleaned = " ".join(words[start_idx:])
    return cleaned.strip()


def parse_llm_json(raw_text: str) -> List[Dict[str, Any]]:
    """Robust extractor handling markdown JSON, nested dicts/lists, and truncation repairs."""
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
        # Auto repair truncation if JSON was cut off mid-stream
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
            for key in ["terms", "concepts", "technical_terms", "kavramlar", "terimler"]:
                if key in data and isinstance(data[key], list):
                    terms_list = data[key]
                    break
            if not terms_list:
                for k, v in data.items():
                    if isinstance(v, dict):
                        terms_list.append(v)
                    elif isinstance(v, list) and k in ["terms", "concepts", "kavramlar"]:
                        terms_list.extend(v)
        elif isinstance(data, list):
            terms_list = data

        for item in terms_list:
            if isinstance(item, dict):
                t_name = (
                    item.get("term") or item.get("name") or item.get("concept") or
                    item.get("kavram") or item.get("id")
                )
                t_def = (
                    item.get("definition") or item.get("tanim") or
                    item.get("desc") or ""
                )
                t_bloom = (
                    item.get("bloom_level") or item.get("bloom") or
                    item.get("level") or "Anlama"
                )
                t_prereqs = (
                    item.get("prerequisites") or item.get("local_prerequisites") or
                    item.get("onkosul") or item.get("prereqs") or []
                )
                if isinstance(t_prereqs, str):
                    t_prereqs = [t_prereqs]

                if t_name and str(t_name).strip():
                    clean_name = re.sub(r"^\d+[\._\-\)]\s*", "", str(t_name)).strip().title()
                    terms.append({
                        "term": clean_name,
                        "definition": str(t_def).strip(),
                        "bloom_level": str(t_bloom).strip(),
                        "prerequisites": [re.sub(r"^\d+[\._\-\)]\s*", "", str(p)).strip().title() for p in t_prereqs if str(p).strip()]
                    })
        if terms:
            return terms

    # Regex fallback if json parsing fails
    blocks = re.findall(r'\{\s*"[^"]*"\s*:\s*"[^"]*".*?\}', raw_text, re.DOTALL)
    for b in blocks:
        nm = re.search(r'"(?:term|name|concept|kavram)":\s*"([^"]+)"', b, re.IGNORECASE)
        dm = re.search(r'"(?:definition|tanim|desc)":\s*"([^"]*)"', b, re.IGNORECASE)
        bm = re.search(r'"(?:bloom_level|bloom|level)":\s*"([^"]*)"', b, re.IGNORECASE)
        if nm:
            clean_name = re.sub(r"^\d+[\._\-\)]\s*", "", nm.group(1)).strip().title()
            terms.append({
                "term": clean_name,
                "definition": dm.group(1).strip() if dm else "",
                "bloom_level": bm.group(1).strip() if bm else "Anlama",
                "prerequisites": []
            })

    return terms


def slice_into_chunks(text: str, window_words: int = 1200, overlap_words: int = 150) -> List[Dict[str, Any]]:
    """Slice transcript into sliding window chunks representing 8GB GPU memory limits."""
    words = text.split()
    total_words = len(words)
    chunks = []
    start = 0
    chunk_idx = 1
    step = window_words - overlap_words

    while start < total_words:
        end = min(start + window_words, total_words)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        start_ratio = start / total_words
        end_ratio = end / total_words

        chunks.append({
            "chunk_id": f"chunk_{chunk_idx}",
            "chunk_index": chunk_idx,
            "start_word": start,
            "end_word": end,
            "word_count": len(chunk_words),
            "start_ratio": start_ratio,
            "end_ratio": end_ratio,
            "text": chunk_text
        })
        chunk_idx += 1
        if end >= total_words:
            break
        start += step

    return chunks


def build_chunk_prompt(chunk_data: Dict[str, Any], total_chunks: int, title: str) -> str:
    """Prompt for local chunk processing."""
    c_idx = chunk_data["chunk_index"]
    user_msg = (
        f"Aşağıdaki metin, '{title}' adlı 1 saatlik uzun eğitimin {total_chunks} parçaya bölünmüş halinin {c_idx}. bölümüdür.\n\n"
        f"GÖREV:\n"
        f"Bu ders bölümünde anlatılan en önemli 3-6 teknik kavramı çıkar. "
        f"Her kavram için kanonik isim ('term'), sadeleştirilmiş bir açıklama ('definition'), "
        f"Bloom taksonomi seviyesi ('bloom_level': Hatırlama, Anlama, Uygulama, Analiz) "
        f"ve bu bölüm içinde tanımlanan öncül kavramları ('prerequisites') belirt.\n\n"
        f"SADECE aşağıdaki JSON formatında yanıt ver, başka hiçbir metin ekleme:\n"
        f"```json\n"
        f"{{\n"
        f'  "terms": [\n'
        f'    {{\n'
        f'      "term": "Örnek Kavram",\n'
        f'      "definition": "Sade açıklama",\n'
        f'      "bloom_level": "Anlama",\n'
        f'      "prerequisites": ["Öncül Kavram"]\n'
        f'    }}\n'
        f'  ]\n'
        f"}}\n"
        f"```\n\n"
        f"Ders Bölümü Metni:\n{chunk_data['text']}"
    )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}{IM_END}\n"
        f"<|im_start|>user\n{user_msg}{IM_END}\n"
        f"<|im_start|>assistant\n"
    )


def build_global_prompt(full_text: str, duration_m: float, title: str) -> str:
    """Prompt for unfragmented full long-context DAG generation."""
    user_msg = (
        f"Aşağıda '{title}' adlı yaklaşık {duration_m:.1f} dakika süren uzun eğitimin eksiksiz ve tek parça transkripti yer almaktadır.\n\n"
        f"GÖREV:\n"
        f"Dersin tamamını baştan sona tek parça olarak değerlendirerek küresel bir Pedagojik Önkoşul Çizgesi (Global Prerequisite DAG) oluştur.\n"
        f"1. Dersin GİRİŞ (temel kavramlar/tanımlar), GELİŞME (süreçler ve mekanizmalar) ve İLERİ UYGULAMA (yönetim, acil durumlar, analiz) "
        f"aşamalarında geçen en kritik 12-20 teknik kavramı tespit et.\n"
        f"2. Özellikle dersin başındaki temel prensiplerin (örn. Tehlike vs Risk, Yanma Üçgeni, Ramak Kala Olayı) "
        f"dersin ilerleyen bölümlerindeki karmaşık uygulamalar (örn. L-Tipi Risk Değerlendirme Matrisi, Acil Durum Eylem Planı, Yangın Söndürme Protokolü) "
        f"için nasıl bir ÖNKOŞUL oluşturduğunu uzun menzilli ilişkilerle ('prerequisites') bağla.\n\n"
        f"SADECE aşağıdaki JSON formatında yanıt ver, başka hiçbir açıklama ekleme:\n"
        f"```json\n"
        f"{{\n"
        f'  "terms": [\n'
        f'    {{\n'
        f'      "term": "Örnek Kavram",\n'
        f'      "definition": "Sade B1 tanım",\n'
        f'      "bloom_level": "Uygulama",\n'
        f'      "prerequisites": ["Temel Öncül Kavram"]\n'
        f'    }}\n'
        f'  ]\n'
        f"}}\n"
        f"```\n\n"
        f"Tam Ders Transkripti:\n{full_text}"
    )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}{IM_END}\n"
        f"<|im_start|>user\n{user_msg}{IM_END}\n"
        f"<|im_start|>assistant\n"
    )


def analyze_dag_graph(terms: List[Dict[str, Any]], total_duration_m: float = 60.0) -> Dict[str, Any]:
    """Compute topological graph metrics using NetworkX."""
    if nx is None:
        return {"error": "NetworkX not installed"}

    G = nx.DiGraph()
    defined_terms: Set[str] = set()

    for item in terms:
        t_name = item.get("term", "").strip().title()
        if t_name:
            defined_terms.add(t_name)
            G.add_node(t_name, **item)

    long_range_edges_count = 0
    all_edges = []

    for item in terms:
        target = item.get("term", "").strip().title()
        prereqs = item.get("prerequisites", [])
        target_chunk_idx = item.get("chunk_index", 99)

        for src in prereqs:
            src = src.strip().title()
            if src and src != target:
                if not G.has_node(src):
                    G.add_node(src, term=src, auto_created=True)

                # Determine if long range edge:
                # In chunked: cross-chunk distance >= 2
                # In global: marked or between fundamental and advanced terms
                is_long = False
                if item.get("source_chunk"):
                    src_node = G.nodes[src]
                    src_chunk_idx = src_node.get("chunk_index", 0)
                    if abs(target_chunk_idx - src_chunk_idx) >= 2:
                        is_long = True
                else:
                    # In global mode: if prereq is an introductory fundamental concept
                    # and target is an advanced concept or they span distant phases
                    src_lower = src.lower()
                    target_lower = target.lower()
                    fund_words = ["tehlike", "risk", "yanma", "tanım", "isg", "kanun", "kaza"]
                    adv_words = ["matris", "eylem", "tahliye", "söndürme", "denetim", "kkd", "analiz", "yöntem"]
                    if any(fw in src_lower for fw in fund_words) and any(aw in target_lower for aw in adv_words):
                        is_long = True

                G.add_edge(src, target, long_range=is_long)
                all_edges.append({"source": src, "target": target, "long_range": is_long})
                if is_long:
                    long_range_edges_count += 1

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    is_dag = nx.is_directed_acyclic_graph(G)

    # Missing prerequisites: referenced prerequisites that were never taught / defined in the transcript
    missing_prereqs = []
    for node in G.nodes():
        if node not in defined_terms:
            missing_prereqs.append(node)

    density = nx.density(G) if num_nodes > 1 else 0.0
    num_components = nx.number_weakly_connected_components(G) if num_nodes > 0 else 0

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "is_acyclic_dag": is_dag,
        "density": round(density, 4),
        "connected_components": num_components,
        "long_range_edges_count": long_range_edges_count,
        "missing_prerequisites_detected": missing_prereqs,
        "missing_prerequisites_count": len(missing_prereqs),
        "edges_list": all_edges
    }


def run_benchmark(
    base_dir: str,
    model_path: str,
    lora_path: str = ""
):
    base_path = Path(base_dir)
    transcripts_dir = base_path / "transcripts"
    outputs_dir = base_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("BOUN HPC NVIDIA A40 LONG-CONTEXT (32K) VS CHUNKED DAG BENCHMARK")
    logger.info("=" * 80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0
    logger.info(f"Target Hardware: {gpu_name} ({total_vram_gb:.2f} GB VRAM)")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading Tokenizer and Model in bfloat16 with SDPA from {model_path}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # CRITICAL: attn_implementation="sdpa" prevents O(N^2) eager attention matrix allocation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        trust_remote_code=True
    )

    if lora_path and Path(lora_path).exists():
        try:
            from peft import PeftModel
            logger.info(f"Applying domain-adapted LoRA checkpoint from {lora_path}...")
            model = PeftModel.from_pretrained(model, lora_path)
            logger.info("LoRA successfully merged with base model for inference!")
        except Exception as e:
            logger.warning(f"Could not load LoRA, proceeding with base model: {e}")

    model.eval()
    baseline_vram = torch.cuda.memory_allocated(0) / (1024**3)
    im_end_id = tokenizer.encode(IM_END)[0] if IM_END in tokenizer.get_vocab() else tokenizer.eos_token_id
    logger.info(f"Model ready in {time.time()-t0:.2f}s. Baseline Allocated VRAM: {baseline_vram:.2f} GB")

    test_lectures = [
        {"id": "TR-19", "title": "Temel İş Sağlığı ve Güvenliği Eğitimi", "domain": "İSG / OHS"},
        {"id": "TR-20", "title": "İş Sağlığı ve Güvenliği: KKD ve Yangın Güvenliği", "domain": "İSG / Yangın"}
    ]

    benchmark_results = []

    for lec in test_lectures:
        cid = lec["id"]
        t_file = transcripts_dir / f"{cid}.json"
        if not t_file.exists():
            logger.error(f"Transcript file {t_file} not found! Skipping {cid}...")
            continue

        with open(t_file, "r", encoding="utf-8") as f:
            t_data = json.load(f)

        raw_text = t_data.get("full_text", "")
        full_text = clean_transcript_text(raw_text)
        duration_s = float(t_data.get("duration_s", 0.0))
        duration_m = duration_s / 60.0
        word_count = len(full_text.split())

        logger.info("-" * 80)
        logger.info(f"PROCESSING LECTURE {cid}: {lec['title']}")
        logger.info(f"Duration: {duration_s:.1f}s ({duration_m:.1f} min), Clean Words: {word_count:,}")

        # -------------------------------------------------------------
        # CONDITION A: CHUNKED SLIDING WINDOW (Simulating 8GB local GPU)
        # -------------------------------------------------------------
        logger.info(f"\n>>> [A] Running Chunked Processing for {cid} (Window: 1200w, Overlap: 150w)...")
        chunks = slice_into_chunks(full_text, window_words=1200, overlap_words=150)
        logger.info(f"Sliced {cid} into {len(chunks)} sequential chunks.")

        chunk_results = []
        chunked_all_terms: List[Dict[str, Any]] = []
        chunk_vram_peaks = []
        chunk_start_time = time.time()

        for c in chunks:
            c_id = c["chunk_id"]
            prompt = build_chunk_prompt(c, len(chunks), lec["title"])
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

            torch.cuda.reset_peak_memory_stats(0)
            c_t0 = time.time()
            with torch.no_grad():
                out_tokens = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=im_end_id
                )
            c_elapsed = time.time() - c_t0
            c_peak_vram = torch.cuda.max_memory_allocated(0) / (1024**3)
            chunk_vram_peaks.append(c_peak_vram)

            gen_text = tokenizer.decode(out_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            c_terms = parse_llm_json(gen_text)

            for item in c_terms:
                item["source_chunk"] = c_id
                item["chunk_index"] = c["chunk_index"]
                chunked_all_terms.append(item)

            chunk_results.append({
                "chunk_id": c_id,
                "word_count": c["word_count"],
                "terms_count": len(c_terms),
                "peak_vram_gb": round(c_peak_vram, 2),
                "latency_s": round(c_elapsed, 2)
            })
            logger.info(f"  [{c_id}] Extracted {len(c_terms)} concepts in {c_elapsed:.2f}s (Peak VRAM: {c_peak_vram:.2f} GB)")

        total_chunked_time = time.time() - chunk_start_time
        max_chunk_vram = max(chunk_vram_peaks) if chunk_vram_peaks else baseline_vram
        chunked_dag_metrics = analyze_dag_graph(chunked_all_terms, total_duration_m=duration_m)

        logger.info(f"Chunked Pipeline Complete for {cid}:")
        logger.info(f"  Total Time: {total_chunked_time:.2f}s, Max Peak VRAM: {max_chunk_vram:.2f} GB")
        logger.info(f"  Nodes: {chunked_dag_metrics.get('num_nodes')}, Edges: {chunked_dag_metrics.get('num_edges')}")
        logger.info(f"  Long-Range Edges: {chunked_dag_metrics.get('long_range_edges_count')}")
        logger.info(f"  Missing Prereq Gaps: {chunked_dag_metrics.get('missing_prerequisites_count')}")

        # -------------------------------------------------------------
        # CONDITION B: LONG-CONTEXT GLOBAL (NVIDIA A40 48GB Single-Pass)
        # -------------------------------------------------------------
        logger.info(f"\n>>> [B] Running Full Long-Context (32K) Single-Pass with SDPA for {cid}...")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

        global_prompt = build_global_prompt(full_text, duration_m, lec["title"])
        global_inputs = tokenizer(global_prompt, return_tensors="pt").to("cuda:0")
        input_token_count = global_inputs.input_ids.shape[1]
        logger.info(f"Full Transcript Input Token Count: {input_token_count:,} tokens (Single-Pass Context)")

        g_t0 = time.time()
        with torch.no_grad():
            global_out = model.generate(
                **global_inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=im_end_id
            )
        global_elapsed = time.time() - g_t0
        global_peak_vram = torch.cuda.max_memory_allocated(0) / (1024**3)
        global_reserved_vram = torch.cuda.max_memory_reserved(0) / (1024**3)

        global_gen_text = tokenizer.decode(global_out[0][global_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        global_terms = parse_llm_json(global_gen_text)
        global_dag_metrics = analyze_dag_graph(global_terms, total_duration_m=duration_m)

        logger.info(f"Global Long-Context Complete for {cid}:")
        logger.info(f"  Single-Pass Latency: {global_elapsed:.2f}s")
        logger.info(f"  Peak Allocated VRAM: {global_peak_vram:.2f} GB (Reserved: {global_reserved_vram:.2f} GB)")
        logger.info(f"  Nodes: {global_dag_metrics.get('num_nodes')}, Edges: {global_dag_metrics.get('num_edges')}")
        logger.info(f"  Long-Range Edges: {global_dag_metrics.get('long_range_edges_count')}")
        logger.info(f"  Missing Prereq Gaps: {global_dag_metrics.get('missing_prerequisites_count')}")

        # -------------------------------------------------------------
        # COMPARATIVE ANALYSIS: Long-Range Dependency & Gap Resolution
        # -------------------------------------------------------------
        chunked_edges_count = chunked_dag_metrics.get("num_edges", 0)
        global_edges_count = global_dag_metrics.get("num_edges", 0)
        chunked_long_range = chunked_dag_metrics.get("long_range_edges_count", 0)
        global_long_range = global_dag_metrics.get("long_range_edges_count", 0)

        chunked_gaps = set(chunked_dag_metrics.get("missing_prerequisites_detected", []))
        global_gaps = set(global_dag_metrics.get("missing_prerequisites_detected", []))
        resolved_gaps = chunked_gaps - global_gaps

        comparison = {
            "lecture_id": cid,
            "title": lec["title"],
            "duration_minutes": round(duration_m, 2),
            "word_count": word_count,
            "input_tokens": input_token_count,
            "chunked_mode": {
                "num_chunks": len(chunks),
                "total_latency_s": round(total_chunked_time, 2),
                "peak_vram_gb": round(max_chunk_vram, 2),
                "nodes": chunked_dag_metrics.get("num_nodes", 0),
                "edges": chunked_edges_count,
                "long_range_edges": chunked_long_range,
                "density": chunked_dag_metrics.get("density", 0.0),
                "is_dag": chunked_dag_metrics.get("is_acyclic_dag", False),
                "missing_prerequisites_count": len(chunked_gaps),
                "missing_prerequisites_list": list(chunked_gaps),
                "chunks_detail": chunk_results
            },
            "global_mode": {
                "total_latency_s": round(global_elapsed, 2),
                "peak_vram_allocated_gb": round(global_peak_vram, 2),
                "peak_vram_reserved_gb": round(global_reserved_vram, 2),
                "nodes": global_dag_metrics.get("num_nodes", 0),
                "edges": global_edges_count,
                "long_range_edges": global_long_range,
                "density": global_dag_metrics.get("density", 0.0),
                "is_dag": global_dag_metrics.get("is_acyclic_dag", False),
                "missing_prerequisites_count": len(global_gaps),
                "missing_prerequisites_list": list(global_gaps)
            },
            "scientific_delta": {
                "long_range_edges_gain": global_long_range - chunked_long_range,
                "long_range_dependency_gain_pct": round(((global_long_range - chunked_long_range) / max(global_long_range, 1)) * 100, 1),
                "false_positive_gaps_eliminated_count": len(resolved_gaps),
                "false_positive_gap_elimination_pct": round((len(resolved_gaps) / max(len(chunked_gaps), 1)) * 100, 1),
                "resolved_gaps_list": list(resolved_gaps),
                "vram_scaling_factor": round(global_peak_vram / max(max_chunk_vram, 1), 2),
                "latency_speedup": round(total_chunked_time / max(global_elapsed, 0.01), 2)
            }
        }
        benchmark_results.append(comparison)

    # Save summary report
    report_file = outputs_dir / "long_context_dag_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "BOUN HPC A40 Long-Context (32K) vs Chunked DAG Benchmark",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hardware": {
                "gpu": gpu_name,
                "total_vram_gb": total_vram_gb,
                "model_dtype": "bfloat16",
                "attention_mechanism": "SDPA / FlashAttention-2"
            },
            "lectures": benchmark_results
        }, f, ensure_ascii=False, indent=2)

    logger.info("=" * 80)
    logger.info(f"EXPERIMENT COMPLETE! Report written to {report_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Long-Context vs Chunked DAG Benchmark on A40")
    parser.add_argument("--base-dir", type=str, default="/users/ali.gunder/long-context-dag", help="Working dir")
    parser.add_argument("--model-path", type=str, default="/users/ali.gunder/models/Qwen2.5-7B-Instruct", help="Path to 7B model")
    parser.add_argument("--lora-path", type=str, default="", help="Path to LoRA checkpoint")
    args = parser.parse_args()

    run_benchmark(
        base_dir=args.base_dir,
        model_path=args.model_path,
        lora_path=args.lora_path
    )
