"""run_hpc_cross_layer_curve.py: Empirical Cross-Layer Error Propagation Benchmark on NVIDIA A40.

Measures the continuous degradation curve of LLM Terminology Extraction F1
and Hallucination rate as ASR Word Error Rate (WER) scales across real ASR models:
- Whisper Tiny (~25-35% TR WER)
- Whisper Base (~18-25% TR WER)
- Whisper Small (~10-15% TR WER)
- Whisper Medium (~3.5-5% TR WER)
- Whisper Large-v3 Turbo (~2.5-3.5% TR WER)
- Whisper Large-v3 (~1.8-2.5% TR WER)
- Gold Ground Truth (0.0% WER reference baseline)

Answers the central open limitation of the thesis:
"ASR hatalarının LLM terim çıkarımına kademeli yayılımı sürekli bir kalite eğrisi olarak
nasıl değişiyor ve ASR WER %5 arttığında LLM terim F1 skoru yüzde kaç düşüyor?"
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
from typing import Any, Dict, List, Optional, Tuple

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("CrossLayerPropagation")


def normalize_turkish_text(text: str) -> str:
    """Normalize text for consistent WER / CER calculation."""
    text = text.lower()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer_cer(ref: str, hyp: str) -> Tuple[float, float]:
    """Compute exact Word Error Rate (WER) and Character Error Rate (CER) via Levenshtein DP."""
    ref_norm = normalize_turkish_text(ref)
    hyp_norm = normalize_turkish_text(hyp)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    # Word level Levenshtein
    n_ref = len(ref_words)
    n_hyp = len(hyp_words)
    if n_ref == 0:
        return 0.0 if n_hyp == 0 else 1.0, 0.0

    dp = [[0] * (n_hyp + 1) for _ in range(n_ref + 1)]
    for i in range(n_ref + 1):
        dp[i][0] = i
    for j in range(n_hyp + 1):
        dp[0][j] = j

    for i in range(1, n_ref + 1):
        for j in range(1, n_hyp + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )
    wer = dp[n_ref][n_hyp] / float(n_ref)

    # Character level Levenshtein
    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))
    c_ref = len(ref_chars)
    c_hyp = len(hyp_chars)
    if c_ref == 0:
        return wer, 0.0

    dp_c = [[0] * (c_hyp + 1) for _ in range(c_ref + 1)]
    for i in range(c_ref + 1):
        dp_c[i][0] = i
    for j in range(c_hyp + 1):
        dp_c[0][j] = j

    for i in range(1, c_ref + 1):
        for j in range(1, c_hyp + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            dp_c[i][j] = min(
                dp_c[i - 1][j] + 1,
                dp_c[i][j - 1] + 1,
                dp_c[i - 1][j - 1] + cost
            )
    cer = dp_c[c_ref][c_hyp] / float(c_ref)

    return wer, cer


def transcribe_audio_whisper(
    model_name: str,
    audio_path: str,
    language: str,
    device: str = "cuda:0"
) -> Tuple[str, float]:
    """Transcribe audio using transformers Whisper pipeline bypassing ffmpeg."""
    import numpy as np
    import scipy.io.wavfile as wavfile
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    torch.cuda.empty_cache()
    t0 = time.time()

    # Load audio natively via scipy to avoid ffmpeg requirement
    sr, audio_raw = wavfile.read(audio_path)
    if audio_raw.dtype == np.int16:
        audio_float = audio_raw.astype(np.float32) / 32768.0
    elif audio_raw.dtype == np.int32:
        audio_float = audio_raw.astype(np.float32) / 2147483648.0
    else:
        audio_float = audio_raw.astype(np.float32)

    if len(audio_float.shape) > 1:
        audio_float = np.mean(audio_float, axis=1)

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        torch_dtype=torch.float16,
        device=device
    )

    audio_input = {"raw": audio_float, "sampling_rate": sr}
    result = pipe(audio_input, generate_kwargs={"language": language, "task": "transcribe", "max_new_tokens": 400})
    elapsed = time.time() - t0
    transcription = result["text"].strip()

    del pipe
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    return transcription, elapsed


def evaluate_extracted_terms(
    extracted_terms: List[str],
    gold_term_definitions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Strict 1-to-1 evaluation of extracted terms against closed gold list."""
    matched_golds = set()
    true_positives = []
    false_positives = []

    # Sort golds by length descending to match longest canonical first
    sorted_golds = sorted(gold_term_definitions, key=lambda g: len(g["canonical"]), reverse=True)
    stop_generic = {"proje", "sistem", "yönetim", "analiz", "plan", "the", "and", "data"}

    for term in extracted_terms:
        term_clean = term.strip().lower()
        if len(term_clean) < 3:
            continue

        matched_gold_canonical = None
        for g in sorted_golds:
            canon = g["canonical"]
            if canon in matched_golds:
                continue  # 1-to-1: each gold matched at most once

            c_clean = canon.lower()
            aliases = [a.lower() for a in g.get("aliases", [])]

            if term_clean == c_clean or term_clean in aliases:
                matched_gold_canonical = canon
                break

            if (c_clean in term_clean and len(c_clean) >= 6) or (term_clean in c_clean and len(term_clean) >= 6):
                if term_clean not in stop_generic:
                    matched_gold_canonical = canon
                    break

        if matched_gold_canonical:
            matched_golds.add(matched_gold_canonical)
            true_positives.append(term)
        else:
            false_positives.append(term)

    tp = len(matched_golds)
    gold_total = len(gold_term_definitions)
    emitted_total = len(extracted_terms)

    recall = tp / float(gold_total) if gold_total > 0 else 0.0
    precision = tp / float(emitted_total) if emitted_total > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    hallucination_rate = len(false_positives) / float(emitted_total) if emitted_total > 0 else 0.0

    return {
        "gold_count": gold_total,
        "emitted_count": emitted_total,
        "tp": tp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "matched_golds": sorted(list(matched_golds)),
        "false_positives": false_positives[:10]
    }


def extract_terms_with_llm(
    model,
    tokenizer,
    transcript: str,
    language: str,
    device: str = "cuda:0"
) -> List[str]:
    """Run structured ChatML terminology extraction on transcript."""
    words = transcript.split()
    sample_text = " ".join(words[:450])

    IM_END = "<|im_end|>"
    if language == "tr":
        sys_msg = (
            "Sen ders ve eğitim konuşmalarını analiz eden uzman bir pedagoji ve terminoloji asistanısın. "
            "Görevin, verilen Türkçe ders transkriptindeki tüm temel teknik kavramları, terimleri ve metodolojileri tespit etmektir. "
            "ÖNEMLİ KURAL: Terimleri KESİNLİKLE Türkçe olarak yazınız, asla İngilizceye çevirmeyiniz! Metinde geçen orijinal Türkçe teknik terimleri yazınız. "
            "Her kavramı alt alta madde işaretiyle liste olarak yaz (- Terim Adı). "
            "Metindeki 10 ile 25 arasındaki anahtar kavramı eksiksiz çıkar. Başka hiçbir açıklama veya giriş cümlesi ekleme."
        )
        user_block = f"Türkçe ders konuşması metni:\n{sample_text}"
    else:
        sys_msg = (
            "You are an expert pedagogical terminology extraction assistant. "
            "Your task is to identify and extract all key technical concepts, terms, and methodologies from the given lecture transcript. "
            "List each term as a bullet point (- Term Name). "
            "Extract 10 to 25 key domain terms comprehensively. Do not include any introductory commentary."
        )
        user_block = f"Lecture transcript text:\n{sample_text}"

    prompt = (
        f"<|im_start|>system\n{sys_msg}{IM_END}\n"
        f"<|im_start|>user\n{user_block}{IM_END}\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.encode(IM_END)[0] if IM_END in tokenizer.get_vocab() else tokenizer.eos_token_id
        )

    resp_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    terms = []

    # Strategy 1: Bullet list lines
    for line in resp_text.splitlines():
        line = line.strip()
        if line.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.")):
            cleaned = re.sub(r"^[\*\-\d\.\•\s]+", "", line).strip()
            term_part = re.split(r"[:\-–—\(\[]", cleaned)[0].strip()
            if term_part and len(term_part) > 2 and len(term_part) < 60:
                terms.append(term_part)

    # Strategy 2: Quoted terms fallback
    if not terms:
        quoted = re.findall(r'"([^"\[\]\n\r\:\}\{]+)"', resp_text)
        skip = {"terms", "kavramlar", "technical_terms", "domain", "status", "result"}
        for q in quoted:
            q_clean = q.strip()
            if len(q_clean) > 2 and q_clean.lower() not in skip:
                terms.append(q_clean)

    # Clean punctuation artifacts
    cleaned_terms = []
    for t in terms:
        t_clean = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', t).strip()
        if t_clean and len(t_clean) > 2 and not re.match(r'^[0-9\W]+$', t_clean):
            cleaned_terms.append(t_clean)

    return list(dict.fromkeys(cleaned_terms))


def main():
    parser = argparse.ArgumentParser(description="Empirical Cross-Layer Error Propagation on NVIDIA A40")
    parser.add_argument("--base-dir", type=str, default="/users/ali.gunder/cross-layer")
    parser.add_argument("--model-path", type=str, default="/users/ali.gunder/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora-path", type=str, default="")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    audio_dir = base_dir / "audio"
    gold_dir = base_dir / "gold"
    out_dir = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(f"Target GPU: {gpu_name}")

    # Load closed gold terms
    gold_terms_file = gold_dir / "gold_terms.json"
    with open(gold_terms_file, "r", encoding="utf-8") as f:
        gold_terms_dict = json.load(f)

    # Clips to evaluate
    clips = [
        {"id": "TR-1", "lang": "tr", "audio": audio_dir / "TR-1.wav", "gold_txt": gold_dir / "TR-1.txt"},
        {"id": "EN-1", "lang": "en", "audio": audio_dir / "EN-1.wav", "gold_txt": gold_dir / "EN-1.txt"}
    ]

    # Models ladder from high WER to lowest WER
    asr_models = [
        {"name": "openai/whisper-tiny", "label": "Whisper Tiny", "type": "high_wer"},
        {"name": "openai/whisper-base", "label": "Whisper Base", "type": "mod_high_wer"},
        {"name": "openai/whisper-small", "label": "Whisper Small", "type": "med_wer"},
        {"name": "openai/whisper-medium", "label": "Whisper Medium", "type": "low_med_wer"},
        {"name": "openai/whisper-large-v3-turbo", "label": "Whisper Large-v3 Turbo", "type": "ultra_low_wer"},
        {"name": "openai/whisper-large-v3", "label": "Whisper Large-v3", "type": "gold_asr"}
    ]

    cache_file = out_dir / "cached_asr_transcripts.json"
    if cache_file.exists():
        logger.info(f"Loading cached ASR transcripts from {cache_file}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            asr_transcripts = json.load(f)
    else:
        logger.info("PHASE 1: Generating real ASR transcripts across the error ladder...")
        asr_transcripts = {}

        for c in clips:
            cid = c["id"]
            lang = c["lang"]
            gold_text = c["gold_txt"].read_text(encoding="utf-8").strip()
            asr_transcripts[cid] = []

            # 1. Gold baseline (WER = 0.0%)
            asr_transcripts[cid].append({
                "model_name": "Gold Reference Transcript",
                "label": "Gold Reference",
                "type": "gold_reference",
                "wer": 0.0,
                "cer": 0.0,
                "latency_s": 0.0,
                "text": gold_text
            })

            # 2. Real Whisper checkpoints
            for asr_info in asr_models:
                m_name = asr_info["name"]
                logger.info(f"Transcribing {cid} with {asr_info['label']} ({m_name})...")
                try:
                    hyp_text, asr_time = transcribe_audio_whisper(m_name, str(c["audio"]), lang)
                    wer, cer = compute_wer_cer(gold_text, hyp_text)
                    logger.info(f"  -> {asr_info['label']} on {cid}: WER: {wer*100:.2f}% | CER: {cer*100:.2f}% | Time: {asr_time:.2f}s")
                    asr_transcripts[cid].append({
                        "model_name": m_name,
                        "label": asr_info["label"],
                        "type": asr_info["type"],
                        "wer": round(wer, 4),
                        "cer": round(cer, 4),
                        "latency_s": round(asr_time, 2),
                        "text": hyp_text
                    })
                except Exception as e:
                    logger.error(f"Error transcribing {m_name}: {e}")

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(asr_transcripts, f, ensure_ascii=False, indent=2)
        logger.info(f"Cached all ASR transcripts to {cache_file}")

    # PHASE 2: Load LLM on GPU
    logger.info("PHASE 2: Loading EduTerm-7B-TR on NVIDIA A40...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )

    if args.lora_path and os.path.exists(args.lora_path):
        model = PeftModel.from_pretrained(model, args.lora_path)
        logger.info(f"Attached LoRA weights from {args.lora_path}")
    model.eval()

    # PHASE 3: Run LLM extraction and cross-layer evaluation
    logger.info("PHASE 3: Evaluating LLM Terminology Extraction F1 vs ASR WER...")
    all_points = []

    for c in clips:
        cid = c["id"]
        lang = c["lang"]
        gold_terms = gold_terms_dict.get(cid, {}).get("terms", [])

        for item in asr_transcripts[cid]:
            logger.info(f"Extracting terms for {cid} | ASR: {item['label']} (WER: {item['wer']*100:.2f}%)...")
            extracted = extract_terms_with_llm(model, tokenizer, item["text"], lang)
            eval_res = evaluate_extracted_terms(extracted, gold_terms)

            record = {
                "clip": cid,
                "language": lang.upper(),
                "asr_model": item["model_name"],
                "asr_label": item["label"],
                "asr_type": item["type"],
                "wer": item["wer"],
                "wer_pct": round(item["wer"] * 100, 2),
                "cer_pct": round(item["cer"] * 100, 2),
                "gold_count": eval_res["gold_count"],
                "emitted_count": eval_res["emitted_count"],
                "tp": eval_res["tp"],
                "precision": eval_res["precision"],
                "recall": eval_res["recall"],
                "f1": eval_res["f1"],
                "hallucination_rate": eval_res["hallucination_rate"],
                "matched_golds": eval_res["matched_golds"],
                "false_positives": eval_res["false_positives"]
            }
            all_points.append(record)
            logger.info(
                f"  RESULT [{cid} - {item['label']}]: WER={record['wer_pct']}% -> "
                f"Recall={record['recall']*100:.1f}%, Precision={record['precision']*100:.1f}%, "
                f"F1={record['f1']*100:.1f}%, Hallucination={record['hallucination_rate']*100:.1f}%"
            )
            logger.info(f"    Extracted ({len(extracted)} terms): {extracted[:6]}")
            logger.info(f"    Matched ({len(eval_res['matched_golds'])} golds): {eval_res['matched_golds'][:6]}")

    # Free GPU memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # PHASE 4: Mathematical Regression Slope & Sensitivity Analysis
    # F1(WER) = F1(0) - beta * WER
    # For Turkish (TR-1)
    tr_points = [p for p in all_points if p["clip"] == "TR-1"]
    en_points = [p for p in all_points if p["clip"] == "EN-1"]

    def compute_linear_fit(pts):
        xs = [p["wer"] for p in pts]
        ys = [p["f1"] for p in pts]
        n = len(xs)
        if n < 2:
            return 0.0, 0.0, 0.0
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
        den = sum((xs[i] - x_mean) ** 2 for i in range(n))
        beta = num / max(1e-9, den)
        alpha = y_mean - beta * x_mean
        # Pearson correlation
        s_y = sum((ys[i] - y_mean) ** 2 for i in range(n))
        r = num / max(1e-9, math.sqrt(den * s_y)) if s_y > 0 else 0.0
        r2 = r ** 2
        return alpha, beta, r2

    alpha_tr, beta_tr, r2_tr = compute_linear_fit(tr_points)
    alpha_en, beta_en, r2_en = compute_linear_fit(en_points)

    # Sensitivity: change in F1 for 5% increase in WER
    # dF1 = beta * 0.05
    delta_f1_per_5pct_wer_tr = abs(beta_tr * 0.05) * 100
    delta_f1_per_5pct_wer_en = abs(beta_en * 0.05) * 100

    report = {
        "title": "Empirical Cross-Layer Error Propagation: ASR WER to LLM Terminology F1 Curve",
        "hardware": {
            "gpu": gpu_name,
            "node": "Boğaziçi HPC (ultramarine)"
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sensitivity_analysis": {
            "turkish_tr1": {
                "regression_equation": f"F1(WER) = {alpha_tr:.4f} + ({beta_tr:.4f}) * WER",
                "slope_beta": round(beta_tr, 4),
                "r_squared": round(r2_tr, 4),
                "f1_drop_per_5pct_wer": round(delta_f1_per_5pct_wer_tr, 2),
                "takeaway": f"ASR WER %5 arttığında Türkçe terim çıkarma F1 skoru %{delta_f1_per_5pct_wer_tr:.2f} düşmektedir."
            },
            "english_en1": {
                "regression_equation": f"F1(WER) = {alpha_en:.4f} + ({beta_en:.4f}) * WER",
                "slope_beta": round(beta_en, 4),
                "r_squared": round(r2_en, 4),
                "f1_drop_per_5pct_wer": round(delta_f1_per_5pct_wer_en, 2),
                "takeaway": f"When ASR WER increases by 5%, English terminology extraction F1 drops by {delta_f1_per_5pct_wer_en:.2f}%."
            }
        },
        "data_points": all_points
    }

    out_file = out_dir / "cross_layer_measured_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Cross-layer report successfully written to {out_file}")

    print("\n" + "=" * 95)
    print(f"{'Clip':<6} | {'ASR Model':<26} | {'WER (%)':<8} | {'CER (%)':<8} | {'Recall':<8} | {'Precision':<10} | {'F1-Score':<9} | {'Hallucination':<13}")
    print("-" * 95)
    for p in all_points:
        print(
            f"{p['clip']:<6} | {p['asr_label']:<26} | {p['wer_pct']:>6.2f}% | {p['cer_pct']:>6.2f}% | "
            f"{p['recall']*100:>6.1f}% | {p['precision']*100:>8.1f}% | {p['f1']*100:>7.1f}% | {p['hallucination_rate']*100:>11.1f}%"
        )
    print("=" * 95)
    print(f"\n[MATHEMATICAL CROSS-LAYER DERIVATIVE]:")
    print(f"TR-1: {report['sensitivity_analysis']['turkish_tr1']['takeaway']}")
    print(f"EN-1: {report['sensitivity_analysis']['english_en1']['takeaway']}\n")


if __name__ == "__main__":
    main()
