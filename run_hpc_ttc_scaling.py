#!/usr/bin/env python3
"""
HPC Test-Time Compute (TTC) Scaling & Self-Consistency Benchmark
Evaluates EduTerm-14B, Trendyol-8B, and Kumru-2B under N=1 (greedy), N=3 (sampling + majority vote),
and N=5 (self-consistency) inference compute budgets.
"""

import os
import sys
import json
import logging
import time
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_CONFIG = [
    {
        "id": "Kumru-2B",
        "family": "kumru",
        "base_path": "/users/ali.gunder/models/Kumru-2B",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_kumru_2b_cefr",
        "load_kwargs": {"torch_dtype": torch.float16, "device_map": "auto"}
    },
    {
        "id": "Trendyol-8B",
        "family": "trendyol",
        "base_path": "/users/ali.gunder/models/Trendyol-LLM-8B-T1",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_trendyol_8b_cefr",
        "load_kwargs": {"torch_dtype": torch.float16, "device_map": "auto"}
    },
    {
        "id": "EduTerm-14B",
        "family": "qwen",
        "base_path": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora",
        "load_kwargs": {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    }
]

EVAL_BENCHMARK = [
    {
        "id": "TTC-CS-1",
        "domain": "Bilgisayar Mühendisliği",
        "text": "Bugünkü dersimizde dengeli ikili arama ağaçları (Balanced Binary Search Trees) konusunu işleyeceğiz. Özellikle AVL ağacı ve kırmızı-siyah ağaç (Red-Black tree) veri yapılarında dönüş işlemleri (rotations) ile ağacın yüksekliğinin logaritmik karmaşıklık olan O(log n) seviyesinde tutulması arama ve ekleme hızını garanti eder.",
        "gold_terms": ["dengeli ikili arama ağaçları", "avl ağacı", "kırmızı-siyah ağaç", "dönüş işlemleri", "logaritmik karmaşıklık"]
    },
    {
        "id": "TTC-BIO-1",
        "domain": "Moleküler Biyoloji",
        "text": "DNA replikasyonu sırasında çift sarmal helikaz enzimi tarafından açılarak replikasyon çatalı oluşturulur. Kesintili zincirde sentezlenen Okazaki parçaları daha sonra DNA ligaz enzimi vasıtasıyla fosfodiester bağları kurularak birleştirilir.",
        "gold_terms": ["dna replikasyonu", "helikaz enzimi", "replikasyon çatalı", "okazaki parçaları", "dna ligaz enzimi", "fosfodiester bağları"]
    },
    {
        "id": "TTC-EE-1",
        "domain": "Elektrik-Elektronik",
        "text": "Devre analizinde düğüm gerilimleri analizi (nodal analysis) Kirchhoff Akım Yasası (KCL) denklemleri kurularak çözülür. İki temel düğüm arasında gerilim kaynağı bulunması durumunda süper düğüm (supernode) yöntemi uygulanır.",
        "gold_terms": ["düğüm gerilimleri analizi", "kirchhoff akım yasası", "kcl denklemleri", "süper düğüm"]
    },
    {
        "id": "TTC-PM-1",
        "domain": "Proje Yönetimi",
        "text": "Proje performans kontrolünde Kazanılmış Değer Analizi (Earned Value Analysis) kritik bir tekniktir. Planlanan Değer (PV) ile Gerçekleşen Maliyet (AC) kıyaslanarak Maliyet Performans İndeksi (CPI) ve Çizelge Performans İndeksi (SPI) türetilir.",
        "gold_terms": ["kazanılmış değer analizi", "planlanan değer", "gerçekleşen maliyet", "maliyet performans indeksi", "çizelge performans indeksi"]
    }
]

def format_prompt(model_family, lecture_text):
    if model_family == "kumru":
        return f"<|im_start|>user\nAşağıdaki ders anlatımındaki pedagojik ve teknik terimleri tam ve eksiksiz bir JSON listesi olarak çıkar:\nMetin: {lecture_text}\nYanıtını SADECE [\"terim1\", \"terim2\"] formatında ver.<|im_end|>\n<|im_start|>assistant\n["
    elif model_family == "trendyol":
        return f"<|im_start|>system\nSen uzman bir pedagojik terim çıkarıcı asistansın.<|im_end|>\n<|im_start|>user\nMetindeki teknik terimleri JSON listesi olarak ver:\n{lecture_text}<|im_end|>\n<|im_start|>assistant\n["
    else: # qwen
        return f"<|im_start|>system\nSen pedagojik terim analiz uzmanısın. Yalnızca geçerli bir JSON listesi döndür.<|im_end|>\n<|im_start|>user\nMetin: {lecture_text}\nTerimleri çıkar:<|im_end|>\n<|im_start|>assistant\n["

def clean_and_parse_json(text):
    text = text.strip()
    if not text.startswith("["):
        idx = text.find("[")
        if idx != -1:
            text = text[idx:]
        else:
            text = "[" + text
    idx_close = text.rfind("]")
    if idx_close != -1:
        text = text[:idx_close+1]
    else:
        text = text + "]"
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        import re
        matches = re.findall(r'"([^"]+)"', text)
        if matches:
            return [m.strip() for m in matches if m.strip()]
    return []

def normalize_term(t):
    return t.lower().replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c").strip()

def evaluate_metrics(extracted, gold):
    norm_gold = [normalize_term(g) for g in gold]
    norm_ext = [normalize_term(e) for e in extracted]
    tp = 0
    for e in norm_ext:
        if any(e == g or e in g or g in e for g in norm_gold):
            tp += 1
    fp = max(0, len(norm_ext) - tp)
    fn = max(0, len(norm_gold) - tp)
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, len(norm_gold))
    f1 = (2 * prec * rec) / max(1e-5, (prec + rec))
    return prec, rec, f1

def run_inference_sample(model, tokenizer, prompt, sample=False, temp=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    # Filter out kwargs like token_type_ids that cause ValueError in LLaMA / Qwen / Kumru
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    
    gen_kwargs = {
        "max_new_tokens": 128,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    if sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temp
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        out = model.generate(input_ids, **gen_kwargs)
    new_tokens = out[0][input_ids.shape[1]:]
    resp = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return clean_and_parse_json(resp)

def run_self_consistency(candidates_list, min_votes=2):
    # candidates_list is list of lists
    counts = Counter()
    norm_map = {}
    for cl in candidates_list:
        seen_in_sample = set()
        for t in cl:
            nt = normalize_term(t)
            if nt not in seen_in_sample:
                counts[nt] += 1
                seen_in_sample.add(nt)
                if nt not in norm_map:
                    norm_map[nt] = t
    
    agreed = [norm_map[nt] for nt, v in counts.items() if v >= min_votes]
    return agreed

def benchmark_model_ttc(cfg):
    logging.info(f"Loading {cfg['id']} for TTC Scaling...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["base_path"], trust_remote_code=True, **cfg["load_kwargs"])
    if os.path.exists(cfg["adapter_path"]):
        logging.info(f"  Loading LoRA adapter: {cfg['adapter_path']}")
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    model.eval()

    results = {
        "N=1_Greedy": {"precision": [], "recall": [], "f1": []},
        "N=3_SelfConsistency": {"precision": [], "recall": [], "f1": []},
        "N=5_SelfConsistency": {"precision": [], "recall": [], "f1": []}
    }

    start_time = time.time()

    for item in EVAL_BENCHMARK:
        prompt = format_prompt(cfg["family"], item["text"])
        
        # 1. N=1 Greedy
        ext_n1 = run_inference_sample(model, tokenizer, prompt, sample=False)
        p1, r1, f1_1 = evaluate_metrics(ext_n1, item["gold_terms"])
        results["N=1_Greedy"]["precision"].append(p1)
        results["N=1_Greedy"]["recall"].append(r1)
        results["N=1_Greedy"]["f1"].append(f1_1)

        # 2. Sample 5 responses
        samples = []
        for _ in range(5):
            s = run_inference_sample(model, tokenizer, prompt, sample=True, temp=0.7, top_p=0.9)
            samples.append(s)

        # N=3 majority vote (min 2 votes out of 3)
        ext_n3 = run_self_consistency(samples[:3], min_votes=2)
        p3, r3, f1_3 = evaluate_metrics(ext_n3, item["gold_terms"])
        results["N=3_SelfConsistency"]["precision"].append(p3)
        results["N=3_SelfConsistency"]["recall"].append(r3)
        results["N=3_SelfConsistency"]["f1"].append(f1_3)

        # N=5 majority vote (min 3 votes out of 5)
        ext_n5 = run_self_consistency(samples, min_votes=3)
        p5, r5, f1_5 = evaluate_metrics(ext_n5, item["gold_terms"])
        results["N=5_SelfConsistency"]["precision"].append(p5)
        results["N=5_SelfConsistency"]["recall"].append(r5)
        results["N=5_SelfConsistency"]["f1"].append(f1_5)

    del model
    del tokenizer
    torch.cuda.empty_cache()

    duration = time.time() - start_time
    summary = {"duration_sec": round(duration, 1)}
    for budget in ["N=1_Greedy", "N=3_SelfConsistency", "N=5_SelfConsistency"]:
        avg_p = sum(results[budget]["precision"]) / len(results[budget]["precision"])
        avg_r = sum(results[budget]["recall"]) / len(results[budget]["recall"])
        avg_f1 = sum(results[budget]["f1"]) / len(results[budget]["f1"])
        summary[budget] = {
            "precision_pct": round(avg_p * 100, 1),
            "recall_pct": round(avg_r * 100, 1),
            "f1_pct": round(avg_f1 * 100, 1)
        }
    logging.info(f"Summary for {cfg['id']}: {summary}")
    return summary

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/ttc_scaling_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_reports = {}

    for cfg in MODELS_CONFIG:
        res = benchmark_model_ttc(cfg)
        all_reports[cfg["id"]] = res

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    logging.info(f"TTC Scaling benchmark complete. Saved to {args.output}")

if __name__ == "__main__":
    main()

