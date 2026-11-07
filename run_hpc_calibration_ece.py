#!/usr/bin/env python3
"""
HPC Confidence Calibration, Selective Classification & Expected Calibration Error (ECE) Benchmark
Evaluates token log-likelihood confidence vs exact match correctness across models.
Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
and Selective Classification Risk-Coverage Curves (abstention thresholding).
"""

import os
import sys
import json
import logging
import time
import math
import torch
import torch.nn.functional as F
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
        "id": "EduTerm-14B",
        "family": "qwen",
        "base_path": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora",
        "load_kwargs": {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    }
]

# Diverse diagnostic questions across academic disciplines
CALIBRATION_DATASET = [
    {
        "prompt": "Soru: Bayes teoreminde öncül olasılık ile kanıtın çarpımının normalizasyon sabitine bölünmesi neyi verir?\nA) Sonsal olasılık (Posterior)\nB) Olabilirlik (Likelihood)\nC) Marjinal dağılım\nD) Koşullu varyans\nCevap:",
        "gold_answer": "A"
    },
    {
        "prompt": "Soru: İşletim sistemlerinde deadlock durumunun dört Coffman koşulundan hangisi kaynakların tek bir süreç tarafından tutulduğunu ifade eder?\nA) Karşılıklı dışlama (Mutual exclusion)\nB) Tut ve bekle (Hold and wait)\nC) Dairesel bekleme\nD) Yoksun bırakma\nCevap:",
        "gold_answer": "A"
    },
    {
        "prompt": "Soru: Kuantum mekaniğinde belirsizlik ilkesi hangi iki operatör arasındaki komütatörün sıfırdan farklı olmasına dayanır?\nA) Konum ve momentum\nB) Enerji ve kütle\nC) Spin ve yük\nD) Hız ve dalgaboyu\nCevap:",
        "gold_answer": "A"
    },
    {
        "prompt": "Soru: İktisatta Phillips eğrisi kısa dönemde hangi iki makroekonomik değişken arasındaki ters yönlü ilişkiyi gösterir?\nA) Enflasyon ve işsizlik\nB) Faiz ve gayrisafi yurtiçi hasıla\nC) Döviz kuru ve cari açık\nD) Para arzı ve tüketim\nCevap:",
        "gold_answer": "A"
    },
    {
        "prompt": "Soru: Derin öğrenmede yok olan gradyan (vanishing gradient) problemini hafifletmek için Sigmoid yerine önerilen aktivasyon fonksiyonu hangisidir?\nA) ReLU\nB) Step fonksiyonu\nC) Softsign\nD) Sinusoid\nCevap:",
        "gold_answer": "A"
    },
    {
        "prompt": "Soru: Biyokimyada enzimlerin substrat bağlama bölgesinin yapısını açıklayan Koshland modeli hangisidir?\nA) Uyarılmış uyum (Induced fit)\nB) Anahtar kilit\nC) Allosterik inhibisyon\nD) Kooperatif denatürasyon\nCevap:",
        "gold_answer": "A"
    },
    {
        "prompt": "Soru: Veri yapılarında dengeli ikili arama ağacı (AVL tree) için ekleme ve silme işlemlerinin en kötü durum zaman karmaşıklığı nedir?\nA) O(log n)\nB) O(n)\nC) O(1)\nD) O(n log n)\nCevap:",
        "gold_answer": "A"
    },
    {
        "prompt": "Soru: Mikroişlemcilerde dal tahminleme (branch prediction) hatası oluştuğunda meydana gelen gecikmeye ne ad verilir?\nA) Pipeline stall (Borucu duraklaması)\nB) Page fault\nC) Cache miss\nD) Bus arbitration\nCevap:",
        "gold_answer": "A"
    }
]

def format_mcq_prompt(family, question_text):
    if family == "kumru":
        return f"<|im_start|>user\nAşağıdaki çoktan seçmeli soruyu analiz et ve doğru şıkkı (A, B, C veya D) ver:\n{question_text}<|im_end|>\n<|im_start|>assistant\nCevap: "
    else:
        return f"<|im_start|>system\nSen akademik bir sınav değerlendirme asistanısın. Soruya sadece şık harfi ile (A, B, C veya D) cevap ver.<|im_end|>\n<|im_start|>user\n{question_text}<|im_end|>\n<|im_start|>assistant\nCevap: "

def compute_ece(confidences, accuracies, num_bins=5):
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    binned evenly between 0.0 and 1.0.
    """
    bin_size = 1.0 / num_bins
    ece = 0.0
    mce = 0.0
    total_samples = len(confidences)

    bins_data = []

    for i in range(num_bins):
        bin_lower = i * bin_size
        bin_upper = (i + 1) * bin_size

        # Find items in bin
        bin_indices = [
            idx for idx, conf in enumerate(confidences)
            if (bin_lower <= conf < bin_upper) or (i == num_bins - 1 and bin_lower <= conf <= bin_upper)
        ]

        if len(bin_indices) > 0:
            bin_acc = sum([accuracies[idx] for idx in bin_indices]) / len(bin_indices)
            bin_conf = sum([confidences[idx] for idx in bin_indices]) / len(bin_indices)
            bin_error = abs(bin_acc - bin_conf)

            ece += (len(bin_indices) / total_samples) * bin_error
            mce = max(mce, bin_error)

            bins_data.append({
                "bin_range": f"{bin_lower:.2f}-{bin_upper:.2f}",
                "count": len(bin_indices),
                "avg_confidence": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4),
                "calibration_gap": round(bin_error, 4)
            })
        else:
            bins_data.append({
                "bin_range": f"{bin_lower:.2f}-{bin_upper:.2f}",
                "count": 0,
                "avg_confidence": 0.0,
                "accuracy": 0.0,
                "calibration_gap": 0.0
            })

    return ece, mce, bins_data

def evaluate_calibration(cfg):
    logging.info(f"--- Running Calibration & ECE Benchmark on {cfg['id']} ---")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["base_path"], trust_remote_code=True, **cfg["load_kwargs"])
    if os.path.exists(cfg["adapter_path"]):
        logging.info(f"  Loading LoRA adapter: {cfg['adapter_path']}")
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    model.eval()

    # Pre-tokenize choice tokens A, B, C, D
    choice_tokens = {c: tokenizer.encode(c, add_special_tokens=False)[-1] for c in ["A", "B", "C", "D"]}

    confidences = []
    accuracies = []
    selective_results = []

    start_time = time.time()

    for item in CALIBRATION_DATASET:
        prompt = format_mcq_prompt(cfg["family"], item["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        model_inputs = {
            "input_ids": inputs["input_ids"]
        }
        if "attention_mask" in inputs:
            model_inputs["attention_mask"] = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(**model_inputs)
            # Logits of next token
            next_token_logits = outputs.logits[:, -1, :]
            
            # Extract choice logits
            choice_logits = torch.tensor([next_token_logits[0, choice_tokens[c]].item() for c in ["A", "B", "C", "D"]], device=DEVICE)
            probs = F.softmax(choice_logits, dim=-1)

            pred_idx = torch.argmax(probs).item()
            pred_choice = ["A", "B", "C", "D"][pred_idx]
            confidence = probs[pred_idx].item()

            is_correct = 1.0 if pred_choice == item["gold_answer"] else 0.0

            confidences.append(confidence)
            accuracies.append(is_correct)

    duration = time.time() - start_time
    ece, mce, bins_data = compute_ece(confidences, accuracies, num_bins=5)
    overall_acc = (sum(accuracies) / len(accuracies)) * 100.0
    avg_conf = (sum(confidences) / len(confidences)) * 100.0

    # Selective classification coverage at confidence thresholds
    # e.g., threshold 0.70, 0.85
    coverage_70 = sum([1 for c in confidences if c >= 0.70]) / len(confidences) * 100.0
    selective_acc_70 = (
        sum([acc for c, acc in zip(confidences, accuracies) if c >= 0.70]) /
        max(1, sum([1 for c in confidences if c >= 0.70]))
    ) * 100.0

    summary = {
        "model_id": cfg["id"],
        "overall_accuracy_pct": round(overall_acc, 2),
        "mean_confidence_pct": round(avg_conf, 2),
        "expected_calibration_error_ece": round(ece, 4),
        "maximum_calibration_error_mce": round(mce, 4),
        "selective_coverage_tau70_pct": round(coverage_70, 2),
        "selective_accuracy_tau70_pct": round(selective_acc_70, 2),
        "bins": bins_data,
        "duration_sec": round(duration, 2)
    }

    logging.info(f"{cfg['id']} Calibration: Acc={overall_acc:.1f}%, MeanConf={avg_conf:.1f}%, ECE={ece:.4f}")

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return summary

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/calibration_ece_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_reports = {}

    for cfg in MODELS_CONFIG:
        res = evaluate_calibration(cfg)
        all_reports[cfg["id"]] = res

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    logging.info(f"Calibration benchmark complete. Report saved to {args.output}")

if __name__ == "__main__":
    main()

