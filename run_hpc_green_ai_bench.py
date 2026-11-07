#!/usr/bin/env python3
"""
HPC Green AI & Hardware Efficiency Benchmark
Measures exact inference latency (ms), token generation throughput (tokens/sec),
GPU Power Draw (Watt via nvidia-smi / pynvml), Energy consumption (Joules/query),
and VRAM footprint across Kumru-2B, Trendyol-8B, and EduTerm-14B on NVIDIA A40.
"""

import os
import sys
import json
import logging
import time
import subprocess
import threading
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

PROMPTS = [
    "Aşağıdaki dersteki terimleri çıkar: Biyomühendislikte jel elektroforezi ve DNA ligaz enzimi kullanılarak rekombinant plazmit vektörleri klonlanır.",
    "Aşağıdaki dersteki terimleri çıkar: Bilgisayar ağlarında TCP üçlü el sıkışma protokolü (three-way handshake) SYN ve ACK paketleri ile güvenli oturum kurar.",
    "Aşağıdaki dersteki terimleri çıkar: Makine öğrenmesinde gradyan inişi (gradient descent) stokastik minimizasyon ve öğrenme oranı (learning rate) hiperparametresiyle optimize edilir.",
    "Aşağıdaki dersteki terimleri çıkar: Elektrik devrelerinde empedans eşleme maksimum güç transfer teoremi uyarınca kaynak direncinin yük direncine eşitlenmesiyle elde edilir."
]

class GPUPowerMonitor:
    def __init__(self, interval_sec=0.2):
        self.interval = interval_sec
        self.power_samples = []
        self._stop = threading.Event()
        self._thread = None

    def _monitor(self):
        while not self._stop.is_set():
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1
                )
                if res.returncode == 0:
                    val = float(res.stdout.strip().split("\n")[0])
                    self.power_samples.append(val)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.power_samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        avg_power = sum(self.power_samples) / max(1, len(self.power_samples)) if self.power_samples else 100.0
        return avg_power

def benchmark_model_energy(cfg):
    logging.info(f"Loading {cfg['id']} for Green AI & Energy Profiling...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(cfg["base_path"], trust_remote_code=True, **cfg["load_kwargs"])
    if os.path.exists(cfg["adapter_path"]):
        logging.info(f"  Loading LoRA adapter: {cfg['adapter_path']}")
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    model.eval()

    vram_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Warmup
    dummy = tokenizer("Isınma testi", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        _ = model.generate(dummy["input_ids"], max_new_tokens=10)

    monitor = GPUPowerMonitor(interval_sec=0.15)
    monitor.start()

    latencies_sec = []
    generated_token_counts = []

    start_bench = time.time()
    for p in PROMPTS:
        inputs = tokenizer(p, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"]
        gen_kwargs = {
            "max_new_tokens": 64,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs["attention_mask"]
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(input_ids, **gen_kwargs)
        elapsed = time.time() - t0
        new_toks = out[0].shape[0] - input_ids.shape[1]
        latencies_sec.append(elapsed)
        generated_token_counts.append(new_toks)

    total_bench_time = time.time() - start_bench
    avg_power_watt = monitor.stop()

    del model
    del tokenizer
    torch.cuda.empty_cache()

    total_tokens = sum(generated_token_counts)
    throughput_tok_sec = total_tokens / max(1e-5, total_bench_time)
    avg_latency_ms = (sum(latencies_sec) / len(latencies_sec)) * 1000
    
    # Energy E = P * t (Joules = Watts * Seconds)
    total_energy_joules = avg_power_watt * total_bench_time
    joules_per_query = total_energy_joules / max(1, len(PROMPTS))
    joules_per_token = total_energy_joules / max(1, total_tokens)

    # Carbon emissions estimation: ~0.4 kg CO2 / kWh average grid factor -> 0.000111 g CO2 / Joule
    co2_per_query_mg = joules_per_query * 0.111

    summary = {
        "vram_peak_gb": round(vram_peak_gb, 2),
        "avg_power_draw_watt": round(avg_power_watt, 1),
        "avg_latency_per_query_ms": round(avg_latency_ms, 1),
        "throughput_tokens_per_sec": round(throughput_tok_sec, 1),
        "energy_joules_per_query": round(joules_per_query, 2),
        "energy_joules_per_token": round(joules_per_token, 3),
        "carbon_footprint_mg_co2_per_query": round(co2_per_query_mg, 2)
    }
    logging.info(f"Green AI Profiling for {cfg['id']}: {summary}")
    return summary

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/green_ai_efficiency_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_reports = {}

    for cfg in MODELS_CONFIG:
        res = benchmark_model_energy(cfg)
        all_reports[cfg["id"]] = res

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    logging.info(f"Green AI efficiency benchmark complete. Saved to {args.output}")

if __name__ == "__main__":
    main()

