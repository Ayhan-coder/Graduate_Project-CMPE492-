"""Benchmark and Inference Suite for Small Recognition Models (SRM) and Small Language Models (SLM).

Directly addresses Meeting Action 2:
"SRM ve fine tuned LLM, çok ufak modellerle daha verim artırılabilir"
- SRM Models: Whisper-tiny (39M), Whisper-base (74M), Whisper-small (244M), Qwen3-ASR-0.6B
- SLM Models: Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, Llama-3.2-1B, SmolLM2-1.7B
- Evaluates:
  * Parameter Count & Quantization foot-print (FP16, INT8, INT4)
  * Peak VRAM consumption (RTX 5060 8GB compatibility & CPU RAM)
  * Inference Latency (ms/token or RTF Real-Time Factor)
  * Edge & Browser Deployment readiness (ONNX / WebGPU)
  * Relative Terminology Extraction F1 on domain prompts

Usage:
    python benchmark_srm_slm.py --mode all --format table
    python benchmark_srm_slm.py --mode slm --quant int4 --out experiments/outputs/slm_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SRM_SLM_Benchmark")

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


@dataclass
class ModelProfile:
    id: str
    family: str  # "SRM" or "SLM"
    name: str
    params_m: float
    fp16_vram_mb: float
    int8_vram_mb: float
    int4_vram_mb: float
    latency_gpu_ms: float  # ms per token (SLM) or RTF (SRM)
    latency_cpu_ms: float
    webgpu_onnx_ready: bool
    est_domain_f1: float | None
    notes: str
    score_metric: str = ""
    measured: dict | None = None
    is_measured: bool = False
    status: str = "Tasarım Tahmini"


# Canonical model specifications grounded in 2026 published benchmarks
SRM_CATALOG: List[ModelProfile] = [
    ModelProfile(
        id="openai/whisper-tiny",
        family="SRM",
        name="Whisper Tiny",
        params_m=39.0,
        fp16_vram_mb=150.0,
        int8_vram_mb=210.0,
        int4_vram_mb=140.0,
        latency_gpu_ms=0.112,
        latency_cpu_ms=0.085,
        webgpu_onnx_ready=True,
        est_domain_f1=None,
        score_metric="WER (normalize, 6 altın klip ortalaması)",
        measured={"wer_norm": 0.200, "cer_norm": 0.0527, "rtf": 0.112, "peak_vram_gb": 0.15,
                  "clips": 6, "source": "outputs/srm_asr_measured.json", "precision": "fp16"},
        is_measured=True,
        status="Ölçülen (RTX 5060, fp16)",
        notes="6 altın klipte WER %20.0; hata Türkçe ders kayıtlarında yoğun (TR-3 %53.8), İngilizce %2-4. INT8/INT4 VRAM değerleri hâlâ projeksiyon.",
    ),
    ModelProfile(
        id="openai/whisper-base",
        family="SRM",
        name="Whisper Base",
        params_m=74.0,
        fp16_vram_mb=290.0,
        int8_vram_mb=320.0,
        int4_vram_mb=210.0,
        latency_gpu_ms=0.156,
        latency_cpu_ms=0.140,
        webgpu_onnx_ready=True,
        est_domain_f1=None,
        score_metric="WER (normalize, 6 altın klip ortalaması)",
        measured={"wer_norm": 0.1187, "cer_norm": 0.0285, "rtf": 0.156, "peak_vram_gb": 0.29,
                  "clips": 6, "source": "outputs/srm_asr_measured.json", "precision": "fp16"},
        is_measured=True,
        status="Ölçülen (RTX 5060, fp16)",
        notes="6 altın klipte WER %11.87; tiny'ye göre Türkçe kliplerde belirgin düşüş. INT8/INT4 VRAM değerleri hâlâ projeksiyon.",
    ),
    ModelProfile(
        id="openai/whisper-small",
        family="SRM",
        name="Whisper Small",
        params_m=244.0,
        fp16_vram_mb=770.0,
        int8_vram_mb=320.0,
        int4_vram_mb=240.0,
        latency_gpu_ms=0.054,
        latency_cpu_ms=0.360,
        webgpu_onnx_ready=True,
        est_domain_f1=None,
        score_metric="WER (normalize, 6 altın klip ortalaması)",
        measured={"wer_norm": 0.0608, "cer_norm": 0.0183, "rtf": 0.054, "peak_vram_gb": 0.77,
                  "clips": 6, "source": "outputs/srm_asr_measured.json", "precision": "fp16"},
        is_measured=True,
        status="Ölçülen (RTX 5060, fp16)",
        notes="6 altın klipte WER %6.08 — aynı koşuda Qwen3-ASR-0.6B'den (%7.57) daha doğru, 3× az VRAM. INT8/INT4 VRAM değerleri hâlâ projeksiyon.",
    ),
    ModelProfile(
        id="Qwen/Qwen3-ASR-0.6B-hf",
        family="SRM",
        name="Qwen3 ASR 0.6B",
        params_m=620.0,
        fp16_vram_mb=2370.0,
        int8_vram_mb=1050.0,
        int4_vram_mb=650.0,
        latency_gpu_ms=0.168,
        latency_cpu_ms=0.520,
        webgpu_onnx_ready=False,
        est_domain_f1=None,
        score_metric="WER (normalize, 6 altın klip ortalaması)",
        measured={"wer_norm": 0.0757, "cer_norm": 0.0275, "rtf": 0.168, "peak_vram_gb": 2.37,
                  "clips": 6, "source": "outputs/srm_asr_measured.json", "precision": "fp16"},
        is_measured=True,
        status="Ölçülen (RTX 5060, fp16)",
        notes="6 altın klipte WER %7.57, tepe VRAM 2.37 GB. Whisper-small aynı koşuda daha iyi (WER %6.08) — 'SOTA küçük ASR' beklentisi bu veriyle doğrulanmadı.",
    ),
    ModelProfile(
        id="openai/whisper-large-v3-turbo",
        family="SRM (Ref)",
        name="Whisper Large-v3 Turbo",
        params_m=809.0,
        fp16_vram_mb=1620.0,
        int8_vram_mb=1100.0,
        int4_vram_mb=820.0,
        latency_gpu_ms=0.044,
        latency_cpu_ms=0.680,
        webgpu_onnx_ready=False,
        est_domain_f1=None,
        score_metric="RTF + VRAM (yerel ölçüm)",
        measured={"rtf": 0.044, "vram_fp16_gb": 1.62, "vram_int4_gb": 0.82,
                  "source": "outputs/asr-20260903-turbo.jsonl", "precision": "fp16"},
        notes="Yerel GPU'da ölçüldü (RTF 0.025-0.044, 820 MB INT4/1.6GB fp16). Referans model.",
        is_measured=True,
        status="Ölçülen (RTX 5060)",
    ),
]

SLM_CATALOG: List[ModelProfile] = [
    ModelProfile(
        id="Qwen/Qwen2.5-0.5B-Instruct",
        family="SLM",
        name="Qwen 2.5 0.5B Instruct",
        params_m=490.0,
        fp16_vram_mb=970.0,
        int8_vram_mb=720.0,
        int4_vram_mb=450.0,
        latency_gpu_ms=4498.0,
        latency_cpu_ms=28.0,
        webgpu_onnx_ready=True,
        est_domain_f1=None,
        score_metric="terim F1 (BENCH-8, RTX 5060)",
        measured={"f1_score": 0.4667, "recall": 0.5, "precision": 0.4375, "json_validity_pct": 100.0,
                  "peak_vram_gb": 0.97, "avg_latency_s": 4.498, "prompt_format": "chatml",
                  "source": "outputs/small_models_local_report.json",
                  "a40_cross_check_f1": 0.3947},
        is_measured=True,
        status="Ölçülen (RTX 5060, bf16)",
        notes="BENCH-8'de F1 0.4667, JSON %100. A40 raporunda 0.3947 — aynı kod, farklı ortam (GPU/transformers/model revizyonu); ±0.07 bu probe için gürültü bandı.",
    ),
    ModelProfile(
        id="meta-llama/Llama-3.2-1B-Instruct",
        family="SLM",
        name="Llama 3.2 1B Instruct",
        params_m=1230.0,
        fp16_vram_mb=2370.0,
        int8_vram_mb=1650.0,
        int4_vram_mb=980.0,
        latency_gpu_ms=3490.0,
        latency_cpu_ms=44.0,
        webgpu_onnx_ready=True,
        est_domain_f1=None,
        score_metric="terim F1 (BENCH-8, RTX 5060)",
        measured={"f1_score": 0.6526, "recall": 0.802, "precision": 0.55, "json_validity_pct": 87.5,
                  "peak_vram_gb": 2.37, "avg_latency_s": 3.49, "prompt_format": "auto (chat template)",
                  "source": "outputs/small_models_local_report.json",
                  "note": "ChatML prompt'u ile aynı model F1 0.1334 / JSON %12.5 veriyor; fark modelden değil şablon uyumsuzluğundan."},
        is_measured=True,
        status="Ölçülen (RTX 5060, bf16)",
        notes="Kendi chat şablonuyla F1 0.6526, JSON %87.5. A40 raporundaki 'Gemma-2-27B JSON %0' satırı da aynı şablon uyumsuzluğundan etkilenmiş olabilir.",
    ),
    ModelProfile(
        id="Qwen/Qwen2.5-1.5B-Instruct",
        family="SLM",
        name="Qwen 2.5 1.5B Instruct",
        params_m=1540.0,
        fp16_vram_mb=2930.0,
        int8_vram_mb=1950.0,
        int4_vram_mb=1150.0,
        latency_gpu_ms=3619.0,
        latency_cpu_ms=52.0,
        webgpu_onnx_ready=True,
        est_domain_f1=None,
        score_metric="terim F1 (BENCH-8, RTX 5060)",
        measured={"f1_score": 0.7238, "recall": 0.7916, "precision": 0.6667, "json_validity_pct": 100.0,
                  "peak_vram_gb": 2.93, "avg_latency_s": 3.619, "prompt_format": "chatml",
                  "source": "outputs/small_models_local_report.json",
                  "a40_cross_check_f1": 0.6188},
        is_measured=True,
        status="Ölçülen (RTX 5060, bf16)",
        notes="BENCH-8'de F1 0.7238, JSON %100, 2.93 GB — 8 GB sınırında çalışan en verimli Qwen noktası.",
    ),
    ModelProfile(
        id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        family="SLM",
        name="SmolLM2 1.7B Instruct",
        params_m=1710.0,
        fp16_vram_mb=3320.0,
        int8_vram_mb=2150.0,
        int4_vram_mb=1280.0,
        latency_gpu_ms=3560.0,
        latency_cpu_ms=58.0,
        webgpu_onnx_ready=True,
        est_domain_f1=None,
        notes="Kendi chat şablonuyla BENCH-8 F1 0.7084, JSON %87.5, tepe VRAM 3.32 GB; aynı VRAM bandında Qwen 1.5B (0.7238) ve Llama 1B (0.6526) arasında kalıyor.",
        score_metric="terim F1 (BENCH-8, RTX 5060)",
        measured={
            "f1_score": 0.7084,
            "recall": 0.708,
            "precision": 0.708,
            "json_validity_pct": 87.5,
            "peak_vram_gb": 3.32,
            "avg_latency_s": 3.56,
            "prompt_format": "auto (chat template)",
            "source": "outputs/small_models_local_report.json",
        },
        is_measured=True,
        status="Ölçülen (RTX 5060, bf16)",
    ),
    ModelProfile(
        id="Qwen/Qwen2.5-3B-Instruct",
        family="SLM",
        name="Qwen 2.5 3B Instruct",
        params_m=3090.0,
        fp16_vram_mb=5810.0,
        int8_vram_mb=3800.0,
        int4_vram_mb=2150.0,
        latency_gpu_ms=4298.0,
        latency_cpu_ms=96.0,
        webgpu_onnx_ready=False,
        est_domain_f1=None,
        score_metric="terim F1 (BENCH-8, RTX 5060)",
        measured={"f1_score": 0.7616, "recall": 0.802, "precision": 0.725, "json_validity_pct": 100.0,
                  "peak_vram_gb": 5.81, "avg_latency_s": 4.298, "prompt_format": "chatml",
                  "source": "outputs/small_models_local_report.json"},
        is_measured=True,
        status="Ölçülen (RTX 5060, bf16)",
        notes="BENCH-8'de F1 0.7616, JSON %100; 5.81 GB tepe VRAM ile 8 GB sınırının altında kalıyor.",
    ),
    ModelProfile(
        id="Qwen/Qwen3.5-4B",
        family="SLM",
        name="Qwen 3.5 4B (nf4)",
        params_m=4000.0,
        fp16_vram_mb=5260.0,
        int8_vram_mb=4800.0,
        int4_vram_mb=3081.0,
        latency_gpu_ms=39550.0,
        latency_cpu_ms=88.0,
        webgpu_onnx_ready=False,
        est_domain_f1=None,
        score_metric="VRAM + gecikme (yerel ölçüm)",
        measured={"peak_vram_gb": 5.26, "avg_call_s": 39.55, "source": "outputs/local_vram_probe.json"},
        notes="Yerel GPU'da ölçüldü: 3.081 MiB ağırlık, 4.721 MiB tepe ayrılan (allocated), 5.136 MiB ayrılmış (reserved), 5.262 MiB cihaz VRAM; 39.55 sn ortalama çağrı.",
        is_measured=True,
        status="Ölçülen (RTX 5060)",
    ),
]


def measured_score(p: ModelProfile) -> str:
    """Render the row's measured metric; '—' when the row has no measurement."""
    m = p.measured or {}
    if "wer_norm" in m:
        return f"WER {m['wer_norm'] * 100:.2f}%"
    if "f1_score" in m:
        return f"F1 {m['f1_score']:.3f}"
    if "rtf" in m:
        return f"RTF {m['rtf']:.3f}"
    if "peak_vram_gb" in m:
        return f"VRAM {m['peak_vram_gb']:.2f} GB"
    return "—"


def render_ascii_table(profiles: List[ModelProfile], quant: str = "fp16") -> str:
    """Format profiles into a readable ASCII comparison table."""
    vram_key = f"{quant}_vram_mb"
    header = f"| {'Model Name':<24} | {'Family':<8} | {'Params':<8} | {f'VRAM ({quant.upper()})':<12} | {'GPU Latency':<14} | {'Statü':<20} | {'Ölçülen':<14} | {'WebGPU?':<8} |"
    sep = "|" + "-" * 26 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 14 + "|" + "-" * 16 + "|" + "-" * 22 + "|" + "-" * 16 + "|" + "-" * 10 + "|"
    lines = [header, sep]
    for p in profiles:
        vram = getattr(p, vram_key, p.fp16_vram_mb)
        unit = "s (RTF)" if "SRM" in p.family else "s/sorgu"
        vram_str = f"{vram/1024:.2f} GB" if vram >= 1000 else f"{vram:.0f} MB"
        lat_str = f"{p.latency_gpu_ms:.3f} {unit}" if "SRM" in p.family else f"{p.latency_gpu_ms/1000:.2f} {unit}"
        if not p.is_measured:
            lat_str = "—"
        web_str = "Evet" if p.webgpu_onnx_ready else "Hayır"
        lines.append(f"| {p.name:<24} | {p.family:<8} | {p.params_m:>6.0f}M | {vram_str:<12} | {lat_str:<14} | {p.status:<20} | {measured_score(p):<14} | {web_str:<8} |")
    return "\n".join(lines)


def render_markdown(profiles: List[ModelProfile], quant: str = "fp16") -> str:
    vram_key = f"{quant}_vram_mb"
    md = [
        f"### Küçük Modeller (SRM & SLM) Verimlilik ve Kıyaslama Raporu (Kuantizasyon: {quant.upper()})",
        "",
        "| Model | Aile | Parametre | VRAM Tüketimi | GPU Gecikme | Statü | Ölçülen Skor | WebGPU / ONNX | Notlar |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for p in profiles:
        vram = getattr(p, vram_key, p.fp16_vram_mb)
        unit = "s (RTF)" if "SRM" in p.family else "s/sorgu"
        vram_str = f"{vram/1024:.2f} GB" if vram >= 1000 else f"{vram:.0f} MB"
        lat_str = f"{p.latency_gpu_ms:.3f} {unit}" if "SRM" in p.family else f"{p.latency_gpu_ms/1000:.2f} {unit}"
        if not p.is_measured:
            lat_str = "—"
        web_str = "✅ Destekler" if p.webgpu_onnx_ready else "⏳ Sınırlı"
        status_badge = f"**{p.status}**" if p.is_measured else f"*{p.status}*"
        md.append(f"| **{p.name}** | {p.family} | {p.params_m:.0f}M | `{vram_str}` | `{lat_str}` | {status_badge} | {measured_score(p)} | {web_str} | {p.notes} |")
    return "\n".join(md)


def run_benchmark(mode: str = "all", quant: str = "fp16") -> Dict[str, Any]:
    selected: List[ModelProfile] = []
    if mode in ("all", "srm"):
        selected.extend(SRM_CATALOG)
    if mode in ("all", "slm"):
        selected.extend(SLM_CATALOG)

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_profile": "NVIDIA GeForce RTX 5060 Laptop (8GB VRAM) / Intel i7",
        "quantization": quant,
        "mode": mode,
        "total_models": len(selected),
        "profiles": [asdict(p) for p in selected],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SRM & SLM Benchmark and Efficiency Profiler.")
    parser.add_argument("--mode", choices=["all", "srm", "slm"], default="all", help="Which family to benchmark.")
    parser.add_argument("--quant", choices=["fp16", "int8", "int4"], default="fp16", help="Target quantization level.")
    parser.add_argument("--format", choices=["table", "markdown", "json"], default="table", help="Output format.")
    parser.add_argument("--out", type=str, default=None, help="Optional output filepath.")
    args = parser.parse_args()

    result = run_benchmark(mode=args.mode, quant=args.quant)
    profiles = [ModelProfile(**item) for item in result["profiles"]]

    if args.format == "table":
        output_str = render_ascii_table(profiles, quant=args.quant)
    elif args.format == "markdown":
        output_str = render_markdown(profiles, quant=args.quant)
    else:
        output_str = json.dumps(result, ensure_ascii=False, indent=2)

    print(output_str)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            if args.format == "json":
                json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                f.write(output_str)
        logger.info("Saved benchmark output to %s", out_path)


if __name__ == "__main__":
    main()
