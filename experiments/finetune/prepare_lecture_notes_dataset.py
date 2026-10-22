"""Lecture Notes Instruction Dataset Builder for Qwen 2.5 Fine-Tuning.

Ingests academic lecture notes and course materials:
- STEM / Engineering: Circuit Theory (TR-1), Molecular Biology (TR-2), Algorithms (TR-3), AI/ML (EN-1/4)
- Domain Lectures: Tourism, OHS/İSG, KVKK, Banking, Cybersecurity
- Custom PDF / TXT / Markdown course materials provided by the user.

Generates standardized Alpaca / ChatML JSONL instruction samples for Qwen 2.5:
  <|im_start|>system ... <|im_end|>
  <|im_start|>user ... <|im_end|>
  <|im_start|>assistant ... <|im_end|>
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LectureDataset")

ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = ROOT / "experiments" / "data" / "gold"
OUTPUT_DIR = ROOT / "experiments" / "finetune" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Curated high-yield academic lecture annotations with gold terms and CEFR B1 explanations
ACADEMIC_LECTURE_ANNOTATIONS: List[Dict[str, Any]] = [
    {
        "source": "EE-101 (Elektrik Devreleri & Kirchhoff)",
        "domain": "Elektrik ve Elektronik Mühendisliği",
        "lecture_text": (
            "Bu dersimizde kapalı bir devrede çevre gerilimlerini ve Kirchhoff Gerilim Yasasını (KVL) inceliyoruz. "
            "Devredeki bağımsız düğüm noktaları belirlendikten sonra düğüm gerilimleri analizi ile matrisel çözüm yapılır. "
            "Dirençler üzerinden geçen akım Ohm Kanunu ile gerilim farkına bağlanır."
        ),
        "terms": [
            {
                "term": "Kirchhoff Gerilim Yasası (KVL)",
                "canonical_id": "EE-01",
                "definition": "Kapalı bir elektrik devresi çevresi boyunca tüm potansiyel farklarının cebirsel toplamının sıfır olduğunu belirten temel devre kuralı.",
                "bloom_level": "Uygulama"
            },
            {
                "term": "Düğüm Gerilimleri Analizi",
                "canonical_id": "EE-02",
                "definition": "Devredeki düğüm noktalarının gerilimlerini temel değişken alarak Kirchhoff Akım Yasası yardımıyla devre denklemlerini çözme yöntemi.",
                "bloom_level": "Analiz"
            },
            {
                "term": "Ohm Kanunu",
                "canonical_id": "EE-03",
                "definition": "Bir iletkenin iki ucu arasındaki gerilim farkının, üzerinden geçen akım ve direnç ile doğru orantılı olduğunu ifade eden bağıntı (V = I * R).",
                "bloom_level": "Hatırlama"
            }
        ]
    },
    {
        "source": "BIO-101 (Moleküler Biyoloji & Genetik)",
        "domain": "Moleküler Biyoloji ve Tıp",
        "lecture_text": (
            "Hücre bölünmesinde DNA replikasyonu yarı korunumlu mekanizmayla gerçekleşir. "
            "Helikaz enzimi çift zinciri açarken DNA polimeraz yeni bazları sentezler. "
            "Ardından transkripsiyon aşamasında genetik kod haberci RNA (mRNA) molekülüne aktarılır."
        ),
        "terms": [
            {
                "term": "DNA Replikasyonu",
                "canonical_id": "BIO-01",
                "definition": "Hücre bölünmesi öncesinde genetik bilginin kopyalanarak iki özdeş DNA molekülü oluşturulması süreci.",
                "bloom_level": "Anlama"
            },
            {
                "term": "DNA Polimeraz",
                "canonical_id": "BIO-02",
                "definition": "Açılan DNA zincirine uygun nükleotitleri ekleyerek yeni DNA ipliğinin sentezini katalizleyen temel enzim.",
                "bloom_level": "Hatırlama"
            },
            {
                "term": "Transkripsiyon",
                "canonical_id": "BIO-03",
                "definition": "DNA üzerindeki genetik bilginin RNA polimeraz enzimi yardımıyla haberci RNA (mRNA) dizisine kopyalanması aşaması.",
                "bloom_level": "Analiz"
            }
        ]
    },
    {
        "source": "CS-101 (Veri Yapıları & Algoritmalar)",
        "domain": "Bilgisayar Mühendisliği",
        "lecture_text": (
            "Algoritmik karmaşıklığı analiz ederken en kötü durum davranışı Big-O gösterimi ile ifade edilir. "
            "İkili arama ağaçlarında (BST) arama maliyeti ağacın yüksekliğine bağlı olup dengesiz ağaçlarda O(n)'e düşerken "
            "kırmızı-siyah dengeli ağaçlarda logaritmik O(log n) süre garantilenir."
        ),
        "terms": [
            {
                "term": "Big-O Gösterimi",
                "canonical_id": "CS-01",
                "definition": "Bir algoritmanın girdi boyutu büyüdükçe gereken çalışma süresi veya bellek miktarının üst sınırını matematiksel olarak ifade eden karmaşıklık gösterimi.",
                "bloom_level": "Anlama"
            },
            {
                "term": "İkili Arama Ağacı (BST)",
                "canonical_id": "CS-02",
                "definition": "Her düğümün en fazla iki çocuğa sahip olduğu ve sol alt düğümlerin daha küçük, sağ alt düğümlerin daha büyük değerler barındırdığı veri yapısı.",
                "bloom_level": "Uygulama"
            },
            {
                "term": "Dengeli Arama Ağacı",
                "canonical_id": "CS-03",
                "definition": "Ekleme ve silme işlemlerinde ağaç yüksekliğini logaritmik sınırda tutarak O(log n) arama hızını garanti eden veri yapısı.",
                "bloom_level": "Analiz"
            }
        ]
    },
    {
        "source": "CS-229 (Machine Learning Concepts)",
        "domain": "Computer Science & AI",
        "lecture_text": (
            "In deep neural networks, backpropagation computes the gradient of the loss function with respect to model weights. "
            "To prevent overfitting on the training set, techniques such as dropout and L2 regularization are commonly applied."
        ),
        "terms": [
            {
                "term": "Backpropagation",
                "canonical_id": "AI-01",
                "definition": "An algorithm for computing the gradient of the loss function with respect to weights by applying the chain rule backward through layers.",
                "bloom_level": "Analyze"
            },
            {
                "term": "Overfitting",
                "canonical_id": "AI-02",
                "definition": "A modelling error where the machine learning model memorizes training noise and fails to generalize to unseen test data.",
                "bloom_level": "Understand"
            },
            {
                "term": "Regularization",
                "canonical_id": "AI-03",
                "definition": "Techniques such as weight decay or dropout added to prevent extreme weight values and improve model generalization.",
                "bloom_level": "Apply"
            }
        ]
    }
]


def create_instruction_sample(domain: str, lecture_text: str, terms: List[Dict[str, Any]]) -> Dict[str, Any]:
    instruction = (
        f"Aşağıdaki {domain} ders notu ve transkript metninden teknik terimleri tespit et. "
        "Her terim için kanonik ID, CEFR B1 düzeyinde sadeleştirilmiş anlaşılır bir açıklama ve Bloom taksonomi seviyesini ata. "
        "Sonucu yapılandırılmış geçerli JSON olarak ver."
    )
    output_obj = {
        "domain": domain,
        "terms": terms
    }
    return {
        "instruction": instruction,
        "input": lecture_text,
        "output": json.dumps(output_obj, ensure_ascii=False)
    }


def parse_custom_lecture_file(file_path: Path) -> List[str]:
    """Parse text, markdown, or PDF files into paragraph chunks."""
    if not file_path.exists():
        logger.warning("Custom file not found: %s", file_path)
        return []

    text = ""
    if file_path.suffix.lower() == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            for page in reader.pages:
                text += page.extract_text() or "" + "\n\n"
        except Exception as exc:
            logger.error("Could not parse PDF %s: %s", file_path, exc)
            return []
    else:
        text = file_path.read_text(encoding="utf-8", errors="ignore")

    # Split into meaningful paragraphs (between 100 and 800 chars)
    raw_paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
    return raw_paras[:20]


def build_lecture_dataset(extra_file: Path | None = None) -> Tuple[Path, Path]:
    """Build train and validation JSONL datasets for Qwen 2.5 instruction tuning."""
    from experiments.finetune.prepare_external_datasets import LeakageGuard
    guard = LeakageGuard()

    raw_candidates: List[Dict[str, Any]] = []

    # 1. Add curated academic lecture samples
    for item in ACADEMIC_LECTURE_ANNOTATIONS:
        sample = create_instruction_sample(item["domain"], item["lecture_text"], item["terms"])
        raw_candidates.append(sample)

    # 2. Add domain-specific samples from prepare_domain_datasets
    try:
        from experiments.finetune.prepare_domain_datasets import DOMAIN_SAMPLES
        raw_candidates.extend(DOMAIN_SAMPLES)
    except Exception as exc:
        logger.warning("Could not import domain samples: %s", exc)

    # 3. Add custom lecture notes if provided
    if extra_file and extra_file.exists():
        paras = parse_custom_lecture_file(extra_file)
        logger.info("Parsed %d paragraphs from custom lecture file %s", len(paras), extra_file.name)
        for i, p in enumerate(paras):
            raw_candidates.append({
                "instruction": "Aşağıdaki ders notu paragrafından teknik terimleri tespit et ve CEFR B1 düzeyinde açıkla.",
                "input": p,
                "output": json.dumps({
                    "domain": "Özel Ders Notu",
                    "terms": [
                        {
                            "term": "Ders Kavramı",
                            "canonical_id": f"CUSTOM-{i+1:02d}",
                            "definition": f"{p[:80]}... bağlamında incelenen temel ders konusu.",
                            "bloom_level": "Anlama"
                        }
                    ]
                }, ensure_ascii=False)
            })

    # Filter all candidate samples through LeakageGuard
    all_samples: List[Dict[str, Any]] = []
    dropped_count = 0
    for s in raw_candidates:
        blob = f"{s.get('instruction', '')} {s.get('input', '')} {s.get('output', '')}"
        if guard.text_is_eval(blob):
            dropped_count += 1
            continue
        try:
            out_obj = json.loads(s.get("output", "{}"))
            terms = out_obj.get("terms", [])
            if any(guard.term_is_eval_answer(t.get("term", "")) for t in terms):
                dropped_count += 1
                continue
        except Exception:
            pass
        all_samples.append(s)

    logger.info("LeakageGuard kept %d samples, dropped %d leaking items", len(all_samples), dropped_count)

    # Train / Val Split (85% train, 15% val)
    split_idx = max(1, int(len(all_samples) * 0.85))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:] if split_idx < len(all_samples) else all_samples[-1:]

    train_file = OUTPUT_DIR / "train_lecture_notes.jsonl"
    val_file = OUTPUT_DIR / "val_lecture_notes.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info("Generated %d train samples -> %s", len(train_samples), train_file)
    logger.info("Generated %d val samples -> %s", len(val_samples), val_file)
    return train_file, val_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lecture notes instruction dataset.")
    parser.add_argument("--extra-file", type=str, default=None, help="Optional custom lecture notes file (PDF/TXT/MD).")
    args = parser.parse_args()

    extra = Path(args.extra_file) if args.extra_file else None
    train_path, val_path = build_lecture_dataset(extra)
    print(f"Dataset successfully built:\nTrain: {train_path}\nValidation: {val_path}")


if __name__ == "__main__":
    main()
