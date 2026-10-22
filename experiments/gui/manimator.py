"""Manimator Integration Module for EduTerm AI Studio.

Turns the semantic layer into dynamic video generation:
  [Lecture / PDF Context]
        │
        ▼
   EVREN cloud LLM  (fallback: local Qwen 3.5, then domain heuristics)
   ├── Terminology Extraction
   ├── CEFR B1 Pedagogical Explanations
   └── Scene-by-Scene Video Narration Script + Manim Animation Code
        │
        ├── TTS Synthesis (Edge-TTS Neural Voice)
        │       │
        └── Manim Community Rendering Engine + FFmpeg
                │
                ▼
        Narrated Explainer Video (.mp4) for Confused Students

Supports:
1. "Takıldığım Yeri Öğret": 30-90s explainer video tailored to the exact moment
   and concept the student struggled with in the lecture.
2. "PDF to Narrated Video": Academic paper / slides PDF to Manim animation
   or Figure Walkthrough video.
3. Self-healing Manim code execution with automatic syntax/attribute repairs.
"""

from __future__ import annotations

import asyncio
import ast
import gc
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gui.manim_scene_builders import generate_modular_manim_code

logger = logging.getLogger("eduterm.manimator")

ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / "gui" / "static" / "generated_videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

TEMP_DIR = ROOT / "gui" / "temp" / "manimator"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Default Edge-TTS Voices
DEFAULT_VOICE_TR = "tr-TR-AhmetNeural"
DEFAULT_VOICE_TR_FEMALE = "tr-TR-EmelNeural"
DEFAULT_VOICE_EN = "en-US-GuyNeural"
DEFAULT_VOICE_EN_FEMALE = "en-US-JennyNeural"

# The Edge voice endpoint rate-limits this machine to roughly one request per minute,
# so a rejection is retried once and then handed to a second online Turkish voice
# rather than to an offline voice that cannot pronounce the language.
_EDGE_RETRY_WAS_S: Tuple[float, ...] = (0.0,)
_GOOGLE_TTS_URL = "https://translate.google.com/translate_tts"
_GOOGLE_TTS_CHUNK_CHARS = 160
_GOOGLE_TTS_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# EVREN is the default script/animation author for Layer 3; the raw platform id
# is what /v1/chat/completions expects.
DEFAULT_EVREN_MODEL = "qwen3.8-flash-next"

# Render tiers: Manim raster settings plus the x264 settings used when the
# narration is muxed back onto the animation.
VIDEO_QUALITY_TIERS: Dict[str, Dict[str, Any]] = {
    "draft":    {"resolution": (854, 480),   "fps": 15.0, "crf": 23, "preset": "veryfast"},
    "standard": {"resolution": (1920, 1080), "fps": 30.0, "crf": 20, "preset": "fast"},
    "high":     {"resolution": (1920, 1080), "fps": 30.0, "crf": 18, "preset": "medium"},
    "ultra":    {"resolution": (1920, 1080), "fps": 60.0, "crf": 17, "preset": "medium"},
}
DEFAULT_VIDEO_QUALITY = "standard"

_RENDER_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 1. Local Qwen 3.5 Prompt Engineering & Generation
# ---------------------------------------------------------------------------

MANIM_PROMPT_TEMPLATE = """<|im_start|>system
Sen uzman bir pedagojik video senaristisin. Gorevin, derste takildigi teknik konuyu B1 duzeyinde kisa bir anlatima donusturmektir.

Kurallar:
1. Terimler ve anlatım kesinlikle CEFR B1 düzeyinde, anlaşılır, somut ve sade olmalıdır.
2. Yalnizca verilen ders baglamiyla desteklenen bilgileri soyle. Baska konuya ait kavram veya genel tavsiye ekleme.
3. Baglam yetersizse bunu acikca soyle; kesin olmayan tanim uydurma. Baglamla uyusan bir benzetme yoksa bos birak.
4. Her sahne için konuya en uygun görsel sahne tipini seç:
   - "dag_concept": Kavramlar arası önkoşul veya hiyerarşi ilişkisi (nodes)
   - "process_flow": Adım adım sıralı süreç, algoritma veya boru hattı (steps)
   - "comparison_table": İki yaklaşımın veya durumun karşılaştırılması (left_title, right_title, left_points, right_points)
   - "equation_step": Matematiksel formül, kural veya fonksiyon analizi (formula, variables)
   - "concept_card": Somut benzetme, tanım ve pedagojik altın çıkarım kartı (takeaway)
5. Çıktıyı SADECE geçerli bir JSON objesi olarak ver. manim_code alanını boş bırak.
<|im_end|>
<|im_start|>user
Öğrencinin Takıldığı Konu / Terim: {TOPIC}
Dersin İlgili Bölümündeki Anlatım: "{CONTEXT}"
Hedef Dil: {LANG}
Hedef Süre: {DURATION_S} saniye

Lütfen aşağıdaki JSON şemasına birebir uyarak yanıt ver:
{{
  "title": "Konu Başlığı",
  "target_concept": "Temel Kavram",
  "pedagogical_hook": "Öğrencinin dikkatini çekecek somut benzetme",
  "terms": [
    {{"term": "Terim Adı", "explanation": "B1 düzeyinde sade tanım"}}
  ],
  "narration_script": "Videonun baştan sona eksiksiz seslendirme metni...",
  "scenes": [
    {{
      "id": 1,
      "scene_type": "concept_card",
      "title": "Giriş ve Problem",
      "narration": "Bu sahnenin anlatımı...",
      "duration_s": 12.0,
      "takeaway": "Ders bağlamına dayalı altın çıkarım ve kilit kural"
    }},
    {{
      "id": 2,
      "scene_type": "process_flow",
      "title": "İşleyiş Aşamaları",
      "steps": [
        {{"title": "1. Girdi", "desc": "Aşama açıklaması"}},
        {{"title": "2. İşlem", "desc": "Aşama açıklaması"}},
        {{"title": "3. Çıktı", "desc": "Aşama açıklaması"}}
      ],
      "narration": "Adımların anlatımı...",
      "duration_s": 18.0
    }},
    {{
      "id": 3,
      "scene_type": "comparison_table",
      "title": "Karşılaştırma ve Özet",
      "left_title": "Yaklaşım A",
      "right_title": "Yaklaşım B",
      "left_points": ["Özellik 1", "Özellik 2"],
      "right_points": ["Özellik 1", "Özellik 2"],
      "narration": "Karşılaştırmalı özet ve altın kural...",
      "duration_s": 15.0,
      "takeaway": "Sınav ve uygulama altın ipucu"
    }}
  ],
  "manim_code": ""
}}
<|im_end|>
<|im_start|>assistant
"""


def _clean_json_output(raw_text: str) -> dict:
    """Robustly extract JSON dictionary from raw model text."""
    text = raw_text.strip()
    # Strip thinking tags if Qwen3.5 think mode was active
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    # Strip markdown code blocks
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    # Search for outermost JSON braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try cleaning trailing commas or escaping backslashes in code
            cleaned = re.sub(r",\s*([\]}])", r"\1", candidate)
            try:
                return json.loads(cleaned)
            except Exception:
                pass

    raise ValueError("Valid JSON structure not found in model output")


EVREN_MANIM_SYSTEM_PROMPT = """Sen uzman bir pedagojik video senaristisin.
Gorevin: bir ogrencinin derste takildigi teknik konuyu, CEFR B1 duzeyinde, kisa ve baglama sadik bir anlatima donusturmek.

Kurallar:
1. Anlatim secilen dilde (Türkçe veya English), somut, sade ve B1 duzeyinde olacak.
2. Yalnizca verilen ders baglamiyla desteklenen bilgileri soyle. Alakasiz genel tavsiye, baska konuya ait kavram, uydurma ornek veya baglamda olmayan bilgi ekleme.
3. Baglam yetersizse bunu acikca soyle; kesin olmayan tanim uydurma. Teknik terimi bu kesitteki anlami ile acikla.
4. Benzetme kullanacaksan teknik fikri dogru korusun ve baglamla uyussun; aksi halde bos birak.
5. Her sahne icin konuya en uygun gorsel sahne tipini ("scene_type") sec ve o sahneye ozel gorsel veri alanlarini (payload) ders iceriginden doldur:
   - "concept_card": "takeaway" alani: Ders baglamina dayali altin cikarim veya kilit kural.
   - "process_flow": "steps" alani: [{"title": "1. Baslik", "desc": "Kisa aciklama"}] (2 veya 3 adim).
   - "comparison_table": "left_title", "right_title", "left_points": ["Madde 1", "Madde 2"], "right_points": ["Karsit 1", "Karsit 2"].
   - "equation_step": Yalnizca gercek bir matematiksel denklem/formul varsa sec. "formula": "LaTeX formulu", "variables": [{"symbol": "EV", "meaning": "Kazanilmis Deger"}]. Formul yoksa ASLA equation_step secme!
   - "dag_concept": "nodes": [{"name": "Onkosul", "desc": "Tanimi"}, {"name": "Hedef", "desc": "Tanimi"}] (2-3 dugum).
6. narration_script, sahnelerdeki anlatimlarin sirayla birlestirilmis hali olsun.
7. Yanit SADECE gecerli bir JSON nesnesi olsun. manim_code alanini bos birak; gorsel sahne olusturucu bu semaya gore animasyonu olusturur.
"""

EVREN_MANIM_USER_TEMPLATE = """Ogrencinin Takildigi Konu / Terim: {TOPIC}
Dersin Ilgili Bolumundeki Anlatim (tek bilgi kaynagi): "{CONTEXT}"
Hedef Dil: {LANG}
Hedef Sure: {DURATION_S} saniye

Asagidaki JSON semasina birebir uyarak yanit ver:
{{
  "title": "Konu Basligi",
  "target_concept": "Temel Kavram",
  "pedagogical_hook": "Ogrencinin dikkatini cekecek somut benzetme",
  "terms": [
    {{"term": "Terim Adi", "explanation": "B1 duzeyinde sade tanim"}}
  ],
  "narration_script": "Videonun bastan sona eksiksiz seslendirme metni",
  "scenes": [
    {{
      "id": 1,
      "scene_type": "concept_card",
      "title": "Giris ve Problem",
      "narration": "Bu sahnenin anlatimi",
      "duration_s": 12.0,
      "takeaway": "Kilit cikarim ve kural"
    }},
    {{
      "id": 2,
      "scene_type": "process_flow",
      "title": "Isleyis Semasi",
      "steps": [
        {{"title": "1. Girdi", "desc": "Ilk adim tanimi"}},
        {{"title": "2. Islem", "desc": "Mekanizma tanimi"}},
        {{"title": "3. Cikti", "desc": "Nihai sonuc"}}
      ],
      "narration": "Adimlarin anlatimi",
      "duration_s": 18.0
    }},
    {{
      "id": 3,
      "scene_type": "comparison_table",
      "title": "Karsilastirma ve Ozet",
      "left_title": "Yaklasim A",
      "right_title": "Yaklasim B",
      "left_points": ["Madde 1", "Madde 2"],
      "right_points": ["Madde 1", "Madde 2"],
      "narration": "Karsilastirma anlatimi ve altin cikarim",
      "duration_s": 15.0,
      "takeaway": "Sinav ve uygulama altin ipucu"
    }}
  ],
  "manim_code": ""
}}
"""



def _complete_spec(parsed: dict, topic_clean: str, lang: str, target_duration_s: float = 0.0) -> dict:
    """Fill in any field the model omitted so the renderer never sees a partial spec."""
    spec = dict(parsed or {})
    spec.setdefault("title", topic_clean)
    spec.setdefault("target_concept", topic_clean)
    spec.setdefault("pedagogical_hook", "")
    spec.setdefault("terms", [])
    spec.setdefault("narration_script", "")
    spec.setdefault("scenes", [])
    if not spec["narration_script"] and spec["scenes"]:
        spec["narration_script"] = " ".join(
            str(s.get("narration", "")).strip() for s in spec["scenes"] if s.get("narration")
        )
    if not spec["terms"] and spec["scenes"]:
        spec["terms"] = [{"term": topic_clean, "explanation": str(spec["scenes"][0].get("narration", ""))[:160]}]
    # Generate modular parametric Manim code using the scene-type builders
    spec["manim_code"] = generate_modular_manim_code(
        spec=spec,
        lang=lang,
        target_duration_s=target_duration_s,
    )
    spec["manim_code_source"] = "template"
    return spec


def generate_explainer_script_evren(
    topic: str,
    context: str = "",
    lang: str = "tr",
    duration_s: int = 50,
    model_id: str = DEFAULT_EVREN_MODEL,
) -> dict:
    """Author the narration script and Manim scene through the EVREN cloud LLM.

    Raises on any platform or parsing failure so the caller can tell EVREN apart
    from the degraded fallbacks.
    """
    from evren_llm import generate_evren

    topic_clean = topic.strip() or "Ders Kavramı"
    is_tr = str(lang).lower().startswith("tr")
    lang_name = "Türkçe" if is_tr else "English"

    user_prompt = EVREN_MANIM_USER_TEMPLATE.format(
        TOPIC=topic_clean,
        CONTEXT=(context[:900].strip() if context else topic_clean),
        LANG=lang_name,
        DURATION_S=duration_s,
    )
    raw = generate_evren(
        model_id,
        user_prompt,
        max_tokens=1500,
        temperature=0.15,
        system_prompt=EVREN_MANIM_SYSTEM_PROMPT,
    ).strip()
    return _complete_spec(_clean_json_output(raw), topic_clean, lang, duration_s)


def generate_explainer_script_qwen(
    topic: str,
    context: str = "",
    lang: str = "tr",
    duration_s: int = 50,
    model_id: str = "Qwen/Qwen3.5-4B",
) -> dict:
    """Generate structured terminology, narration script, and Manim code via local Qwen 3.5.

    Raises so the caller's backend chain can degrade to the heuristic spec instead of
    silently labelling a hand-written template as model output.
    """
    from run_llm import generate_hf

    topic_clean = topic.strip() or "Ders Kavramı"
    is_tr = str(lang).lower().startswith("tr")
    lang_name = "Türkçe" if is_tr else "English"

    prompt = MANIM_PROMPT_TEMPLATE.format(
        TOPIC=topic_clean,
        CONTEXT=context[:450] if context else topic_clean,
        LANG=lang_name,
        DURATION_S=duration_s,
    )
    raw = generate_hf(model_id, prompt, max_new_tokens=650).strip()
    return _complete_spec(_clean_json_output(raw), topic_clean, lang, duration_s)


def generate_explainer_script(
    topic: str,
    context: str = "",
    lang: str = "tr",
    duration_s: int = 50,
    backend: str = "evren",
    evren_model: str = DEFAULT_EVREN_MODEL,
    local_model_id: str = "Qwen/Qwen3.5-4B",
) -> dict:
    """Author the explainer spec, using the requested backend with a template safety net.

    A failing backend degrades to the heuristic spec, never to a different model: the
    engine the student picked is the engine the result is attributed to. The returned
    'script_backend' records which one actually produced the script.
    """
    chain = {
        "evren": ("evren", "heuristic"),
        "local": ("local", "heuristic"),
        "heuristic": ("heuristic",),
    }.get(str(backend).lower(), ("evren", "heuristic"))

    topic_clean = topic.strip() or "Ders Kavramı"
    backend_error = None
    for step in chain:
        try:
            if step == "evren":
                spec = generate_explainer_script_evren(topic_clean, context, lang, duration_s, evren_model)
            elif step == "local":
                spec = generate_explainer_script_qwen(topic_clean, context, lang, duration_s, local_model_id)
            else:
                spec = generate_fallback_explainer_spec(topic_clean, context, lang, duration_s)
            return {**spec, "script_backend": step, "script_backend_error": backend_error}
        except Exception as exc:
            logger.warning("Manimator script backend %r failed: %s", step, exc)
            backend_error = re.sub(r"evren_llm_[A-Za-z0-9_-]+", "[anahtar gizlendi]", str(exc))[:300]

    return {
        **generate_fallback_explainer_spec(topic_clean, context, lang, duration_s),
        "script_backend": "heuristic",
        "script_backend_error": backend_error,
    }


def generate_fallback_explainer_spec(
    topic: str,
    context: str = "",
    lang: str = "tr",
    duration_s: int = 50,
) -> dict:
    """Generate instant domain-aware high-fidelity explainer script & Manim code."""
    topic_clean = topic.strip() or "Temel Kavram"
    norm_topic = topic_clean.lower()
    is_tr = str(lang).lower().startswith("tr")

    if "kazanılmış değer" in norm_topic or "evm" in norm_topic or "sapma" in norm_topic:
        title = "Kazanılmış Değer Analizi (EVM)"
        hook = "Bir inşaat ustasının duvar örmesi gibi düşünün: 10 günde 100 tuğla örmeyi planladınız (PV), 80 tuğla ördünüz (EV) ama 12 günlük para harcadınız (AC)!"
        terms = [
            {"term": "PV (Planlanan Değer)", "explanation": "Belli bir tarihe kadar tamamlanması hedeflenen işin bütçesi."},
            {"term": "EV (Kazanılmış Değer)", "explanation": "Gerçekte tamamlanan işin hak ettiği bütçe değeri."},
            {"term": "AC (Gerçek Maliyet)", "explanation": "O işi bitirmek için kasadan fiilen çıkan para."},
            {"term": "CV (Maliyet Sapması)", "explanation": "EV - AC. Negatifse bütçe aşılmıştır, tehlike çanları çalar!"},
        ]
        script = (
            "Merhaba! Bu konuda kafanızın karışması çok normal. "
            "Kazanılmış Değer Analizi, projenizin bütçesinde ve takviminde nerede olduğunuzu gösteren bir pusuladır. "
            "Maliyet Sapması CV eşittir EV eksi AC'dir. EV gerçekleşen işi, AC harcanan parayı temsil eder. "
            "CV pozitifse proje tasarrufta, negatifse bütçe aşımında demektir. "
            "Unutmayın: EV daima ilk terimdir ve pozitif sonuç her zaman başarıyı gösterir."
        )
        scenes = [
            {
                "id": 1,
                "scene_type": "concept_card",
                "title": "Problem & Pusula",
                "narration": "Projenizin bütçesinde ve takviminde nerede olduğunuzu nasıl anlarsınız?",
                "duration_s": 10.0,
                "takeaway": "EVM, sadece harcanan parayı değil; paranın karşılığı üretilen gerçek değeri ölçer.",
            },
            {
                "id": 2,
                "scene_type": "equation_step",
                "title": "Maliyet Sapması Formülü",
                "formula": "CV = EV - AC",
                "variables": [
                    {"symbol": "EV", "meaning": "Kazanılmış Değer (Hak edilen bütçe)"},
                    {"symbol": "AC", "meaning": "Gerçekleşen Maliyet (Harcanan fiili nakit)"},
                ],
                "narration": "Maliyet Sapması CV eşittir EV eksi AC'dir. EV gerçekleşen işi, AC harcanan parayı temsil eder.",
                "duration_s": 18.0,
            },
            {
                "id": 3,
                "scene_type": "comparison_table",
                "title": "Sapma Analizi Karşılaştırması",
                "left_title": "CV > 0 (Bütçe Altı)",
                "right_title": "CV < 0 (Bütçe Aşımı)",
                "left_points": ["Tasarruf sağlanmış", "Maliyet kontrol altında", "Müdahale gerekmez"],
                "right_points": ["Bütçe aşımı / zarar", "Maliyet sapması negatif", "Düzeltici önlem şart"],
                "narration": "CV pozitifse proje tasarrufta, negatifse bütçe aşımında demektir. Düzeltici önlem alınmalıdır.",
                "duration_s": 14.0,
                "takeaway": "Altın Kural: EV daima formülün ilk terimidir. Pozitif değer tasarruf, negatif değer zarardır.",
            },
        ]
    elif "turizm" in norm_topic or "uktkg" in norm_topic:
        title = "Turizm Bileşenleri ve UKTKG Modeli"
        hook = "Harika bir restoran açtığınızı düşünün: Yemek lezzetli olsa bile yol yoksa kimse gelemez (Ulaşım), masa yoksa oturamaz (Konaklama)!"
        terms = [
            {"term": "Ulaşım (U)", "explanation": "Turistin destinasyona güvenli ve hızlı erişimini sağlar."},
            {"term": "Konaklama (K)", "explanation": "Otel, pansiyon ve barınma altyapısı."},
            {"term": "Tanıtım (T)", "explanation": "Bölgenin cazibesini dünyaya duyuran pazarlama."},
            {"term": "UKTKG İlkesi", "explanation": "Turizm bileşenlerinin bir bütün halinde işlemesini sağlayan kural."},
        ]
        script = (
            "Turizm dersinde takıldığınız UKTKG kavramını hemen basitleştirelim. "
            "Turizm tek bir işletmeden ibaret değil, birbirine kenetli bir hizmet zinciridir. "
            "Ulaşım, Konaklama, Tanıtım, Kamu Hizmetleri ve Güvenlik birbirini tamamlar. "
            "Turizm ürünleri depolanamaz veya ertesi güne stoklanamaz; üretim ve tüketim eşzamanlıdır."
        )
        scenes = [
            {
                "id": 1,
                "scene_type": "concept_card",
                "title": "Turizm Zinciri",
                "narration": "Turizm neden tek bir işletmeden ibaret değildir?",
                "duration_s": 10.0,
                "takeaway": "Turizm birleşik bir hizmettir; tek bir halkanın kopması tüm deneyimi bozar.",
            },
            {
                "id": 2,
                "scene_type": "process_flow",
                "title": "UKTKG Hizmet Zinciri",
                "steps": [
                    {"title": "1. Ulaşım & Erişim", "desc": "Hedefe hızlı ve güvenli varış"},
                    {"title": "2. Konaklama & Ağırlama", "desc": "Nitelikli otel ve dinlenme altyapısı"},
                    {"title": "3. Tanıtım & Güvenlik", "desc": "Kamu desteği, güvenlik ve memnuniyet"},
                ],
                "narration": "Ulaşım, Konaklama, Tanıtım, Kamu ve Güvenlik birbirine kenetlidir.",
                "duration_s": 18.0,
            },
            {
                "id": 3,
                "scene_type": "comparison_table",
                "title": "Turizm Hizmeti vs Fiziki Mal",
                "left_title": "Fiziki Ürünler",
                "right_title": "Turizm Hizmetleri",
                "left_points": ["Depolanabilir ve stoklanır", "Kullanım zamanı ertelenebilir"],
                "right_points": ["Stoklanamaz ve anlıktır", "Üretim ve tüketim eşzamanlıdır"],
                "narration": "Turizm ürünleri depolanamaz veya ertesi güne stoklanamaz; üretim ve tüketim eşzamanlıdır.",
                "duration_s": 14.0,
                "takeaway": "Altın Kural: Boş kalan otel odası veya uçak koltuğu ertesi güne devredilemez.",
            },
        ]
    elif "veri" in norm_topic or "python" in norm_topic or "model" in norm_topic:
        title = "Veri Ön İşleme (Data Preprocessing)"
        hook = "Ham petrolü rafineride benzine dönüştürmek gibidir: Ham veri doğrudan makine öğrenmesi modeline verilirse model boğulur!"
        terms = [
            {"term": "Eksik Veri (Missing Data)", "explanation": "Doldurulması veya temizlenmesi gereken boş hücreler."},
            {"term": "Ölçekleme (Scaling)", "explanation": "Farklı aralıktaki sayıları aynı teraziye getirmek."},
            {"term": "Girdi-Çıktı Ayrımı", "explanation": "Öznitelikler (X) ve hedef değişkenin (y) hazırlanması."},
        ]
        script = (
            "Yapay zeka ve makine öğrenmesinde vaktin yüzde sekseninin veri ön işlemeye harcandığını biliyor musunuz? "
            "Ham veri kirli ve eksiktir. Pandas ile eksik değerleri temizler, ölçekler ve modele uygun hale getiririz. "
            "Çöp giren sisteme çöp çıkar prensibini aklınızdan çıkarmayın!"
        )
        scenes = [
            {
                "id": 1,
                "scene_type": "concept_card",
                "title": "Ham Veri Sorunu",
                "narration": "Ham veri neden doğrudan modele verilemez?",
                "duration_s": 10.0,
                "takeaway": "Çöp giren sisteme çöp çıkar; model başarısının temeli temiz veridir.",
            },
            {
                "id": 2,
                "scene_type": "process_flow",
                "title": "Veri Hazırlık Adımları",
                "steps": [
                    {"title": "1. Eksik Veri Tespiti", "desc": "Boş hücrelerin temizlenmesi"},
                    {"title": "2. Özellik Ölçekleme", "desc": "Sayıları ortak teraziye getirme"},
                    {"title": "3. X ve y Ayrımı", "desc": "Girdi ve hedef matrislerinin oluşumu"},
                ],
                "narration": "Temizleme, ölçekleme ve dönüştürme adımları sırayla uygulanır.",
                "duration_s": 18.0,
            },
            {
                "id": 3,
                "scene_type": "comparison_table",
                "title": "Ham Veri vs Temizlenmiş Veri",
                "left_title": "Ham Veri",
                "right_title": "İşlenmiş Veri",
                "left_points": ["Uç değerler ve gürültü", "Modelde yüksek yanılgı"],
                "right_points": ["Dengeli ve standart ölçek", "Yüksek tahmin doğruluğu"],
                "narration": "Temiz veri, yüksek doğruluklu tahminler ve genellenebilir modeller üretir.",
                "duration_s": 14.0,
                "takeaway": "Altın Kural: Model algoritmasından önce veri kalitesini optimize edin.",
            },
        ]
    elif "proje" in norm_topic or "yönetici" in norm_topic or "alternatif" in norm_topic or "problem" in norm_topic:
        title = "Proje Yönetimi ve Karar Verme"
        hook = "Bir orkestra şefini düşünün: Tüm enstrümanların (zaman, maliyet, ekip) doğru zamanda çalmasını sağlar ve aksaklıklarda en uygun alternatifi seçer!"
        terms = [
            {"term": "Proje Yöneticisi", "explanation": "Ortaya çıkan problemlerde hedeflere en uygun alternatife karar veren ve kaynakları yöneten lider."},
            {"term": "Alternatiflerin Değerlendirilmesi", "explanation": "Problemleri çözerken zaman ve maliyet hedeflerine en uygun seçeneği belirleme süreci."},
            {"term": "İzleme ve Kontrol", "explanation": "Faaliyetler sürerken maliyetleri düzenli izleyip sapmaları erkenden tespit etme."},
            {"term": "Düzeltici Önlemler", "explanation": "Proje hedeflerinden sapma olduğunda çok geç olmadan devreye sokulan müdahaleler."},
        ]
        script = (
            "Dersin bu anında tam olarak takıldığınız noktayı aydınlatalım. "
            "Proje yöneticisi, ortaya çıkan problemlerin çözümü için önerilen alternatiflerden hangisinin proje amaçlarına daha uygun olduğuna karar verir. "
            "Öte yandan faaliyetler devam ederken maliyetler izlenmeli ve gerektiğinde düzeltici önlemler çok geç olmadan alınmalıdır. "
            "İzleme nerede olduğumuzu gösterir, kontrol ise rotaya geri döndürür."
        )
        scenes = [
            {
                "id": 1,
                "scene_type": "concept_card",
                "title": "Karar Verme Sorumluluğu",
                "narration": "Proje yöneticisi alternatifler arasından en uygun seçeneği belirler.",
                "duration_s": 12.0,
                "takeaway": "Yönetici sadece kriz çözmez; alternatifleri proje amaçlarına göre rasyonel seçer.",
            },
            {
                "id": 2,
                "scene_type": "process_flow",
                "title": "Karar ve Müdahale Akışı",
                "steps": [
                    {"title": "1. Problem Tespiti", "desc": "Sapmanın erkenden belirlenmesi"},
                    {"title": "2. Alternatif Analizi", "desc": "Zaman ve maliyet etki analizi"},
                    {"title": "3. Düzeltici Aksiyon", "desc": "Hedefe dönüş müdahalesi"},
                ],
                "narration": "Faaliyetler sürerken maliyetler ve ilerleme düzenli izlenir.",
                "duration_s": 18.0,
            },
            {
                "id": 3,
                "scene_type": "comparison_table",
                "title": "İzleme vs Kontrol Ayrımı",
                "left_title": "İzleme (Monitoring)",
                "right_title": "Kontrol (Controlling)",
                "left_points": ["Veri toplama ve kaydetme", "Performans raporlama", "Mevcut durumu görme"],
                "right_points": ["Sapmayı giderme", "Düzeltici önlem alma", "Planlanan hedefe geri döndürme"],
                "narration": "Sapmalar büyümeden çok geç olmadan önlem alınmalıdır.",
                "duration_s": 14.0,
                "takeaway": "İzleme hız göstergesine bakmaktır; kontrol ise hız aşılınca frene basmaktır.",
            },
        ]
    elif "izleme" in norm_topic or "kontrol" in norm_topic:
        title = "Proje İzleme ve Kontrol Süreci"
        hook = "Arabayla seyahat ederken hız göstergesini kontrol etmek 'İzleme', hız sınırı aşılınca frene basmak ise 'Kontrol ve Düzeltici Eylem'dir!"
        terms = [
            {"term": "İzleme (Monitoring)", "explanation": "Proje performansına ilişkin bilgilerin düzenli olarak toplanması, kaydedilmesi ve raporlanması."},
            {"term": "Kontrol (Controlling)", "explanation": "Fiili durum ile planlanan hedefler arasındaki farkın belirlenip düzeltici önlem alınması."},
            {"term": "Performans Kısıtları", "explanation": "Projenin zaman, maliyet ve kalite sınırları."},
            {"term": "Düzeltici Eylemler", "explanation": "Sapmayı gidererek projeyi ana plana geri döndüren müdahaleler."},
        ]
        script = (
            "Dersin bu anında geçen İzleme ve Kontrol kavramlarını netleştirelim. "
            "İzleme; projenin nerede olduğunu verilerle raporlama sürecidir. "
            "Kontrol ise planlanan hedef ile gerçekleşen durum arasındaki farkı tespit edip gereken düzeltici önlemleri çok geç olmadan almaktır. "
            "Zaman, maliyet ve performans kısıtları kontrolün temel direkleridir!"
        )
        scenes = [
            {
                "id": 1,
                "scene_type": "concept_card",
                "title": "İzleme Nedir?",
                "narration": "Performans verilerinin toplanıp raporlanması sürecidir.",
                "duration_s": 12.0,
                "takeaway": "İzleme olmadan sapmalar görülemez; düzenli veri toplama şarttır.",
            },
            {
                "id": 2,
                "scene_type": "process_flow",
                "title": "Kontrol Döngüsü",
                "steps": [
                    {"title": "1. Veri Toplama", "desc": "Fiili ilerleme tespiti"},
                    {"title": "2. Sapma Analizi", "desc": "Planlanan ile fiili farkı"},
                    {"title": "3. Düzeltici Eylem", "desc": "Plana dönüş müdahalesi"},
                ],
                "narration": "Planlanan ile fiili durum arasındaki fark incelenir.",
                "duration_s": 18.0,
            },
            {
                "id": 3,
                "scene_type": "comparison_table",
                "title": "İzleme vs Kontrol",
                "left_title": "İzleme (Monitoring)",
                "right_title": "Kontrol (Controlling)",
                "left_points": ["Pasif gözlem ve kayıt", "Raporlama odaklı"],
                "right_points": ["Aktif müdahale", "Sapmayı düzeltme odaklı"],
                "narration": "Gecikme ve maliyet aşımına karşı çok geç olmadan önlem alınır.",
                "duration_s": 14.0,
                "takeaway": "İzleme rotayı gösterir, kontrol direksiyonu kırar.",
            },
        ]
    elif context and len(context.strip()) > 20:
        clean_ctx = context.strip()
        first_sent = clean_ctx.split(".")[0].strip() if "." in clean_ctx else clean_ctx[:180]
        if len(first_sent) > 180:
            first_sent = first_sent[:177].rsplit(" ", 1)[0] + "..."
        title = topic_clean
        hook = f"Ders kesitindeki ifade: {first_sent}"
        terms = [
            {"term": topic_clean, "explanation": f"Bu kesitteki anlamı: {first_sent}"},
            {"term": "Temel Kural", "explanation": "Ders bağlamında terimin üstlendiği pedagojik görev."},
        ]
        script = (
            f"Bu ders kesitinde odak kavram {topic_clean}. "
            f"İlgili ifade şöyle: {first_sent}. "
            "Bu otomatik kısa özet, kesitteki bu ifadenin dışına çıkarak ek bilgi varsaymıyor. "
            "Terimin cümledeki görevine ve önceki-sonraki ifadelerle ilişkisine dikkat edin."
        )
        scenes = [
            {
                "id": 1,
                "scene_type": "concept_card",
                "title": f"Odak: {topic_clean}",
                "narration": f"Bu kesitte odak kavram {topic_clean}.",
                "duration_s": 12.0,
                "takeaway": f"{topic_clean} kavramını ders bağlamındaki anlamıyla değerlendirin.",
            },
            {
                "id": 2,
                "scene_type": "dag_concept",
                "title": "Kavram Bağıntısı",
                "nodes": [
                    {"name": "Ders Bağlamı", "desc": "Öncül anlatım"},
                    {"name": topic_clean, "desc": "Kilit terim"},
                    {"name": "Sonuç Çıkarımı", "desc": "Uygulama"},
                ],
                "narration": first_sent,
                "duration_s": 18.0,
            },
            {
                "id": 3,
                "scene_type": "concept_card",
                "title": "Bağlamı İzle",
                "narration": "Terimin cümledeki görevine ve çevresindeki ifadelere dikkat edin.",
                "duration_s": 14.0,
                "takeaway": "Altın Kural: Ezberden kaçının; terimi cümledeki rolüyle öğrenin.",
            },
        ]
    else:
        title = topic_clean
        hook = f"{topic_clean} kavramı, hedeflere ulaşırken sürecin temel taşlarını doğru sırayla birleştirmek gibidir."
        terms = [
            {"term": topic_clean, "explanation": "Bu ders bölümünde incelenen kilit pedagojik kavram."},
            {"term": "Temel İlke", "explanation": "Önceki adımlarla bağlantılı olarak yürütülen ana kural."},
            {"term": "Uygulama", "explanation": "Kavramın pratik problem çözümündeki yansıması."},
        ]
        script = (
            f"Videoda takıldığınız {topic_clean} konusunu kısaca özetleyelim. "
            "Karmaşık gibi görünen bu konu aslında adım adım bir mantık zincirine dayanır. "
            "Önce temel tanımı netleştiriyoruz, ardından pratik uygulamadaki yerine bakıyoruz. "
            "Böylece sınavda karşınıza çıktığında neden ve sonuç ilişkisini hemen kurabilirsiniz."
        )
        scenes = [
            {
                "id": 1,
                "scene_type": "concept_card",
                "title": "Giriş ve Problem",
                "narration": f"{topic_clean} kavramının temel sorusu.",
                "duration_s": 10.0,
                "takeaway": f"{topic_clean} konusunun temel mantığını adım adım kavrayın.",
            },
            {
                "id": 2,
                "scene_type": "process_flow",
                "title": "Çekirdek Mantık Akışı",
                "steps": [
                    {"title": f"1. {safe_text(topic_clean, 14)}", "desc": "Temel tanım"},
                    {"title": "2. İşleyiş", "desc": "Neden-sonuç mekanizması"},
                    {"title": "3. Çözüm", "desc": "Pratik uygulama"},
                ],
                "narration": "Adım adım işleyen neden-sonuç mekanizması incelenir.",
                "duration_s": 18.0,
            },
            {
                "id": 3,
                "scene_type": "dag_concept",
                "title": "Özet ve İpucu",
                "nodes": [
                    {"name": "Zemin", "desc": "Önkoşul"},
                    {"name": topic_clean, "desc": "Kavram"},
                    {"name": "Sonuç", "desc": "Sınav ipucu"},
                ],
                "narration": "Akılda tutulması gereken temel nokta ve çıkarım.",
                "duration_s": 12.0,
            },
        ]

    spec_res = {
        "title": title,
        "target_concept": topic_clean,
        "pedagogical_hook": hook,
        "terms": terms,
        "narration_script": script,
        "scenes": scenes,
    }
    spec_res["manim_code"] = generate_modular_manim_code(spec_res, lang=lang, target_duration_s=duration_s)
    spec_res["manim_code_source"] = "template"
    return spec_res


import textwrap

# Beat budget of the template scene below: 8 animations and 5 holds. The literal
# run_time values in generate_default_manim_code must keep matching these totals.
TEMPLATE_ANIMATION_S = 9.1
TEMPLATE_HOLD_S = 6.0
TEMPLATE_SCENE_S = TEMPLATE_ANIMATION_S + TEMPLATE_HOLD_S


def _template_pace(target_duration_s: float) -> tuple[float, float]:
    """Return (animation, hold) multipliers that stretch the template to the target.

    Holds absorb the extra seconds first so the writing animations keep a natural
    drawing speed instead of crawling when the narration runs much longer than the
    unscaled template.
    """
    if target_duration_s <= 0:
        return 1.0, 1.0
    scale = max(0.5, min(8.0, target_duration_s / TEMPLATE_SCENE_S))
    anim_scale = max(0.5, min(2.5, scale))
    hold_scale = (scale * TEMPLATE_SCENE_S - TEMPLATE_ANIMATION_S * anim_scale) / TEMPLATE_HOLD_S
    return anim_scale, max(0.5, min(8.0, hold_scale))


def generate_default_manim_code(
    topic: str,
    analogy: str,
    terms: list[dict],
    lang: str = "tr",
    target_duration_s: float = 0.0,
) -> str:
    """Generate guaranteed error-free Manim Community Python code with full text wrapping.

    Avoids LaTeX dependency by using pure Text and VGroup geometric shapes. When
    target_duration_s is given, the scene is paced to that length so the merge step
    does not have to freeze the closing frame under the narration.
    """
    anim_scale, hold_scale = _template_pace(target_duration_s)

    def beat(seconds: float, hold: bool = False) -> str:
        return f"{seconds * (hold_scale if hold else anim_scale):.2f}"

    def wrapped_block(raw: str, width: int, max_lines: int) -> str:
        """Wrap text into card-sized lines, marking any cut so it never reads as a broken sentence."""
        lines = textwrap.wrap(raw, width=width)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1].rsplit(" ", 1)[0].rstrip(",.;:!?") + "…"
        return "\\n".join(lines) if lines else ""

    clean_topic = re.sub(r'["\\]', '', topic).strip()[:50]
    title_font_size = 36 if len(clean_topic) <= 28 else (30 if len(clean_topic) <= 42 else 24)

    clean_analogy = re.sub(r'["\\]', '', analogy).strip() if analogy else "Pedagojik Anlatım"
    # Wrap analogy into clean lines without clipping words or sentences
    wrapped_analogy_str = wrapped_block(clean_analogy, 54, 4)
    num_analogy_lines = max(1, wrapped_analogy_str.count("\\n") + 1)
    hook_box_height = max(1.25, 0.42 * num_analogy_lines + 0.45)
    hook_font_size = 24 if num_analogy_lines <= 2 else 21

    # Terms and Descriptions wrapped cleanly
    term1 = re.sub(r'["\\]', '', terms[0].get("term", "Kavram 1")).strip()[:28] if len(terms) > 0 else "Kavram 1"
    desc1_raw = re.sub(r'["\\]', '', terms[0].get("explanation", "")).strip() if len(terms) > 0 else "Temel Tanım"
    desc1_str = wrapped_block(desc1_raw, 24, 4)

    term2 = re.sub(r'["\\]', '', terms[1].get("term", "Kavram 2")).strip()[:28] if len(terms) > 1 else "Kavram 2"
    desc2_raw = re.sub(r'["\\]', '', terms[1].get("explanation", "")).strip() if len(terms) > 1 else "Pratik Kural"
    desc2_str = wrapped_block(desc2_raw, 24, 4)

    code = f'''from manim import *

class ExplainerScene(Scene):
    def construct(self):
        # Configure dark aesthetic
        self.camera.background_color = "#0B0F19"

        # 1. Header Title Banner
        title = Text("{clean_topic}", font_size={title_font_size}, weight=BOLD, color="#38BDF8")
        title.to_edge(UP, buff=0.5)
        
        underline = Line(LEFT * 4.2, RIGHT * 4.2, color="#818CF8", stroke_width=3)
        underline.next_to(title, DOWN, buff=0.18)
        
        self.play(Write(title), GrowFromCenter(underline), run_time={beat(1.3)})
        self.wait({beat(0.4, True)})

        # 2. Hook / Analogy Card with Full Multiline Text
        hook_box = RoundedRectangle(corner_radius=0.2, height={hook_box_height}, width=10.2, color="#6366F1")
        hook_box.set_fill("#1E1B4B", opacity=0.85)
        hook_box.next_to(underline, DOWN, buff=0.35)
        
        hook_text = Text("{wrapped_analogy_str}", font_size={hook_font_size}, line_spacing=0.85, color="#F8FAFC")
        hook_text.move_to(hook_box.get_center())
        
        self.play(FadeIn(hook_box), Write(hook_text), run_time={beat(1.4)})
        self.wait({beat(0.8, True)})

        # 3. Two Key Concept Visual Cards
        card1 = RoundedRectangle(corner_radius=0.15, height=2.6, width=4.9, color="#059669")
        card1.set_fill("#064E3B", opacity=0.8)
        card1.move_to(LEFT * 2.75 + DOWN * 1.6)

        c1_title = Text("{term1}", font_size=26, weight=BOLD, color="#34D399")
        c1_title.next_to(card1.get_top(), DOWN, buff=0.25)
        c1_desc = Text("{desc1_str}", font_size=20, line_spacing=0.8, color="#E2E8F0")
        c1_desc.next_to(c1_title, DOWN, buff=0.25)
        group1 = VGroup(card1, c1_title, c1_desc)

        card2 = RoundedRectangle(corner_radius=0.15, height=2.6, width=4.9, color="#D97706")
        card2.set_fill("#78350F", opacity=0.8)
        card2.move_to(RIGHT * 2.75 + DOWN * 1.6)

        c2_title = Text("{term2}", font_size=26, weight=BOLD, color="#FBBF24")
        c2_title.next_to(card2.get_top(), DOWN, buff=0.25)
        c2_desc = Text("{desc2_str}", font_size=20, line_spacing=0.8, color="#E2E8F0")
        c2_desc.next_to(c2_title, DOWN, buff=0.25)
        group2 = VGroup(card2, c2_title, c2_desc)

        # Arrow connecting concepts
        arrow = Arrow(card1.get_right(), card2.get_left(), buff=0.15, color="#CBD5E1", stroke_width=4)

        self.play(FadeIn(group1, shift=RIGHT), run_time={beat(1.2)})
        self.play(GrowArrow(arrow), run_time={beat(0.8)})
        self.play(FadeIn(group2, shift=LEFT), run_time={beat(1.2)})
        self.wait({beat(1.5, True)})

        # 4. Golden Rule Highlight
        rule_box = SurroundingRectangle(group2, color="#F59E0B", buff=0.15, stroke_width=3)
        self.play(Create(rule_box), run_time={beat(1.0)})
        self.wait({beat(1.8, True)})

        # 5. Outro Fade
        self.play(
            FadeOut(group1),
            FadeOut(arrow),
            FadeOut(group2),
            FadeOut(rule_box),
            FadeOut(hook_box),
            FadeOut(hook_text),
            run_time={beat(1.0)}
        )
        
        final_box = RoundedRectangle(corner_radius=0.18, height=1.6, width=10.0, color="#10B981")
        final_box.set_fill("#064E3B", opacity=0.9)
        final_box.move_to(DOWN * 0.6)
        final_title = Text("💡 Altın Çıkarım", font_size=22, weight=BOLD, color="#6EE7B7")
        final_title.next_to(final_box.get_top(), DOWN, buff=0.2)
        final_desc = Text("Temel mantığı kavra, adımları ve kuralları bağlamında uygula.", font_size=16, color="#FFFFFF")
        final_desc.next_to(final_title, DOWN, buff=0.15)
        final_grp = VGroup(final_box, final_title, final_desc)
        self.play(GrowFromCenter(final_grp), run_time={beat(1.0)})
        self.wait({beat(1.5, True)})
'''
    return code


# ---------------------------------------------------------------------------
# 2. TTS Voiceover Audio Generation (Edge-TTS)
# ---------------------------------------------------------------------------

async def _synthesize_edge_tts_async(text: str, voice: str, out_path: Path) -> Path:
    """Synthesize speech using Microsoft Edge Neural TTS."""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))
    return out_path


def _synthesize_with_google_tts(text: str, out_path: Path, lang: str) -> bool:
    """Synthesize the narration with Google's public TTS endpoint, which speaks Turkish correctly.

    Used when the Edge neural voice throttles this machine to roughly one request per
    minute. The endpoint caps how much text it accepts, so the narration is fetched in
    chunks and the same-format MP3 frames are concatenated.
    """
    import requests

    chunks = textwrap.wrap(text, width=_GOOGLE_TTS_CHUNK_CHARS) or [text]
    audio = b""
    try:
        for idx, chunk in enumerate(chunks):
            res = requests.get(
                _GOOGLE_TTS_URL,
                params={
                    "ie": "UTF-8", "tl": str(lang).lower()[:2], "client": "tw-ob",
                    "total": str(len(chunks)), "idx": str(idx),
                    "q": chunk, "textlen": str(len(chunk)),
                },
                headers={"User-Agent": _GOOGLE_TTS_UA},
                timeout=25,
            )
            if res.status_code != 200 or len(res.content) < 500:
                logger.warning(
                    "Google TTS rejected chunk %d/%d: HTTP %s", idx + 1, len(chunks), res.status_code
                )
                return False
            audio += res.content
    except Exception as exc:
        logger.warning("Google TTS narration failed: %s", str(exc)[:160])
        return False

    out_path.write_bytes(audio)
    return out_path.exists() and out_path.stat().st_size > 1000


_SAPI_PS = r"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$want = $env:EDUTERM_TTS_CULTURE
$match = $synth.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -like "$want*" } | Select-Object -First 1
if ($match) { $synth.SelectVoice($match.VoiceInfo.Name) }
else { Write-Warning "No installed SAPI voice for culture '$want'; falling back to '$($synth.Voice.Name)'." }
$synth.SetOutputToWaveFile($env:EDUTERM_TTS_WAV)
Write-Output "$($synth.Voice.Name)|$($synth.Voice.Culture.Name)"
Get-Content -Raw -Encoding UTF8 $env:EDUTERM_TTS_TXT | ForEach-Object { $synth.Speak($_) }
$synth.Dispose()
"""


def _synthesize_with_windows_sapi(text: str, out_path: Path, lang: str = "tr") -> Optional[str]:
    """Speak the narration with an installed OS voice; return the voice label or None.

    The narration is passed through files rather than the command line, so
    model-authored text can never be interpreted as shell input. The voice is chosen
    by culture because a mismatched (e.g. English) voice mangles Turkish text, and
    the label is reported so the caller can say so out loud instead of passing the
    result off as the requested neural voice.
    """
    if os.name != "nt":
        return None

    txt = TEMP_DIR / f"{out_path.stem}.sapi.txt"
    wav = TEMP_DIR / f"{out_path.stem}.sapi.wav"
    try:
        txt.write_text(text, encoding="utf-8")
        env = {
            **os.environ,
            "EDUTERM_TTS_TXT": str(txt),
            "EDUTERM_TTS_WAV": str(wav),
            "EDUTERM_TTS_CULTURE": f"{str(lang).lower()[:2]}-",
        }
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _SAPI_PS],
            env=env, capture_output=True, text=True, timeout=180,
        )
        if res.returncode != 0 or not wav.exists() or wav.stat().st_size < 5000:
            logger.warning("Windows SAPI narration failed: %s", (res.stderr or "")[:200])
            return None
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", "128k", str(out_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=120,
        )
        if not (out_path.exists() and out_path.stat().st_size > 500):
            return None
        spoken = (res.stdout or "").strip().splitlines()
        name, _, culture = (spoken[-1] if spoken else "").partition("|")
        return f"{name} ({culture})" if name else "Windows SAPI5"
    except Exception as exc:
        logger.warning("Windows SAPI narration error: %s", exc)
        return None
    finally:
        txt.unlink(missing_ok=True)
        wav.unlink(missing_ok=True)


def synthesize_narration_audio(
    script_text: str,
    lang: str = "tr",
    voice: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> Tuple[Path, Dict[str, str]]:
    """Generate the voiceover (.mp3) and report which engine and voice really produced it.

    Chain: Edge neural voice (retried once, because it throttles this machine) ->
    Google TTS -> the offline system voice -> silence. Only the first two pronounce
    the language reliably, so the engine is reported rather than assumed.
    """
    is_tr = str(lang).lower().startswith("tr")
    selected_voice = voice or (DEFAULT_VOICE_TR if is_tr else DEFAULT_VOICE_EN)

    out_name = output_filename or f"tts_{uuid.uuid4().hex[:8]}.mp3"
    out_path = TEMP_DIR / out_name

    clean_text = script_text.strip()
    if not clean_text:
        clean_text = "Ders anlatımı tamamlandı."

    for attempt, delay in enumerate(_EDGE_RETRY_WAS_S, 1):
        if delay:
            time.sleep(delay)
        try:
            asyncio.run(_synthesize_edge_tts_async(clean_text, selected_voice, out_path))
            if out_path.exists() and out_path.stat().st_size > 500:
                return out_path, {"narration_engine": "edge-tts", "narration_voice": selected_voice}
            failure = "no audio bytes returned"
        except Exception as exc:
            failure = str(exc)[:160]
        logger.warning("Edge-TTS attempt %d/%d failed: %s", attempt, len(_EDGE_RETRY_WAS_S), failure)

    if _synthesize_with_google_tts(clean_text, out_path, lang):
        google_voice = f"Google TTS ({str(lang).lower()[:2]})"
        logger.warning("Edge-TTS unavailable; narration used %s instead of %s.", google_voice, selected_voice)
        return out_path, {"narration_engine": "google-tts", "narration_voice": google_voice}

    sapi_voice = _synthesize_with_windows_sapi(clean_text, out_path, lang)
    if sapi_voice:
        logger.warning(
            "Edge-TTS unavailable; narration used the offline system voice %s instead of %s.",
            sapi_voice, selected_voice,
        )
        return out_path, {"narration_engine": "sapi5-offline", "narration_voice": sapi_voice}

    # Last resort: 5s of silence, so the render still completes without a voice track
    logger.error("No text-to-speech engine produced audio; the video will be silent.")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "5", str(out_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception as e:
        logger.error("FFmpeg silence creation failed: %s", e)

    return out_path, {"narration_engine": "silent", "narration_voice": ""}


def get_media_duration_seconds(file_path: Path) -> float:
    """Probe audio or video duration in seconds via ffprobe/ffmpeg."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return max(1.0, float(res.stdout.strip()))
    except Exception:
        return 15.0


def synthesize_scenes_narration_audio(
    scenes: List[Dict[str, Any]],
    lang: str = "tr",
    voice: Optional[str] = None,
    master_output_filename: Optional[str] = None,
    fallback_script: str = "",
) -> Tuple[Path, List[float], Dict[str, str]]:
    """Synthesize each scene's narration to measure exact scene-by-scene audio durations.

    Returns:
        (master_audio_path, scene_durations, narration_info)
    """
    valid_scenes = [s for s in scenes if s.get("narration") and str(s.get("narration")).strip()]
    master_name = master_output_filename or f"tts_master_{uuid.uuid4().hex[:8]}.mp3"
    master_path = TEMP_DIR / master_name

    # If there are multiple scenes with narration, attempt per-scene audio synthesis
    if len(valid_scenes) >= 2:
        scene_files: List[Path] = []
        scene_durations: List[float] = []
        master_info: Dict[str, str] = {}
        all_succeeded = True

        for idx, sc in enumerate(valid_scenes):
            sc_text = str(sc.get("narration")).strip()
            sc_out_name = f"sc_{uuid.uuid4().hex[:6]}_{idx}.mp3"
            sc_path, sc_info = synthesize_narration_audio(
                script_text=sc_text,
                lang=lang,
                voice=voice,
                output_filename=sc_out_name,
            )
            if not master_info and sc_info.get("narration_engine") != "silent":
                master_info = sc_info

            dur = get_media_duration_seconds(sc_path)
            if not sc_path.exists() or dur <= 0.5:
                all_succeeded = False
                break
            scene_files.append(sc_path)
            scene_durations.append(dur)

        if all_succeeded and len(scene_files) == len(valid_scenes):
            concat_list = TEMP_DIR / f"concat_{uuid.uuid4().hex[:6]}.txt"
            try:
                concat_list.write_text(
                    "\n".join(f"file '{p.resolve().as_posix()}'" for p in scene_files),
                    encoding="utf-8",
                )
                cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(concat_list),
                    "-c:a", "libmp3lame", "-b:a", "192k",
                    str(master_path),
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                if master_path.exists() and master_path.stat().st_size > 1000:
                    for p in scene_files:
                        p.unlink(missing_ok=True)
                    return master_path, scene_durations, master_info or {"narration_engine": "edge-tts", "narration_voice": ""}
            except Exception as exc:
                logger.warning("Scene audio concatenation failed: %s; falling back to monolithic synthesis", exc)
            finally:
                concat_list.unlink(missing_ok=True)
                for p in scene_files:
                    p.unlink(missing_ok=True)

    # Monolithic fallback: synthesize full text and allocate durations proportionally
    full_script = " ".join(str(s.get("narration", "")).strip() for s in scenes if s.get("narration"))
    if not full_script.strip():
        full_script = fallback_script.strip() or "Ders anlatımı tamamlandı."


    master_path, master_info = synthesize_narration_audio(
        script_text=full_script,
        lang=lang,
        voice=voice,
        output_filename=master_name,
    )
    total_audio_s = get_media_duration_seconds(master_path)

    # Proportionally allocate total measured audio duration among scenes
    sum_declared = sum(float(s.get("duration_s") or 10.0) for s in scenes) or 1.0
    scene_durations = [
        max(2.0, total_audio_s * (float(s.get("duration_s") or 10.0) / sum_declared))
        for s in scenes
    ]
    if scene_durations:
        # Guarantee exact sum match down to milliseconds
        diff = total_audio_s - sum(scene_durations)
        scene_durations[-1] = max(1.0, scene_durations[-1] + diff)

    return master_path, scene_durations, master_info


# ---------------------------------------------------------------------------
# 3. Manim Code Execution & Self-Healing Engine
# ---------------------------------------------------------------------------

def _heal_manim_code(code: str, error_trace: str) -> str:
    """Repair common syntax or missing attribute errors in LLM Manim code."""
    healed = code

    # 1. Ensure essential imports
    if "from manim import *" not in healed:
        healed = "from manim import *\n" + healed

    # 2. Replace LaTeX MathTex with Text if TexTemplate/LaTeX failed
    if "LaTeX" in error_trace or "MathTex" in error_trace or "xdvipdfmx" in error_trace:
        healed = re.sub(r"MathTex\((.*?)\)", r"Text(\1)", healed)

    # 3. Repair unclosed quotes or brackets
    if "SyntaxError" in error_trace:
        lines = healed.splitlines()
        clean_lines = []
        for l in lines:
            # Fix unbalanced double quotes in single-line Text
            if "Text(" in l and l.count('"') % 2 != 0:
                l = l + '"'
            clean_lines.append(l)
        healed = "\n".join(clean_lines)

    # 4. Remove unsupported LaTeX math wrappers
    healed = healed.replace("$", "").replace("\\frac", "").replace("\\text", "")

    return healed


_BLOCKED_MANIM_NAMES = {
    "open", "eval", "exec", "compile", "input", "breakpoint", "__import__",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "os", "sys", "subprocess", "shutil", "socket", "requests", "urllib",
    "pathlib", "tempfile", "ctypes", "pickle", "base64",
}


def _validate_manim_code(code: str) -> Tuple[bool, str]:
    """Static AST whitelist check for LLM-generated Manim scene code.

    Only 'from manim import *' is allowed; filesystem, network, and process
    access plus dunder attribute tricks are rejected before the code ever
    reaches a subprocess. Returns (ok, reason).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            return False, f"import of {node.names[0].name!r} is not allowed"
        if isinstance(node, ast.ImportFrom) and node.module != "manim":
            return False, f"import from {node.module!r} is not allowed"
        if isinstance(node, ast.Name) and node.id in _BLOCKED_MANIM_NAMES:
            return False, f"use of {node.id!r} is not allowed"
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and node.attr.endswith("__")
        ):
            return False, f"dunder attribute {node.attr!r} is not allowed"
    return True, ""


def _resolve_tier(quality: str) -> Dict[str, Any]:
    """Map a quality name to its render settings, falling back to the default tier."""
    return VIDEO_QUALITY_TIERS.get(str(quality).lower(), VIDEO_QUALITY_TIERS[DEFAULT_VIDEO_QUALITY])


def _manim_render_cmd(work_dir: Path, script_path: Path, tier: Dict[str, Any]) -> List[str]:
    """Build the Manim CLI call for one tier: explicit resolution and frame rate."""
    width, height = tier["resolution"]
    return [
        sys.executable,
        "-m", "manim",
        "-r", f"{width},{height}",
        "--fps", str(tier["fps"]),
        "--media_dir", str(work_dir),
        str(script_path),
        "ExplainerScene",
    ]


def render_manim_scene(
    scene_code: str,
    output_name: str,
    quality: str = DEFAULT_VIDEO_QUALITY,
) -> Optional[Path]:
    """Execute Python Manim code and render MP4 video file.

    Features automatic error recovery / self-healing up to 2 attempts.
    """
    job_id = uuid.uuid4().hex[:8]
    work_dir = TEMP_DIR / f"job_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    tier = _resolve_tier(quality)
    current_code = scene_code
    script_path = work_dir / "scene.py"

    code_ok, reject_reason = _validate_manim_code(current_code)
    if not code_ok:
        logger.warning("Rejecting generated Manim code before execution: %s", reject_reason)

    for attempt in range(2 if code_ok else 0):
        script_path.write_text(current_code, encoding="utf-8")

        cmd = _manim_render_cmd(work_dir, script_path, tier)

        logger.info(
            "Rendering Manim scene at %dx%d@%.0ffps (attempt %d)...",
            tier["resolution"][0], tier["resolution"][1], tier["fps"], attempt + 1,
        )
        res = subprocess.run(cmd, capture_output=True, text=True)

        if res.returncode == 0:
            # Locate generated MP4 in work_dir/videos/
            mp4_candidates = list(work_dir.glob("videos/**/*.mp4"))
            # Filter out partial_movie_files
            finals = [p for p in mp4_candidates if "partial_movie_files" not in str(p)]
            if finals:
                target_path = VIDEOS_DIR / f"{output_name}.mp4"
                shutil.copy2(finals[0], target_path)
                # Cleanup temp work dir
                shutil.rmtree(work_dir, ignore_errors=True)
                return target_path
        else:
            logger.warning("Manim render error on attempt %d:\n%s", attempt + 1, res.stderr[:500])
            current_code = _heal_manim_code(current_code, res.stderr)
            code_ok, reject_reason = _validate_manim_code(current_code)
            if not code_ok:
                logger.warning("Healed Manim code failed safety validation: %s", reject_reason)
                break

    # Fallback to guaranteed default code if custom code failed both times
    logger.info("Falling back to guaranteed default Manim code template...")
    fallback_code = generate_default_manim_code(
        topic="Özel Anlatım",
        analogy="Öğrencinin anlamadığı kavram adım adım modellenmiştir.",
        terms=[{"term": "Kavram", "explanation": "Pedagojik özet"}],
    )
    script_path.write_text(fallback_code, encoding="utf-8")
    subprocess.run(_manim_render_cmd(work_dir, script_path, tier), capture_output=True)
    finals = [p for p in work_dir.glob("videos/**/*.mp4") if "partial_movie_files" not in str(p)]
    if finals:
        target_path = VIDEOS_DIR / f"{output_name}.mp4"
        shutil.copy2(finals[0], target_path)
        shutil.rmtree(work_dir, ignore_errors=True)
        return target_path

    shutil.rmtree(work_dir, ignore_errors=True)
    return None


# ---------------------------------------------------------------------------
# 4. Audio-Video Multiplexing & Subtitle Synchronization
# ---------------------------------------------------------------------------

def merge_video_and_narration(
    video_path: Path,
    audio_path: Path,
    final_output_path: Path,
    quality: str = DEFAULT_VIDEO_QUALITY,
) -> Path:
    """Combine rendered video and TTS narration with ffmpeg.

    When the narration outlasts the animation the closing frame is held rather than
    the clip being looped, so the viewer is not shown a restart mid-sentence.
    """
    tier = _resolve_tier(quality)
    v_dur = get_media_duration_seconds(video_path)
    a_dur = get_media_duration_seconds(audio_path)
    final_dur = a_dur + 0.5
    hold_s = max(0.0, final_dur - v_dur)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-vf", f"tpad=stop_mode=clone:stop_duration={hold_s:.2f},fps={tier['fps']:g}",
        "-c:v", "libx264",
        "-preset", tier["preset"],
        "-crf", str(tier["crf"]),
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-t", f"{final_dur:.2f}",
        str(final_output_path),
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return final_output_path


# ---------------------------------------------------------------------------
# 5. Figure Walkthrough Mode (PDF Figures or SVG Slides)
# ---------------------------------------------------------------------------

def extract_pdf_figures_and_text(pdf_path: Path) -> dict:
    """Extract page images and text segments from an academic PDF."""
    results = {"text": "", "figures": [], "page_count": 0}
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(pdf_path))
        results["page_count"] = len(pdf)

        text_parts = []
        for i, page in enumerate(pdf):
            textpage = page.get_textpage()
            text_parts.append(textpage.get_text_range())

            # Render page image as figure candidate (first 5 pages max)
            if i < 5:
                pil_image = page.render(scale=1.5).to_pil()
                img_name = f"fig_p{i+1}_{uuid.uuid4().hex[:6]}.png"
                img_path = VIDEOS_DIR / img_name
                pil_image.save(img_path)
                results["figures"].append({
                    "page": i + 1,
                    "image_url": f"/static/generated_videos/{img_name}",
                    "local_path": str(img_path),
                })

        results["text"] = "\n".join(text_parts)
    except Exception as exc:
        logger.warning("pypdfium2 extraction failed, trying pypdf: %s", exc)
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
            results["page_count"] = len(reader.pages)
            text_parts = [p.extract_text() or "" for p in reader.pages]
            results["text"] = "\n".join(text_parts)
        except Exception as e2:
            logger.error("All PDF extractors failed: %s", e2)

    return results


def render_figure_walkthrough_video(
    figures: list[dict],
    narration_audio_path: Path,
    final_output_path: Path,
    quality: str = DEFAULT_VIDEO_QUALITY,
) -> Path:
    """Assemble PDF figures into a smooth narrated walkthrough video via ffmpeg."""
    tier = _resolve_tier(quality)
    width, height = tier["resolution"]
    a_dur = get_media_duration_seconds(narration_audio_path)
    fig_count = max(1, len(figures))
    per_fig_dur = a_dur / fig_count

    # Create concat demuxer file
    concat_txt = TEMP_DIR / f"concat_{uuid.uuid4().hex[:6]}.txt"
    lines = []
    for f in figures:
        p = f.get("local_path")
        if p and Path(p).exists():
            lines.append(f"file '{Path(p).resolve().as_posix()}'")
            lines.append(f"duration {per_fig_dur:.2f}")

    if not lines:
        raise ValueError("No valid figures available for walkthrough")

    # Demuxer requires repeating the last file
    lines.append(lines[-2])
    concat_txt.write_text("\n".join(lines), encoding="utf-8")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_txt),
        "-i", str(narration_audio_path),
        "-vf", (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
            f"flags=lanczos,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps={tier['fps']:g}"
        ),
        "-c:v", "libx264",
        "-preset", tier["preset"],
        "-crf", str(tier["crf"]),
        "-profile:v", "high",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-shortest",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(final_output_path),
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    concat_txt.unlink(missing_ok=True)
    return final_output_path


# ---------------------------------------------------------------------------
# 6. High-Level Master Generator Function
# ---------------------------------------------------------------------------

# Model-authored text reaches ffmpeg inside a drawtext filter, where these characters
# would end the option or the filter chain.
_CARD_TEXT_UNSAFE = re.compile(r"['\\:%;\[\],=]")


def _card_text(value: str, limit: int) -> str:
    """Make generated title/hook text safe for an ffmpeg drawtext filter."""
    return _CARD_TEXT_UNSAFE.sub(" ", str(value or ""))[:limit].strip()


def produce_student_explainer_video(
    topic: str,
    context: str = "",
    lang: str = "tr",
    video_style: str = "manim_animation",  # "manim_animation" or "figure_walkthrough"
    duration_s: int = 50,
    voice: Optional[str] = None,
    pdf_path: Optional[Path] = None,
    backend: str = "evren",
    evren_model: str = DEFAULT_EVREN_MODEL,
    local_model_id: str = "Qwen/Qwen3.5-4B",
    video_quality: str = DEFAULT_VIDEO_QUALITY,
) -> dict:
    """Master pipeline: Concept/Context -> EVREN LLM -> Script -> TTS -> Manim/Walkthrough -> MP4 Video.

    Returns complete metadata, video URLs, subtitles, and extracted terminology.
    """
    with _RENDER_LOCK:
        start_time = time.time()
        video_id = f"explainer_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        tier = _resolve_tier(video_quality)

        # Step 1: Script & Code Generation via EVREN (degrades to local Qwen, then heuristics)
        pdf_meta = {}
        if pdf_path and pdf_path.exists():
            pdf_meta = extract_pdf_figures_and_text(pdf_path)
            if not context:
                context = pdf_meta.get("text", "")[:800]

        spec = generate_explainer_script(
            topic=topic,
            context=context,
            lang=lang,
            duration_s=duration_s,
            backend=backend,
            evren_model=evren_model,
            local_model_id=local_model_id,
        )
        used_model = evren_model if spec.get("script_backend") == "evren" else local_model_id

        # Step 2: TTS Synthesis (Scene-aware audio-first pacing)
        narration = spec.get("narration_script", "")
        audio_file, scene_durations, narration_info = synthesize_scenes_narration_audio(
            scenes=spec.get("scenes", []),
            lang=lang,
            voice=voice,
            master_output_filename=f"{video_id}.mp3",
            fallback_script=narration,
        )
        audio_duration = get_media_duration_seconds(audio_file)


        # Pace the modular scene against the measured narration durations per scene
        # so visual flow and animations perfectly match the spoken audio without freezing or stalling.
        if spec.get("manim_code_source") == "template" and audio_duration > 0:
            spec["manim_code"] = generate_modular_manim_code(
                spec=spec,
                lang=lang,
                target_duration_s=audio_duration,
                scene_durations=scene_durations,
            )


        final_mp4_path = VIDEOS_DIR / f"{video_id}.mp4"

        # Step 3: Video Rendering (Manim or Figure Walkthrough)
        if video_style == "figure_walkthrough" and pdf_meta.get("figures"):
            render_figure_walkthrough_video(
                figures=pdf_meta["figures"],
                narration_audio_path=audio_file,
                final_output_path=final_mp4_path,
                quality=video_quality,
            )
        else:
            # Default: Manim Animation Video
            raw_video = render_manim_scene(
                scene_code=spec["manim_code"],
                output_name=f"raw_{video_id}",
                quality=video_quality,
            )
            if raw_video and raw_video.exists():
                merge_video_and_narration(raw_video, audio_file, final_mp4_path, quality=video_quality)
                raw_video.unlink(missing_ok=True)
            else:
                # If Manim render could not produce output, generate dynamic title card video via ffmpeg
                width, height = tier["resolution"]
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "lavfi",
                    "-i", f"color=c=#0B0F19:s={width}x{height}:d={audio_duration:.2f}",
                    "-i", str(audio_file),
                    "-vf", (
                        f"drawtext=text='{_card_text(spec.get('title', topic), 60)}'"
                        f":fontcolor=white:fontsize={int(height * 0.055)}:x=(w-text_w)/2:y=(h-text_h)/2-50,"
                        f"drawtext=text='{_card_text(spec.get('pedagogical_hook', ''), 90)}'"
                        f":fontcolor=#38BDF8:fontsize={int(height * 0.032)}:x=(w-text_w)/2:y=(h-text_h)/2+40,"
                        f"fps={tier['fps']:g}"
                    ),
                    "-c:v", "libx264",
                    "-preset", tier["preset"],
                    "-crf", str(tier["crf"]),
                    "-c:a", "aac", "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-shortest",
                    str(final_mp4_path),
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Step 4: Subtitles & Scene Breakdown
        # The authored per-scene durations are only a plan, so the timeline is spread
        # over the real narration proportionally; otherwise a cue click seeks to a
        # timestamp that the finished video never reaches.
        planned = [max(0.1, float(s.get("duration_s", 10.0))) for s in spec.get("scenes", [])]
        plan_total = sum(planned) or 1.0
        subtitles = []
        cur_t = 0.0
        for s, planned_dur in zip(spec.get("scenes", []), planned):
            s_dur = (audio_duration + 0.5) * planned_dur / plan_total
            subtitles.append({
                "start": cur_t,
                "end": cur_t + s_dur,
                "title": s.get("title", ""),
                "text": s.get("narration", ""),
            })
            cur_t += s_dur

        elapsed = time.time() - start_time
        logger.info("Explainer video %s generated in %.2f seconds", video_id, elapsed)

        return {
            "video_id": video_id,
            "video_url": f"/static/generated_videos/{final_mp4_path.name}",
            "audio_url": f"/api/manimator/audio/{audio_file.name}" if audio_file.exists() else None,
            "title": spec.get("title", topic),
            "target_concept": spec.get("target_concept", topic),
            "pedagogical_hook": spec.get("pedagogical_hook", ""),
            "terms": spec.get("terms", []),
            "narration_script": narration,
            "scenes": spec.get("scenes", []),
            "subtitles": subtitles,
            "manim_code": spec.get("manim_code", ""),
            "duration_s": audio_duration,
            "generation_time_s": round(elapsed, 2),
            "video_style": video_style,
            "script_backend": spec.get("script_backend", "heuristic"),
            "script_backend_error": spec.get("script_backend_error"),
            "llm_model": used_model,
            "video_quality": video_quality,
            "resolution": f"{tier['resolution'][0]}x{tier['resolution'][1]}",
            "fps": tier["fps"],
            **narration_info,
        }
