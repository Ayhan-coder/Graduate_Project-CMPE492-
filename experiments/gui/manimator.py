"""Manimator Integration Module for EduTerm AI Studio.

Extends the local Qwen 3.5 semantic layer to dynamic video generation:
  [PDF / Lecture Context]
        │
        ▼
   Qwen 3.5 (Local)
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
1. "Takıldığım Yeri Öğret": Instant 30-90s explainer video tailored to the
   exact moment and concept the student struggled with in the lecture.
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

_RENDER_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 1. Local Qwen 3.5 Prompt Engineering & Generation
# ---------------------------------------------------------------------------

MANIM_PROMPT_TEMPLATE = """<|im_start|>system
Sen uzman bir pedagojik video yapımcısı ve Manim (Mathematical Animation Engine) geliştiricisisin.
Görevin: Başlangıç seviyesindeki bir öğrencinin videoda/derste anlamadığı, kafasının karıştığı teknik konuyu 45-60 saniyelik harika bir animasyonlu anlatım videosuna dönüştürmektir.

Kurallar:
1. Terimler ve anlatım kesinlikle CEFR B1 düzeyinde, anlaşılır, somut ve sade olmalıdır.
2. Günlük hayattan akılda kalıcı bir benzetme (analogy) kullan.
3. Manim kodu Manim Community v0.18+ uyumlu, 'class ExplainerScene(Scene):' adında olmalıdır.
4. Manim kodunda LaTeX kurulumu gerektirmeyen 'Text' nesneleri kullan (MathTex yerine Text veya basit formüller).
5. Çıktıyı SADECE geçerli bir JSON objesi olarak ver. Markdown kod bloğu içine alma.
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
      "title": "Giriş ve Problem",
      "narration": "Bu sahnenin anlatımı...",
      "duration_s": 12.0
    }},
    {{
      "id": 2,
      "title": "Kavram ve Benzetme",
      "narration": "Benzetmenin anlatımı...",
      "duration_s": 18.0
    }},
    {{
      "id": 3,
      "title": "Kural ve Özet",
      "narration": "Sınav kuralı ve altın ipucu...",
      "duration_s": 15.0
    }}
  ],
  "manim_code": "from manim import *\\n\\nclass ExplainerScene(Scene):\\n    def construct(self):\\n        # Scene animations\\n"
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


def generate_explainer_script_qwen(
    topic: str,
    context: str = "",
    lang: str = "tr",
    duration_s: int = 50,
    model_id: str = "Qwen/Qwen3.5-4B",
    use_local_llm: bool = True,
) -> dict:
    """Generate structured terminology, narration script, and Manim code via Qwen 3.5.

    Falls back cleanly to domain-aware heuristic generator if local model
    is not loaded or execution fails.
    """
    topic_clean = topic.strip() or "Ders Kavramı"
    is_tr = str(lang).lower().startswith("tr")
    lang_name = "Türkçe" if is_tr else "English"

    if use_local_llm:
        try:
            from run_llm import generate_hf

            prompt = MANIM_PROMPT_TEMPLATE.format(
                TOPIC=topic_clean,
                CONTEXT=context[:450] if context else topic_clean,
                LANG=lang_name,
                DURATION_S=duration_s,
            )
            raw = generate_hf(model_id, prompt, max_new_tokens=650).strip()
            parsed = _clean_json_output(raw)
            if parsed and "narration_script" in parsed:
                # Ensure manim_code is present
                if not parsed.get("manim_code") or "class ExplainerScene" not in parsed.get("manim_code", ""):
                    parsed["manim_code"] = generate_default_manim_code(
                        topic=parsed.get("title", topic_clean),
                        analogy=parsed.get("pedagogical_hook", ""),
                        terms=parsed.get("terms", []),
                        lang=lang,
                    )
                return parsed
        except Exception as exc:
            logger.warning("Local Qwen 3.5 script generation fallback: %s", exc)

    # High-quality domain heuristic fallback
    return generate_fallback_explainer_spec(topic_clean, context, lang, duration_s)


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
            "Gelin duvar ustası örneğine bakalım: Planlanan Değer, yani PV, başlangıçta hedeflediğiniz tuğla bütçesidir. "
            "Kazanılmış Değer, yani EV ise ustanın gerçekten ördüğü tuğlanın değeridir. "
            "Eğer harcadığınız para yani AC, örülen tuğladan fazlaysa Maliyet Sapması negatif çıkar. "
            "Sınav altın kuralı: CV eşittir EV eksi AC. Negatif sonuç, bütçeyi aştığınızı gösterir!"
        )
        scenes = [
            {"id": 1, "title": "Problem & Pusula", "narration": "Projenizin bütçesinde ve takviminde nerede olduğunuzu nasıl anlarsınız?", "duration_s": 10.0},
            {"id": 2, "title": "Duvar Ustası Benzetmesi", "narration": "PV planlanan iş, EV tamamlanan iş, AC ise harcanan paradır.", "duration_s": 22.0},
            {"id": 3, "title": "Sınav Altın Kuralı", "narration": "CV = EV - AC formülü negatifse bütçe aşılmıştır!", "duration_s": 13.0},
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
            "Turizm, tek bir parça değil, birbirine kenetli bir zincirdir. "
            "U harfi Ulaşımı, K Konaklamayı, T Tanıtımı, ikinci K Kamu Hizmetlerini, G ise Güvenliği temsil eder. "
            "Biri bile aksarsa turist memnuniyeti çöker. Unutmayın: Turizm ürünleri stoklanamaz!"
        )
        scenes = [
            {"id": 1, "title": "Turizm Zinciri", "narration": "Turizm neden tek bir işletmeden ibaret değildir?", "duration_s": 10.0},
            {"id": 2, "title": "UKTKG Çerçevesi", "narration": "Ulaşım, Konaklama, Tanıtım, Kamu ve Güvenlik birbirine bağlıdır.", "duration_s": 20.0},
            {"id": 3, "title": "Sınav Kuralı", "narration": "Boş bir otel odası ertesi güne stoklanamaz!", "duration_s": 12.0},
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
            {"id": 1, "title": "Ham Veri Sorunu", "narration": "Ham veri neden doğrudan modele verilemez?", "duration_s": 10.0},
            {"id": 2, "title": "Rafineri Adımları", "narration": "Temizleme, ölçekleme ve dönüştürme adımları uygulanır.", "duration_s": 20.0},
            {"id": 3, "title": "Model Çıktısı", "narration": "Temiz veri, yüksek doğruluklu tahminler üretir.", "duration_s": 12.0},
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
            "Proje yöneticisi, ortaya çıkan problemlerin çözümü için önerilen alternatiflerden hangisinin proje amaçlarına daha uygun olduğuna karar vermek durumundadır. "
            "Öte yandan faaliyetler devam ederken maliyetler izlenmeli ve gerektiğinde düzeltici önlemler çok geç olmadan alınmalıdır. "
            "Unutmayın: Başarılı proje yönetimi, problemleri erkenden teşhis edip doğru alternatifi uygulamaktır!"
        )
        scenes = [
            {"id": 1, "title": "Karar Verme Sorumluluğu", "narration": "Proje yöneticisi alternatifler arasından en uygun seçeneği belirler.", "duration_s": 12.0},
            {"id": 2, "title": "İzleme ve Maliyet Takibi", "narration": "Faaliyetler sürerken maliyetler ve ilerleme düzenli izlenir.", "duration_s": 20.0},
            {"id": 3, "title": "Düzeltici Önlemler", "narration": "Sapmalar büyümeden çok geç olmadan önlem alınmalıdır.", "duration_s": 14.0},
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
            {"id": 1, "title": "İzleme Nedir?", "narration": "Performans verilerinin toplanıp raporlanması sürecidir.", "duration_s": 12.0},
            {"id": 2, "title": "Kontrol ve Sapma Tespiti", "narration": "Planlanan ile fiili durum arasındaki fark incelenir.", "duration_s": 18.0},
            {"id": 3, "title": "Düzeltici Önlemler", "narration": "Gecikme ve maliyet aşımına karşı çok geç olmadan önlem alınır.", "duration_s": 14.0},
        ]
    elif context and len(context.strip()) > 20:
        clean_ctx = context.strip()
        first_sent = clean_ctx.split(".")[0].strip() if "." in clean_ctx else clean_ctx[:100]
        title = topic_clean
        hook = f"{topic_clean} kavramı, dersin bu anında takıldığınız cümlenin mantık haritasıdır."
        terms = [
            {"term": topic_clean, "explanation": "Öğrencinin videoda takıldığı ve incelenen temel pedagojik kavram."},
            {"term": "Bağlam Analizi", "explanation": f"\"{first_sent[:80]}...\" ifadesinde geçen temel ilke."},
            {"term": "Pratik Çözüm", "explanation": "Kavramın gerçek senaryolarda ve sınav sorularında uygulanması."},
        ]
        script = (
            f"Videoda takıldığınız {topic_clean} konusunu hemen netleştirelim. "
            f"Dersin bu anında: '{first_sent}.' "
            "Burada amaç, hedeflere ulaşırken sapmaları erkenden görüp doğru kararı vermektir. "
            "Temel mantığı kavradığınızda hem dersteki akışı yakalayacak hem de sınavda rahatça uygulayacaksınız."
        )
        scenes = [
            {"id": 1, "title": f"Odak: {topic_clean}", "narration": f"Takıldığınız {topic_clean} konusunun temel sorusu.", "duration_s": 12.0},
            {"id": 2, "title": "Ders Bağlamı", "narration": f"{first_sent[:60]}...", "duration_s": 20.0},
            {"id": 3, "title": "Çözüm & Uygulama", "narration": "Kavramın pratik problem çözümündeki yansıması.", "duration_s": 14.0},
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
            {"id": 1, "title": "Giriş", "narration": f"{topic_clean} kavramının temel sorusu.", "duration_s": 10.0},
            {"id": 2, "title": "Çekirdek Mantık", "narration": "Adım adım işleyen neden-sonuç mekanizması.", "duration_s": 18.0},
            {"id": 3, "title": "Özet ve İpucu", "narration": "Akılda tutulması gereken temel nokta.", "duration_s": 12.0},
        ]

    manim_code = generate_default_manim_code(title, hook, terms, lang=lang)

    return {
        "title": title,
        "target_concept": topic_clean,
        "pedagogical_hook": hook,
        "terms": terms,
        "narration_script": script,
        "scenes": scenes,
        "manim_code": manim_code,
    }


import textwrap

def generate_default_manim_code(
    topic: str,
    analogy: str,
    terms: list[dict],
    lang: str = "tr",
) -> str:
    """Generate guaranteed error-free Manim Community Python code with full text wrapping.

    Avoids LaTeX dependency by using pure Text and VGroup geometric shapes.
    """
    clean_topic = re.sub(r'["\\]', '', topic).strip()[:50]
    title_font_size = 36 if len(clean_topic) <= 28 else (30 if len(clean_topic) <= 42 else 24)

    clean_analogy = re.sub(r'["\\]', '', analogy).strip() if analogy else "Pedagojik Anlatım"
    # Wrap analogy into clean lines without clipping words or sentences
    wrapped_analogy_lines = textwrap.wrap(clean_analogy, width=52)[:3]
    wrapped_analogy_str = "\\n".join(wrapped_analogy_lines)
    num_analogy_lines = max(1, len(wrapped_analogy_lines))
    hook_box_height = max(1.25, 0.42 * num_analogy_lines + 0.45)
    hook_font_size = 18 if num_analogy_lines <= 2 else 16

    # Terms and Descriptions wrapped cleanly
    term1 = re.sub(r'["\\]', '', terms[0].get("term", "Kavram 1")).strip()[:28] if len(terms) > 0 else "Kavram 1"
    desc1_raw = re.sub(r'["\\]', '', terms[0].get("explanation", "")).strip() if len(terms) > 0 else "Temel Tanım"
    desc1_lines = textwrap.wrap(desc1_raw, width=28)[:3]
    desc1_str = "\\n".join(desc1_lines)

    term2 = re.sub(r'["\\]', '', terms[1].get("term", "Kavram 2")).strip()[:28] if len(terms) > 1 else "Kavram 2"
    desc2_raw = re.sub(r'["\\]', '', terms[1].get("explanation", "")).strip() if len(terms) > 1 else "Pratik Kural"
    desc2_lines = textwrap.wrap(desc2_raw, width=28)[:3]
    desc2_str = "\\n".join(desc2_lines)

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
        
        self.play(Write(title), GrowFromCenter(underline), run_time=1.3)
        self.wait(0.4)

        # 2. Hook / Analogy Card with Full Multiline Text
        hook_box = RoundedRectangle(corner_radius=0.2, height={hook_box_height}, width=10.2, color="#6366F1")
        hook_box.set_fill("#1E1B4B", opacity=0.85)
        hook_box.next_to(underline, DOWN, buff=0.35)
        
        hook_text = Text("{wrapped_analogy_str}", font_size={hook_font_size}, line_spacing=0.85, color="#F8FAFC")
        hook_text.move_to(hook_box.get_center())
        
        self.play(FadeIn(hook_box), Write(hook_text), run_time=1.4)
        self.wait(0.8)

        # 3. Two Key Concept Visual Cards
        card1 = RoundedRectangle(corner_radius=0.15, height=2.6, width=4.9, color="#059669")
        card1.set_fill("#064E3B", opacity=0.8)
        card1.move_to(LEFT * 2.75 + DOWN * 1.6)

        c1_title = Text("{term1}", font_size=20, weight=BOLD, color="#34D399")
        c1_title.next_to(card1.get_top(), DOWN, buff=0.25)
        c1_desc = Text("{desc1_str}", font_size=14, line_spacing=0.8, color="#E2E8F0")
        c1_desc.next_to(c1_title, DOWN, buff=0.25)
        group1 = VGroup(card1, c1_title, c1_desc)

        card2 = RoundedRectangle(corner_radius=0.15, height=2.6, width=4.9, color="#D97706")
        card2.set_fill("#78350F", opacity=0.8)
        card2.move_to(RIGHT * 2.75 + DOWN * 1.6)

        c2_title = Text("{term2}", font_size=20, weight=BOLD, color="#FBBF24")
        c2_title.next_to(card2.get_top(), DOWN, buff=0.25)
        c2_desc = Text("{desc2_str}", font_size=14, line_spacing=0.8, color="#E2E8F0")
        c2_desc.next_to(c2_title, DOWN, buff=0.25)
        group2 = VGroup(card2, c2_title, c2_desc)

        # Arrow connecting concepts
        arrow = Arrow(card1.get_right(), card2.get_left(), buff=0.15, color="#CBD5E1", stroke_width=4)

        self.play(FadeIn(group1, shift=RIGHT), run_time=1.2)
        self.play(GrowArrow(arrow), run_time=0.8)
        self.play(FadeIn(group2, shift=LEFT), run_time=1.2)
        self.wait(1.5)

        # 4. Golden Rule Highlight
        rule_box = SurroundingRectangle(group2, color="#F59E0B", buff=0.15, stroke_width=3)
        self.play(Create(rule_box), run_time=1.0)
        self.wait(1.8)

        # 5. Outro Fade
        self.play(
            FadeOut(group1),
            FadeOut(arrow),
            FadeOut(group2),
            FadeOut(rule_box),
            FadeOut(hook_box),
            FadeOut(hook_text),
            run_time=1.0
        )
        
        final_msg = Text("Tebrikler! Konuyu Başarıyla Kavradınız.", font_size=26, color="#10B981")
        final_msg.move_to(DOWN * 0.8)
        self.play(Write(final_msg), run_time=1.2)
        self.wait(1.5)
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


def synthesize_narration_audio(
    script_text: str,
    lang: str = "tr",
    voice: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> Path:
    """Generate synchronized voiceover audio (.mp3) using edge-tts.

    Falls back to silent dummy audio if network or TTS fails.
    """
    is_tr = str(lang).lower().startswith("tr")
    selected_voice = voice or (DEFAULT_VOICE_TR if is_tr else DEFAULT_VOICE_EN)

    out_name = output_filename or f"tts_{uuid.uuid4().hex[:8]}.mp3"
    out_path = TEMP_DIR / out_name

    clean_text = script_text.strip()
    if not clean_text:
        clean_text = "Ders anlatımı tamamlandı."

    try:
        asyncio.run(_synthesize_edge_tts_async(clean_text, selected_voice, out_path))
        if out_path.exists() and out_path.stat().st_size > 500:
            return out_path
    except Exception as exc:
        logger.warning("Edge-TTS network error, attempting pyttsx3 or fallback: %s", exc)

    # Secondary fallback: pyttsx3 or synthetic silent audio
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(clean_text, str(out_path))
        engine.runAndWait()
        if out_path.exists() and out_path.stat().st_size > 200:
            return out_path
    except Exception:
        pass

    # Create 5s silent mp3 via ffmpeg if all else fails
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "5", str(out_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception as e:
        logger.error("FFmpeg silence creation failed: %s", e)

    return out_path


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


def render_manim_scene(
    scene_code: str,
    output_name: str,
    quality: str = "low_quality",  # "low_quality" (480p15, 1.2s) or "medium_quality" (720p30)
) -> Optional[Path]:
    """Execute Python Manim code and render MP4 video file.

    Features automatic error recovery / self-healing up to 2 attempts.
    """
    job_id = uuid.uuid4().hex[:8]
    work_dir = TEMP_DIR / f"job_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    current_code = scene_code
    script_path = work_dir / "scene.py"

    code_ok, reject_reason = _validate_manim_code(current_code)
    if not code_ok:
        logger.warning("Rejecting generated Manim code before execution: %s", reject_reason)

    for attempt in range(2 if code_ok else 0):
        script_path.write_text(current_code, encoding="utf-8")

        # Manim CLI rendering command
        flag = "-ql" if quality == "low_quality" else "-qm"
        cmd = [
            sys.executable,
            "-m", "manim",
            flag,
            "--media_dir", str(work_dir),
            str(script_path),
            "ExplainerScene",
        ]

        logger.info("Rendering Manim scene (attempt %d)...", attempt + 1)
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
    subprocess.run(
        [sys.executable, "-m", "manim", "-ql", "--media_dir", str(work_dir), str(script_path), "ExplainerScene"],
        capture_output=True,
    )
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
) -> Path:
    """Combine rendered video and TTS narration with ffmpeg.

    If audio is longer than video, loops or slows video smoothly to match audio duration.
    """
    v_dur = get_media_duration_seconds(video_path)
    a_dur = get_media_duration_seconds(audio_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop", "-1",  # Loop video stream if audio is longer
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-b:a", "192k",
        "-t", f"{a_dur + 0.5:.2f}",  # Match audio length + 0.5s pause
        "-shortest",
        "-pix_fmt", "yuv420p",
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
) -> Path:
    """Assemble PDF figures into a smooth 1080p narrated walkthrough video via ffmpeg."""
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
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,fps=24",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-pix_fmt", "yuv420p",
        str(final_output_path),
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    concat_txt.unlink(missing_ok=True)
    return final_output_path


# ---------------------------------------------------------------------------
# 6. High-Level Master Generator Function
# ---------------------------------------------------------------------------

def produce_student_explainer_video(
    topic: str,
    context: str = "",
    lang: str = "tr",
    video_style: str = "manim_animation",  # "manim_animation" or "figure_walkthrough"
    duration_s: int = 50,
    voice: Optional[str] = None,
    pdf_path: Optional[Path] = None,
    use_local_qwen: bool = True,
    model_id: str = "Qwen/Qwen3.5-4B",
) -> dict:
    """Master pipeline: Concept/Context -> Qwen 3.5 -> Script -> TTS -> Manim/Walkthrough -> MP4 Video.

    Returns complete metadata, video URLs, subtitles, and extracted terminology.
    """
    with _RENDER_LOCK:
        start_time = time.time()
        video_id = f"explainer_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        # Step 1: Script & Code Generation via Qwen 3.5
        pdf_meta = {}
        if pdf_path and pdf_path.exists():
            pdf_meta = extract_pdf_figures_and_text(pdf_path)
            if not context:
                context = pdf_meta.get("text", "")[:800]

        spec = generate_explainer_script_qwen(
            topic=topic,
            context=context,
            lang=lang,
            duration_s=duration_s,
            model_id=model_id,
            use_local_llm=use_local_qwen,
        )

        # Step 2: TTS Synthesis
        narration = spec.get("narration_script", "")
        audio_file = synthesize_narration_audio(
            script_text=narration,
            lang=lang,
            voice=voice,
            output_filename=f"{video_id}.mp3",
        )
        audio_duration = get_media_duration_seconds(audio_file)

        final_mp4_path = VIDEOS_DIR / f"{video_id}.mp4"

        # Step 3: Video Rendering (Manim or Figure Walkthrough)
        if video_style == "figure_walkthrough" and pdf_meta.get("figures"):
            render_figure_walkthrough_video(
                figures=pdf_meta["figures"],
                narration_audio_path=audio_file,
                final_output_path=final_mp4_path,
            )
        else:
            # Default: Manim Animation Video
            raw_video = render_manim_scene(
                scene_code=spec["manim_code"],
                output_name=f"raw_{video_id}",
                quality="low_quality",  # 480p15 for ultra-fast generation
            )
            if raw_video and raw_video.exists():
                merge_video_and_narration(raw_video, audio_file, final_mp4_path)
                raw_video.unlink(missing_ok=True)
            else:
                # If Manim render could not produce output, generate dynamic title card video via ffmpeg
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "lavfi",
                    "-i", f"color=c=#0B0F19:s=1280x720:d={audio_duration:.2f}",
                    "-i", str(audio_file),
                    "-vf", f"drawtext=text='{spec.get('title', topic)}':fontcolor=white:fontsize=44:x=(w-text_w)/2:y=(h-text_h)/2-50,drawtext=text='{spec.get('pedagogical_hook', '')[:60]}':fontcolor=#38BDF8:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2+40",
                    "-c:v", "libx264", "-c:a", "aac",
                    "-shortest",
                    str(final_mp4_path),
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Step 4: Subtitles & Scene Breakdown
        subtitles = []
        cur_t = 0.0
        for s in spec.get("scenes", []):
            s_dur = float(s.get("duration_s", 10.0))
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
        }
