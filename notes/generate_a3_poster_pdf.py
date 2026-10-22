"""Generate a publication-grade Academic Graduation Project A3 Presentation Poster PDF.

Project: Çok Dilli Eğitim İçin Konuşma Tanıma (ASR), Büyük Dil Modeli (LLM) Tabanlı
         Terminoloji Çıkarma ve Üretken Video Katmanlarının Üç Katmanlı Değerlendirilmesi
Institution: Boğaziçi Üniversitesi Bilgisayar Mühendisliği Bölümü
Author: Ali Ayhan Günder
Dimensions: ISO A3 Landscape (420 mm x 297 mm = 1190.55 pt x 841.89 pt)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT / "Bitirme_Projesi_A3_Posteri.pdf"

# Fonts: the SciFig "Classic" scheme is set in Arial; fall back to Calibri, then the
# base-14 faces, so the poster still builds on machines without the MS core fonts.
FONT_DIR = Path(r"C:\Windows\Fonts")
_FONT_SETS = [
    ("Arial", {"": "arial.ttf", "-Bold": "arialbd.ttf", "-Italic": "ariali.ttf", "-BoldItalic": "arialbi.ttf"}),
    ("Calibri", {"": "calibri.ttf", "-Bold": "calibrib.ttf", "-Italic": "calibrii.ttf", "-BoldItalic": "calibriz.ttf"}),
]
FAMILY = "Helvetica"
for _fam, _files in _FONT_SETS:
    _paths = {suffix: FONT_DIR / name for suffix, name in _files.items()}
    if all(p.exists() for p in _paths.values()):
        for suffix, path in _paths.items():
            pdfmetrics.registerFont(TTFont(_fam + suffix, str(path)))
        FAMILY = _fam
        break

if FAMILY == "Helvetica":
    F_BODY, F_BOLD, F_ITALIC, F_BOLDITALIC = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"
else:
    F_BODY, F_BOLD, F_ITALIC, F_BOLDITALIC = FAMILY, FAMILY + "-Bold", FAMILY + "-Italic", FAMILY + "-BoldItalic"
pdfmetrics.registerFontFamily(FAMILY, normal=F_BODY, bold=F_BOLD, italic=F_ITALIC, boldItalic=F_BOLDITALIC)

# ---------------------------------------------------------------------------
# SciFig open-template design tokens — "Classic" layout, Navy colour scheme.
# Source: github.com/scifig-ai/scientific-poster-and-presentation-templates
# (design scheme only; no template file is copied into this build).
# ---------------------------------------------------------------------------
NAVY_HEX, KHAKI_HEX, SLATE_HEX = "#1B365D", "#8A7B5C", "#5A5F66"
NAVY = colors.HexColor(NAVY_HEX)         # accent1 — header band, section bars, table heads
KHAKI = colors.HexColor(KHAKI_HEX)       # accent2 — second data colour, rules under the band
KHAKI_PALE = colors.HexColor("#C8B894")  # accent2 lightened for text on navy
INK = colors.HexColor("#1A1A1A")         # dk1 — body text
SLATE = colors.HexColor(SLATE_HEX)       # dk2 — secondary text
MUTED = colors.HexColor("#70767D")      # captions / footnotes
LIGHT_BG = colors.HexColor("#F4F5F7")   # lt2 — tinted boxes, alternating table rows
KHAKI_BG = colors.HexColor("#F2EEE6")   # accent2 wash for withdrawal notes
RULE = colors.HexColor("#C4C8CF")       # panel outline
RULE_LIGHT = colors.HexColor("#E8EAEE") # inner grid lines
BAND_BLUE = colors.HexColor("#A9BBD0")  # affiliation lines on navy
BAND_GREY = colors.HexColor("#C4C8CF")  # subtitle line on navy
NAVY_DEEP = colors.HexColor("#142846")  # key-results strip, one shade under the band
STRIP_INK = colors.HexColor("#DCE3EC")  # strip text on navy


BAND_H = 104   # navy header band, including the key-results strip
STRIP_H = 21   # bottom slice of the band: three headline figures
FOOT_H = 24    # footer band

S_BAR = ParagraphStyle(
    name="SectionBar",
    fontName=F_BOLD,
    fontSize=10.2,
    leading=12.6,
    textColor=colors.white,
)


def note_box(paragraph, width: float, tone: str = "navy") -> Table:
    """SciFig 'key message' callout: tinted panel with a single heavy accent rule."""
    accent, bg = (NAVY, LIGHT_BG) if tone == "navy" else (KHAKI, KHAKI_BG)
    box = Table([[paragraph]], colWidths=[width])
    box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg),
        ('LINEBEFORE', (0, 0), (0, -1), 2.6, accent),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, RULE_LIGHT),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return box


def make_card(title: str, content_flowables: list, width: float) -> Table:
    """Solid navy section bar with white uppercase title over a white body panel."""
    match = re.match(r"^(\d+)\.\s+(.*)$", title, flags=re.S)
    num, label = (match.group(1).zfill(2), match.group(2)) if match else ("", title)
    header_p = Paragraph(
        f'<font color="#C8B894">{num}</font>&nbsp;&nbsp;<b>{label.upper()}</b>', S_BAR
    )
    card = Table([[header_p], [content_flowables]], colWidths=[width])
    card.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), NAVY),
        ('BACKGROUND', (0, 1), (0, 1), colors.white),
        ('BOX', (0, 0), (-1, -1), 0.7, RULE),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (0, 0), 4.5),
        ('BOTTOMPADDING', (0, 0), (0, 0), 4.5),
        ('TOPPADDING', (0, 1), (0, 1), 5),
        ('BOTTOMPADDING', (0, 1), (0, 1), 6),
    ]))
    return card


def _fit_font_size(text: str, font: str, size: float, max_width: float, canvas_obj, min_size: float = 6.5) -> float:
    while size > min_size and canvas_obj.stringWidth(text, font, size) > max_width:
        size -= 0.5
    return size


def draw_poster_background(canvas_obj: canvas.Canvas, doc_obj: BaseDocTemplate):
    """Classic scheme: full-width navy band, centred title block, key-results strip, light footer."""
    canvas_obj.saveState()
    w, h = landscape(A3)
    cx = w / 2.0

    # Page: the Classic template prints on plain white.
    canvas_obj.setFillColor(colors.white)
    canvas_obj.rect(0, 0, w, h, fill=1, stroke=0)

    # --- Header band: centred title block over a key-results strip ---------
    band_y = h - BAND_H
    strip_y = band_y
    canvas_obj.setFillColor(NAVY)
    canvas_obj.rect(0, strip_y + STRIP_H, w, BAND_H - STRIP_H, fill=1, stroke=0)
    canvas_obj.setFillColor(NAVY_DEEP)
    canvas_obj.rect(0, strip_y, w, STRIP_H, fill=1, stroke=0)
    canvas_obj.setFillColor(KHAKI)
    canvas_obj.rect(0, strip_y + STRIP_H - 1.2, w, 1.2, fill=1, stroke=0)
    canvas_obj.rect(0, band_y - 2.5, w, 2.5, fill=1, stroke=0)

    eyebrow = "BOĞAZİÇİ ÜNİVERSİTESİ  •  BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ  •  LİSANS BİTİRME PROJESİ"
    canvas_obj.setFont(F_BOLD, 8.6)
    canvas_obj.setFillColor(BAND_BLUE)
    track = 1.6
    eyebrow_w = canvas_obj.stringWidth(eyebrow, F_BOLD, 8.6) + track * (len(eyebrow) - 1)
    eyebrow_text = canvas_obj.beginText(cx - eyebrow_w / 2.0, h - 16)
    eyebrow_text.setCharSpace(track)
    eyebrow_text.textOut(eyebrow)
    canvas_obj.drawText(eyebrow_text)
    # Tc persists in the page graphics state, so it must be cleared explicitly.
    reset_spacing = canvas_obj.beginText()
    reset_spacing.setCharSpace(0)
    canvas_obj.drawText(reset_spacing)

    title_lines = [
        "Çok Dilli Eğitim İçin ASR, LLM Tabanlı Terminoloji Çıkarma",
        "ve Üretken Video: Üç Katmanlı Değerlendirme",
    ]
    title_size = min(
        _fit_font_size(t, F_BOLD, 16.5, w - 170, canvas_obj) for t in title_lines
    )
    canvas_obj.setFont(F_BOLD, title_size)
    canvas_obj.setFillColor(colors.white)
    canvas_obj.drawCentredString(cx, h - 36, title_lines[0])
    canvas_obj.drawCentredString(cx, h - 36 - (title_size + 2.5), title_lines[1])

    canvas_obj.setFont(F_ITALIC, 9.2)
    canvas_obj.setFillColor(BAND_GREY)
    canvas_obj.drawCentredString(
        cx, h - 67,
        "A Three-Layer Evaluation: Speech-to-Text, LLM Terminology Extraction and Generative Explainer Video (EduTerm AI)",
    )
    canvas_obj.setFont(F_BODY, 8.4)
    canvas_obj.setFillColor(BAND_BLUE)
    canvas_obj.drawCentredString(
        cx, h - 79,
        "Yazar: Ali Ayhan Günder   |   Danışmanlık: Bitirme Projesi Komisyonu   |   GPU: NVIDIA RTX 5060 (CUDA 13)   |   Eylül 2026",
    )

    # Key-results strip — every figure is restated from the body cards below.
    highlights = [
        "En iyi makro WER: 0,021 TR / 0,010 EN (Whisper Large-v3)",
        "Large-v3 → Turbo hızlanması: 3,4× TR / 7,7× EN (havuzlanmış duvar saati)",
        "İki değerlendirici (N = 115): κ_R = 0,782 — belirgin uyum",
    ]
    cell_w = w / 3.0 - 26
    strip_size = min(_fit_font_size(t, F_BOLD, 8.6, cell_w, canvas_obj, 6.8) for t in highlights)
    canvas_obj.setFont(F_BOLD, strip_size)
    canvas_obj.setFillColor(STRIP_INK)
    for i, text in enumerate(highlights):
        canvas_obj.drawCentredString(w * (i + 0.5) / 3.0, strip_y + 7, text)
    canvas_obj.setStrokeColor(BAND_BLUE)
    canvas_obj.setLineWidth(0.6)
    for frac in (1 / 3.0, 2 / 3.0):
        canvas_obj.line(w * frac, strip_y + 4, w * frac, strip_y + STRIP_H - 4)

    # --- Footer band -------------------------------------------------------
    canvas_obj.setFillColor(LIGHT_BG)
    canvas_obj.rect(0, 0, w, FOOT_H, fill=1, stroke=0)
    canvas_obj.setFillColor(NAVY)
    canvas_obj.rect(0, FOOT_H, w, 1.6, fill=1, stroke=0)
    canvas_obj.setFont(F_BODY, 8)
    canvas_obj.setFillColor(INK)
    canvas_obj.drawString(36, FOOT_H - 10, "Boğaziçi Üniversitesi Lisans Bitirme Projesi Poster Sunumu  •  EduTerm AI Studio (FastAPI + Whisper + Qwen 3.5 + Manimator)")
    canvas_obj.drawRightString(w - 36, FOOT_H - 10, "Kod & Veri Seti: https://github.com/aliayhangunder/Bitirme-Projesi  •  İletişim: ali.gunder@boun.edu.tr")
    canvas_obj.setFont(F_ITALIC, 7)
    canvas_obj.setFillColor(MUTED)
    canvas_obj.drawCentredString(
        cx, FOOT_H - 19,
        "Düzen ve renk şeması SciFig AI açık poster şablonlarından (Classic / Navy) uyarlanmıştır; şablon dosyaları bu ürüne kopyalanmamıştır.",
    )

    canvas_obj.restoreState()


def build_poster_pdf():
    # If xelatex and Bitirme_Projesi_A3_Posteri.tex exist, compile the Gemini LaTeX poster
    tex_path = ROOT / "Bitirme_Projesi_A3_Posteri.tex"
    if tex_path.exists():
        import shutil, subprocess
        if shutil.which("xelatex"):
            res = subprocess.run(["xelatex", "-interaction=nonstopmode", "Bitirme_Projesi_A3_Posteri.tex"], cwd=str(ROOT), capture_output=True)
            if res.returncode == 0 and OUT_PDF.exists():
                print(f"Academic A3 Poster PDF successfully generated via XeLaTeX Gemini Template at: {OUT_PDF} ({os.path.getsize(OUT_PDF)} bytes)")
                return

    out_path = OUT_PDF
    w, h = landscape(A3)
    
    # Column Frame Geometry
    margin_x = 34
    col_gap = 16
    margin_bottom = FOOT_H + 6
    margin_top = BAND_H + 2.5 + 11  # below the band and its khaki edge rule

    col_w = (w - (2 * margin_x) - (2 * col_gap)) / 3.0
    inner_w = col_w - 18  # usable width inside a card's left/right padding
    col_h = h - margin_top - margin_bottom
    
    x1 = margin_x
    x2 = x1 + col_w + col_gap
    x3 = x2 + col_w + col_gap
    y = margin_bottom
    
    frame1 = Frame(x1, y, col_w, col_h, id='col1', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    frame2 = Frame(x2, y, col_w, col_h, id='col2', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    frame3 = Frame(x3, y, col_w, col_h, id='col3', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    
    template = PageTemplate(id='poster_layout', frames=[frame1, frame2, frame3], onPage=draw_poster_background)
    doc = BaseDocTemplate(str(out_path), pagesize=landscape(A3))
    doc.addPageTemplates([template])
    
    # Base Typography Styles
    styles = getSampleStyleSheet()
    
    s_body = ParagraphStyle(
        name='PosterBody',
        fontName=F_BODY,
        fontSize=8.1,
        leading=10.2,
        textColor=INK,
        alignment=TA_LEFT,
    )
    
    s_body_bold = ParagraphStyle(
        name='PosterBodyBold',
        parent=s_body,
        fontName=F_BOLD,
    )
    
    s_body_just = ParagraphStyle(
        name='PosterBodyJust',
        parent=s_body,
        alignment=TA_JUSTIFY,
    )
    
    s_callout = ParagraphStyle(
        name='PosterCallout',
        parent=s_body,
        fontName=F_ITALIC,
        fontSize=7.7,
        leading=9.8,
        textColor=INK,
    )
    
    s_th = ParagraphStyle(
        name='PosterTH',
        fontName=F_BOLD,
        fontSize=7.6,
        leading=9.5,
        textColor=colors.white,
        alignment=TA_CENTER,
    )
    
    s_td = ParagraphStyle(
        name='PosterTD',
        fontName=F_BODY,
        fontSize=7.3,
        leading=9.0,
        textColor=INK,
        alignment=TA_CENTER,
    )
    
    s_td_l = ParagraphStyle(
        name='PosterTDL',
        parent=s_td,
        alignment=TA_LEFT,
    )
    
    s_td_bold = ParagraphStyle(
        name='PosterTDBold',
        parent=s_td,
        fontName=F_BOLD,
    )

    flowables = []
    
    # =========================================================================
    # SÜTUN 1: Problem, Motivasyon & Sistem Mimarisi
    # =========================================================================
    
    # Kart 1.1: Motivasyon & Problem Tanımı
    c1_1 = [
        Paragraph(
            "<b>Eğitimde Bilişsel Yük ve 2-Sigma Problemi:</b> Dijital video dersler zengin bilgi sunmasına "
            "rağmen uzmanlık jargonu içermekte; başlangıç (beginner) düzeyindeki öğrenciler "
            "için terminolojik engeller yaratmaktadır. Bloom'un ünlü <b>2-Sigma Problemi</b> (1984), bire bir öğretmen "
            "rehberliğindeki öğrencilerin 2 standart sapma daha başarılı olduğunu kanıtlamıştır. Bu çalışmanın amacı; "
            "ses sinyalindeki bilgiyi çözümleyen, zorlayıcı terimleri yakalayıp <b>CEFR B1 düzeyinde</b> yalınlaştıran ve "
            "kişiselleştirilmiş animasyon videoları üreten yapay zekâ boru hattını bilimsel titizlikle inşa etmektir.",
            s_body_just,
        ),
    ]
    flowables.append(make_card("1. Motivasyon ve Problem Tanımı", c1_1, col_w))
    flowables.append(Spacer(1, 3))
    
    # Kart 1.2: Araştırma Soruları (RQs)
    c1_2 = [
        Paragraph("• <b>RQ1 (ASR Doğruluğu & Hız):</b> Çok dilli modeller (Whisper, Qwen3-ASR, Nemotron) Türkçe ve İngilizce eğitim konuşmalarında WER, CER ve RTF açısından nasıl performans sergiler?", s_body),
        Spacer(1, 2),
        Paragraph("• <b>RQ2 (LLM Semantik Kapsamı):</b> Transkriptler verildiğinde ticari (GPT-4.1) ve açık ağırlıklı modeller (Qwen 3.5), terim yakalama kapsamı ve B1 açıklama doğruluğunda nasıl ayrışır?", s_body),
        Spacer(1, 2),
        Paragraph("• <b>RQ3 (Katmanlar Arası Eşleşme):</b> Doğruluk-öncelikli ve gecikme-öncelikli kullanımda hangi ASR+LLM ikilileri seçilmelidir? Sabit bir yerel LLM farklı Whisper transkriptleriyle beslendiğinde semantik tutarlılığı korur mu?", s_body),
        Spacer(1, 2),
        Paragraph("• <b>RQ4 (Üretken Video Katmanının İşlerliği):</b> Katman 2'nin semantik durumu yerel bir modelle Manim tabanlı kısa anlatım videosuna dönüştürülebilir mi? Bu çalışmada yanıt bir <b>işlerlik kanıtıdır</b> (diskte 7 render), kıyaslama değil.", s_body),
    ]
    flowables.append(make_card("2. Temel Araştırma Soruları (RQs)", c1_2, col_w))
    flowables.append(Spacer(1, 3))
    
    # Kart 1.3: 3 Katmanlı Sistem Mimarisi
    c1_3 = [
        Paragraph(
            "<b>Katman 1 (ASR Transkripsiyon):</b> 16 kHz Mono WAV sesten sabit 30s pencerelerle ardışık metin çıkarımı. "
            "VAD parçalama ve terim koşullama deney olarak ölçüldü; ikisi de WER'i düşürmedi (Bölüm 4).<br/>"
            "<b>Katman 2 (Semantik Çıkarım - LLM):</b> Sabit yönlendirmeyle transkriptten alana özgü çok kelimeli "
            "terimlerin tespiti ve CEFR B1 düzeyinde yapılandırılmış (JSON) yalın tanımların üretimi.<br/>"
            "<b>Katman 3 (Dinamik Video Üretimi - Manimator):</b> Anlaşılmayan anda 45s bağlamı yakalayarak Qwen 3.5 "
            "ile pedagojik analoji senaryosu, Python Manim animasyonu ve Edge-TTS ile MP4 remediyesi (diskteki 7 ürün: 22,3-41,8 sn, 854x480@15).",
            s_body_just,
        ),
        Spacer(1, 4),
        # Mini Architecture Diagram Box
        Table([
            [Paragraph("<b>[Ham Ses/Video]</b>", s_td), Paragraph("→", s_td), Paragraph(f"<b>Katman 1: ASR</b><br/><font size=6.5 color='{NAVY_HEX}'>Whisper Turbo / Qwen3</font>", s_td), Paragraph("→", s_td), Paragraph(f"<b>Katman 2: LLM</b><br/><font size=6.5 color='{KHAKI_HEX}'>Qwen 3.5 / GPT-4</font>", s_td), Paragraph("→", s_td), Paragraph(f"<b>Katman 3: Video</b><br/><font size=6.5 color='{SLATE_HEX}'>Manim + Edge-TTS</font>", s_td)]
        ], colWidths=[inner_w * 0.16, inner_w * 0.05, inner_w * 0.25, inner_w * 0.05, inner_w * 0.23, inner_w * 0.05, inner_w * 0.21],
        style=[
            ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
            ('PADDING', (0, 0), (-1, -1), 3),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 0.5, RULE),
        ])
    ]
    flowables.append(make_card("3. Çok Katmanlı Sistem Mimarisi", c1_3, col_w))
    flowables.append(Spacer(1, 3))
    
    # Kart 1.4: Akustik Ön-İşleme & İyileştirmeler
    # Sayılar elle yazılmaz: experiments/outputs/interventions_summary.json'dan okunur.
    _iv_path = Path(__file__).resolve().parents[1] / "experiments" / "outputs" / "interventions_summary.json"
    _iv = json.loads(_iv_path.read_text(encoding="utf-8")) if _iv_path.exists() else None

    def _iv_cell(stage, lang, key="wer_norm_macro"):
        if not stage:
            return "—"
        cell = stage.get("per_lang", {}).get(lang) or {}
        val = cell.get(key)
        return "—" if val is None else f"{val:.4f}"

    _stages = {s["stage"][:3]: s for s in (_iv["stages"] if _iv else [])}
    _order = ["(1)", "(2)", "(3)", "(4)"]
    _labels = {
        "(1)": "(1) Ham Whisper Large-v3",
        "(2)": "(2) + Enerji Tabanlı VAD",
        "(3)": "(3) + Terim Koşullama (TR-1/EN-1)",
        "(4)": "(4) + LLM ASR Düzeltme",
    }
    _rows = [[Paragraph("Boru Hattı Yapılandırması", s_th), Paragraph("TR WER", s_th),
              Paragraph("EN WER", s_th), Paragraph("Kapsam TR / EN", s_th)]]
    for k in _order:
        st = _stages.get(k)
        cov_tr = _iv_cell(st, "tr", "gold_mention_coverage")
        cov_en = _iv_cell(st, "en", "gold_mention_coverage")
        cov_tr = cov_tr if cov_tr == "—" else f"{float(cov_tr) * 100:.1f}%"
        cov_en = cov_en if cov_en == "—" else f"{float(cov_en) * 100:.1f}%"
        _rows.append([
            Paragraph(f"<b>{_labels[k]}</b>", s_td_l),
            Paragraph(_iv_cell(st, "tr"), s_td),
            Paragraph(_iv_cell(st, "en"), s_td),
            Paragraph(f"{cov_tr} / {cov_en}", s_td),
        ])

    c1_4 = [
        Paragraph(
            "Üç akustik müdahale aynı 7 klipte, aynı Whisper Large-v3 denetim noktasında "
            "<b>gerçekten koşuldu</b>. Aşağıdaki değerler "
            "<font size=6.5>outputs/interventions_summary.json</font> artefaktından okunur; "
            "hiçbiri elle yazılmadı. Terim sütunu bir F1 skoru değil, kapalı altın listede "
            "<b>geçiş (mention coverage)</b> oranıdır.",
            s_body,
        ),
        Spacer(1, 3),
        Table(_rows, colWidths=[inner_w * 0.40, inner_w * 0.16, inner_w * 0.16, inner_w * 0.28],
              style=[
                  ('BACKGROUND', (0, 0), (-1, 0), NAVY),
                  ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
                  ('GRID', (0, 0), (-1, -1), 0.4, RULE),
                  ('PADDING', (0, 0), (-1, -1), 2.0),
                  ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
              ]),
        Spacer(1, 3),
        note_box(Paragraph(
            "<i>Olumsuz sonuç, sonuçtur: hiçbir kademe WER'i düşürmedi. Enerji VAD'i 7 klipte de "
            "kötüleştirdi ve GPU temel koşumunda sökülecek tekrar döngüsü yoktu (8-gram tekrarı 0,0122/0,0000). "
            "Sözlük koşullaması altın listeden beslendiği için kapsam artışı sızıntıdır; TR-1'de önek yinelenmesi "
            "(prompt-continuation) üretti. Eski '-%83,7 hata azalma / %74,4 Terim F1' satırları ve %26,3 gürültü "
            "stres testindeki '+%29,8 F1' iddiası diskte artefaktı olmadığı için geri çekilmiştir.</i>",
            s_callout,
        ), inner_w, "khaki"),
    ]
    flowables.append(make_card("4. Akustik Ön-İşleme ve İyileştirmeler", c1_4, col_w))
    
    flowables.append(FrameBreak())
    
    # =========================================================================
    # SÜTUN 2: Veri Kümesi, Doğrulama & Deneysel ASR Bulguları
    # =========================================================================
    
    # Kart 2.1: Veri Kümesi ve Doğrulama Protokolü
    c2_1 = [
        Paragraph(
            "<b>Çok Dilli Eğitim Külliyatı:</b> 7 video dersi (yerel WAV'lar 95,6 dk, altın referanslarda 13.467 kelime; klip klip 1.256-3.459), "
            "5 disiplin; yedisi de hazırlanmış/senaryolu, tek konuşmacılı kayıtlardır. "
            "TR-1: Proje Yönetimi (12:01), TR-2: Turizm (11:17), TR-3: Veri Bilimi (14:15), "
            "EN-1: Çevik Yöntemler (13:17), EN-2: Sürdürülebilirlik (09:16), EN-3: Üretken YZ (17:57), EN-4: Isı Transferi (17:33).<br/>"
            "'Dengeli terminolojik yük' ifadesinin nasıl ölçüldüğü kayıtlı değildir ve bu çalışmada yeniden ölçülmemiştir; küme seçimi "
            "ölçülmüş bir yoğunluk dengesine değil disiplin çeşitliliğine dayanır.",
            s_body_just,
        ),
        Spacer(1, 3),
        note_box(Paragraph(
            "<b>Akademik Dürüstlük Notu (Dataset Card):</b> Referanslar sıfırdan bağımsız transkripsiyon değildir; "
            "otomatik altyazı ve taslakların insan denetiminde terim ve noktalama düzeltmeleriyle (post-editing) "
            "iyileştirilmesidir. Jüri sunumunda Whisper lehine oluşabilecek hafif biçimsel avantaj açıkça belgelenmiştir.",
            s_callout,
        ), inner_w, "navy")
    ]
    flowables.append(make_card("5. Veri Kümesi ve Doğrulama Protokolü", c2_1, col_w))
    flowables.append(Spacer(1, 3))
    
    # Kart 2.2: ASR GPU Benchmark Tablosu
    c2_2 = [
        Paragraph("Altın referanslar üzerinde <b>RTX 5060 Laptop GPU</b> ortamında doğrulanmış sonuçlar:", s_body),
        Spacer(1, 3),
        Table([
            [Paragraph("Model", s_th), Paragraph("Diller", s_th), Paragraph("TR WER", s_th), Paragraph("EN WER", s_th), Paragraph("TR RTF", s_th), Paragraph("EN RTF", s_th)],
            [Paragraph("<b>Whisper Medium</b>", s_td_l), Paragraph("tr+en", s_td), Paragraph("0.049", s_td), Paragraph("0.011", s_td), Paragraph("0.296", s_td), Paragraph("0.321", s_td)],
            [Paragraph("<b>Whisper Large-v3</b>", s_td_l), Paragraph("tr+en", s_td), Paragraph("<b>0.021</b>", s_td_bold), Paragraph("<b>0.010</b>", s_td_bold), Paragraph("0.151", s_td), Paragraph("0.216", s_td)],
            [Paragraph("<b>Whisper Large-v3 Turbo</b>", s_td_l), Paragraph("tr+en", s_td), Paragraph("0.037", s_td), Paragraph("0.011", s_td), Paragraph("<b>0.044</b>", s_td_bold), Paragraph("<b>0.025</b>", s_td_bold)],
            [Paragraph("<b>selimc/whisper-turbo-tr</b>", s_td_l), Paragraph("tr", s_td), Paragraph("0.082", s_td), Paragraph("—", s_td), Paragraph("0.076", s_td), Paragraph("—", s_td)],
            [Paragraph("<b>Qwen3-ASR-0.6B</b>", s_td_l), Paragraph("tr+en", s_td), Paragraph("0.134", s_td), Paragraph("0.021", s_td), Paragraph("0.180", s_td), Paragraph("0.118", s_td)],
            [Paragraph("<b>Qwen3-ASR-1.7B</b>", s_td_l), Paragraph("tr+en", s_td), Paragraph("0.061", s_td), Paragraph("0.015", s_td), Paragraph("0.382", s_td), Paragraph("0.203", s_td)],
            [Paragraph("<b>NVIDIA Nemotron 3.5</b>", s_td_l), Paragraph("tr+en", s_td), Paragraph("0.214", s_td), Paragraph("0.035", s_td), Paragraph("0.242", s_td), Paragraph("0.176", s_td)],
            [Paragraph("<b>Wav2Vec 2.0 / SpeechBrain</b>", s_td_l), Paragraph("en", s_td), Paragraph("—", s_td), Paragraph("—", s_td), Paragraph("—", s_td), Paragraph("—", s_td)],
        ], colWidths=[inner_w * 0.35, inner_w * 0.11, inner_w * 0.14, inner_w * 0.14, inner_w * 0.13, inner_w * 0.13],
        style=[
            ('BACKGROUND', (0, 0), (-1, 0), NAVY),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ('GRID', (0, 0), (-1, -1), 0.4, RULE),
            ('PADDING', (0, 0), (-1, -1), 2.0),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]),
        Spacer(1, 3),
        Paragraph(
            "<b>Bulgular:</b> <b>Whisper Large-v3</b> en yüksek doğruluğu verirken (TR 0,021 / EN 0,010 makro WER), "
            "<b>Large-v3 Turbo</b> ölçülen RTF 0,044 (TR) / 0,025 (EN) ile sesi gerçek zamanın ~23-40 katı hızında işler. "
            "Aynı kartta Large-v3'e göre <b>ölçülmüş havuzlanmış duvar saati hızlanması 3,4x (TR: 110,6 sn -> 32,5 sn) ve "
            "7,7x (EN: 170,7 sn -> 22,1 sn)</b>'dir; ağırlıksız klip RTF oranları 3,4x/8,5x verir. '40 kat' bir RTF "
            "çevirisidir, Large-v3'e kıyasla ölçülmüş hızlanma değildir. Bedeli: TR WER 0,021 -> 0,037. "
            "Qwen3-ASR 1.7B açık kaynaklı güçlü bir alternatiftir. Wav2Vec 2.0 ve SpeechBrain Türkçe desteği "
            "olmadığından dürüstçe boş ('—') bırakılmıştır. TR WER'in yüksekliği dilin zorluğuna da akustik ortama da "
            "atfedilemez: yedi klipte de hiçbir akustik nicelik (yankılanma, gürültü tabanı, kanal) ölçülmedi ve "
            "DATASET_CARD yediyi de hazırlanmış, tek konuşmacılı, senaryolu eğitici videolar olarak kaydeder; klip "
            "zorluğu açıklanmış değil <b>gözlenmiş</b> bir özelliktir (n=7).",
            s_body_just,
        )
    ]
    flowables.append(make_card("6. ASR GPU Benchmark ve Hız Kıyaslaması", c2_2, col_w))
    flowables.append(Spacer(1, 3))
    
    # Kart 2.3: Katman 3 - Manimator Pedagojik Video Üretimi
    c2_3 = [
        Paragraph(
            "<b>Ders İçi Remediye Videosu ('Takıldığım Yeri Öğret'):</b> "
            "Öğrencinin video oynatılırken zorlandığı anda tıkladığı tek butonla, son 45 saniyelik transkript kesiti "
            "ve aktif terimler yakalanır. Yerel Qwen 3.5 modeli:<br/>"
            "• CEFR B1 düzeyinde kilit kavramı özetler ve günlük hayattan somut bir analoji kurar,<br/>"
            "• seslendirme senaryosu yazar (tasarım hedefi 30-90 sn; <b>ölçülen çıktı süresi 22,3-41,8 sn</b>, "
            "yedi üretimden dördü 30 sn'nin altında) ve Manim Community Python kodu üretir,<br/>"
            "• Edge-TTS nöral seslendirmesi (`tr-TR-AhmetNeural`) ve FFmpeg ile altyazılı MP4 oluşturulur "
            "(senkron ölçülmemiştir).<br/>"
            "• <b>Kendi Kendini Onaran (Self-Healing) Kod Yürütme:</b> sözdizimi/özellik hataları regex düzeltmeleri ve "
            "<code>ast</code> allow-list denetimiyle onarılır; azami 2 deneme (<code>gui/manimator.py</code>). "
            "Üretim çevrimdışı/batch'tir: 'anlık' ya da 'canlı' değildir.",
            s_body_just,
        ),
        Spacer(1, 3),
        Table([
            [Paragraph("Özellik", s_th), Paragraph("Orijinal Manimator", s_th), Paragraph("EduTerm AI Uyarlaması", s_th)],
            [Paragraph("<b>Model Bağımlılığı</b>", s_td_l), Paragraph("Ücretli Bulut API (Claude/GPT-4)", s_td), Paragraph("<b>Yerel Açık Ağırlıklı Qwen 3.5</b>", s_td_bold)],
            [Paragraph("<b>Girdi Kaynağı</b>", s_td_l), Paragraph("Yalnızca statik PDF belgesi", s_td), Paragraph("<b>Videodan kesit penceresi (t&minus;45 sn &rarr; t+10 sn)</b>", s_td_bold)],
            [Paragraph("<b>Ses ve Senkron</b>", s_td_l), Paragraph("Harici TTS / Manuel birleşim", s_td), Paragraph("<b>Nöral Edge-TTS + VTT Altyazı Eşleme</b>", s_td_bold)],
        ], colWidths=[inner_w * 0.28, inner_w * 0.36, inner_w * 0.36],
        style=[
            ('BACKGROUND', (0, 0), (-1, 0), KHAKI),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ('GRID', (0, 0), (-1, -1), 0.4, RULE),
            ('PADDING', (0, 0), (-1, -1), 2.0),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ])
    ]
    flowables.append(make_card("7. Katman 3: Manimator Dinamik Video Üretimi", c2_3, col_w))
    
    flowables.append(FrameBreak())
    
    # =========================================================================
    # SÜTUN 3: Semantik Değerlendirme, Güvenilirlik & Sonuç
    # =========================================================================
    
    # Kart 3.1: Semantik Katman ve Çapraz Ablasyon
    c3_1 = [
        Paragraph("Aynı açık ağırlıklı LLM'e (<b>Qwen 3.5-4B</b>) farklı ASR hipotezleri beslenerek semantik tutarlılık ölçülmüştür:", s_body),
        Spacer(1, 3),
        Table([
            [Paragraph("ASR Kaynağı", s_th), Paragraph("Dil", s_th), Paragraph("Terim", s_th), Paragraph("R1% (Bağlam)", s_th), Paragraph("D1% (Doğruluk)", s_th)],
            [Paragraph("<b>Whisper Medium</b>", s_td_l), Paragraph("TR / EN", s_td), Paragraph("28 / 48", s_td), Paragraph("%89.3 / %97.9", s_td), Paragraph("%92.9 / %100.0", s_td)],
            [Paragraph("<b>Whisper Large-v3</b>", s_td_l), Paragraph("TR / EN", s_td), Paragraph("31 / 51", s_td), Paragraph("<b>%90.3 / %98.0</b>", s_td_bold), Paragraph("<b>%93.5 / %100.0</b>", s_td_bold)],
            [Paragraph("<b>Whisper Turbo</b>", s_td_l), Paragraph("TR / EN", s_td), Paragraph("27 / 48", s_td), Paragraph("%96.3 / %95.8", s_td), Paragraph("%96.3 / %97.9", s_td)],
            [Paragraph("<b>Qwen3-ASR 0.6B</b>", s_td_l), Paragraph("TR / EN", s_td), Paragraph("35 / 56", s_td), Paragraph("%94.3 / %94.6", s_td), Paragraph("%100.0 / %100.0", s_td)],
            [Paragraph("<b>Nemotron 3.5</b>", s_td_l), Paragraph("TR / EN", s_td), Paragraph("29 / 51", s_td), Paragraph("%96.6 / %94.1", s_td), Paragraph("%100.0 / %96.1", s_td)],
        ], colWidths=[inner_w * 0.28, inner_w * 0.11, inner_w * 0.13, inner_w * 0.21, inner_w * 0.27],
        style=[
            ('BACKGROUND', (0, 0), (-1, 0), NAVY),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ('GRID', (0, 0), (-1, -1), 0.4, RULE),
            ('PADDING', (0, 0), (-1, -1), 2.0),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]),
        Spacer(1, 3),
        note_box(Paragraph(
            "<i>Not: R1% kapalı altın sözlük olmadığından kesinliktir (precision); tam duyarlılık (recall) ölçülmemiştir. "
            "Etiket kaynağı satır satır ayrılır: Whisper satırları 1. insan değerlendiricinin 233 çiftlik etiketi "
            "(labels_a1.csv), Qwen3-ASR ve Nemotron satırları ise yerel Qwen3.5-4B hakeminin kapalı uçlu rubrik "
            "çağrılarıdır (labels_ablation_real_judge.csv, 200 çift). Bu hakemin insana karşı yanlış-pozitif oranını "
            "ölçen bir artefakt olmadığı için son iki satır üst sınır olarak okunur; eski %88,6 / %93,1 değerleri "
            "kelime-sezgisi (mock) hakeminden geliyordu ve geri çekilmiştir.</i>",
            s_callout,
        ), inner_w, "khaki"),
    ]
    flowables.append(make_card("8. Semantik Çıkarım & Ablasyon Bulguları", c3_1, col_w))
    flowables.append(Spacer(1, 3))
    
    # Kart 3.2: Çift Değerlendirici & Cohen's Kappa Analizi
    c3_2 = [
        Paragraph(
            "<b>İki Bağımsız Değerlendirici ile Çift Etiketleme (N = 115):</b><br/>"
            "• <b>Bağlam Uyumluluğu (R):</b> %96.5 ham anlaşma (111/115), <b>κ_R = 0.782 (Belirgin Uyum / Substantial Agreement)</b>. "
            "Negatif sınıfın temsil edilmesi (8-12 negatif) ve 4 sınır anlaşmazlık sayesinde değerlendirici tarafsızlığını tescilleyen <b>asıl kilit metrik</b>tir.<br/>"
            "• <b>Açıklama Doğruluğu (D):</b> %100.0 ham anlaşma, <b>κ_D = 1.000 (Mükemmel Uyum)</b>.<br/>"
            "<i>Kappa Paradoksu (Prevalence Paradox): 112 pozitif / 3 negatif dağılımında şans eseri beklenen uyum zaten %95'tir (Pe ≈ 0.949); bu nedenle κ_D = 1.000 matematiksel bir sonuç olup asıl kanıt κ_R'dir.</i>",
            s_body_just,
        ),
    ]
    flowables.append(make_card("9. İki Bağımsız Değerlendirici & Kappa Doğrulaması", c3_2, col_w))
    flowables.append(Spacer(1, 3))
    
    # Kart 3.3: Dağıtım Önerileri & Yerel Demo Stüdyosu
    c3_3 = [
        Paragraph(
            "<b>Aday Boru Hattı Yapılandırmaları (Dağıtılmış Değil, Öneri):</b><br/>"
            "• <b>Doğruluk-Öncelikli:</b> Whisper Large-v3 (TR 0,021 / EN 0,010 makro WER) + GPT-4.1. "
            "Qwen 3.5-9B hücresi girdi dizimi <b>loglanmadığı</b> için (<code>llm-20260913-210556.jsonl</code>) "
            "bir ön uç etkisi olarak okunamaz.<br/>"
            "• <b>Hız-Öncelikli:</b> Whisper Large-v3 Turbo (ölçülen RTF 0,025 EN / 0,044 TR; havuzlanmış duvar "
            "saati hızlanması 3,4× TR / 7,7× EN) + Qwen 3.5-4B. Eşleştirme ne dağıtıldı ne yük testine sokuldu; "
            "'40×' ifadesi RTF 0,025'in gerçek zamana göre çevirisi olup Large-v3'e kıyasla ölçülmüş hızlanma 3,4×/7,7×'dir.<br/>"
            "<b>EduTerm AI Etkileşimli Demo Stüdyosu:</b> FastAPI ve Web GUI üzerinde YouTube oynatıcıyla "
            "senkronize CEFR B1 terim kartları ve Manimator video remediyesi yerel olarak çalışır durumda gösterilmiştir; "
            "üretim <b>çevrimdışı/batch</b>'tir (7 render <code>ffprobe</code> ile 22,3-41,8 sn, 854x480@15), dolayısıyla "
            "'anlık' değildir ve ses-görüntü senkronu ölçülmemiştir. "
            "Webcam göz takibi (Eye Tracker) kolu kodda mevcuttur ancak MediaPipe FaceMesh'i çalışma zamanında bir CDN'den "
            "indirir; çevrimdışı deney ortamında bu yol hiç çalıştırılmadı ve kamera erişimi olmadığında elle tıklayılan bir "
            "<b>simülasyon moduna</b> düşer. Bu nedenle göz takibi ölçülmüş ya da doğrulanmış bir özellik değil, "
            "koşumu kanıtsız kalan bir arayüz koludur.",
            s_body_just,
        ),
    ]
    flowables.append(make_card("10. Dağıtım Önerileri ve EduTerm AI Studio", c3_3, col_w))
    flowables.append(Spacer(1, 3))
    
    # Kart 3.4: Sonuç & Kilit Kaynakça
    c3_4 = [
        note_box(Paragraph(
            "<b>Sonuç:</b> Üç katmanın ilk ikisi ölçülmüştür: düşük WER rejiminde semantik tutarlılık korunuyor "
            "(R1 %90,3 TR / %98,0 EN) ve kesinlik yüksekken duyarlılık tavan yapıyor (kapalı altın listede TR %28,1). "
            "Katman 3 uçtan uca çalışır (7 video render edildi) ama kıyaslanmamıştır. Öğrenme kazanımı bu çalışmada "
            "ölçülmediği için 2-Sigma/Bloom iddiası bir tasarım motivasyonudur, ampirik bir bulgu değildir.",
            s_body_just,
        ), inner_w, "navy"),
        Spacer(1, 4),
        Paragraph(
            "<b>Kilit Kaynakça:</b> [1] Bloom, <i>Educ. Researcher</i> 1984. [2] Radford et al., <i>ICML</i> 2023. "
            "[3] Wang et al., <i>Qwen3-ASR</i> 2026. [4] Cohen, <i>Educ. Psych. Meas.</i> 1960. "
            "[5] Vanzo et al., <i>ACL</i> 2025. [6] Tran et al., <i>ACM Comput. Surv.</i> 2026.",
            s_callout,
        ),
    ]
    flowables.append(make_card("11. Sonuç ve Kaynakça", c3_4, col_w))
    
    # Build Document
    doc.build(flowables)
    print(f"Academic A3 Poster PDF successfully generated at: {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    build_poster_pdf()
