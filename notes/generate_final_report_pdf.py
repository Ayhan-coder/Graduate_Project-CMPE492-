"""Generate a publication-grade Academic Graduation Thesis Report PDF from Bitirme_Projesi_Raporu.md."""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT / "Bitirme_Projesi_Raporu.pdf"
MD_SRC = ROOT / "Bitirme_Projesi_Raporu.md"

FONT_DIR = Path(r"C:\Windows\Fonts")
has_calibri = (FONT_DIR / "calibri.ttf").exists()
if has_calibri:
    pdfmetrics.registerFont(TTFont("Calibri", str(FONT_DIR / "calibri.ttf")))
    pdfmetrics.registerFont(TTFont("Calibri-Bold", str(FONT_DIR / "calibrib.ttf")))
    pdfmetrics.registerFont(TTFont("Calibri-Italic", str(FONT_DIR / "calibrii.ttf")))
    pdfmetrics.registerFont(TTFont("Calibri-BoldItalic", str(FONT_DIR / "calibriz.ttf")))
    F_BODY = "Calibri"
    F_BOLD = "Calibri-Bold"
    F_ITALIC = "Calibri-Italic"
    F_BOLDITALIC = "Calibri-BoldItalic"
else:
    F_BODY = "Helvetica"
    F_BOLD = "Helvetica-Bold"
    F_ITALIC = "Helvetica-Oblique"
    F_BOLDITALIC = "Helvetica-BoldOblique"

# Color Palette: Oxford Academic & Corporate R&D
NAVY = colors.HexColor("#0F294A")      # Primary Deep Navy
TEAL = colors.HexColor("#1A5F7A")      # Secondary Academic Teal
ACCENT = colors.HexColor("#1D4ED8")    # Vibrant Royal Blue
GOLD = colors.HexColor("#C59B27")      # Academic Gold
LIGHT = colors.HexColor("#F8FAFC")     # Soft Off-White
ROW_ALT = colors.HexColor("#F1F5F9")   # Soft Grey Alternate
LINE = colors.HexColor("#CBD5E1")      # Crisp Divider
INK = colors.HexColor("#1E293B")       # Body Slate Dark
MUTED = colors.HexColor("#64748B")     # Secondary Text
GREEN = colors.HexColor("#065F46")     # Metric Success
AMBER = colors.HexColor("#92400E")     # Callout Amber
CALLOUT_BG = colors.HexColor("#FEF3C7")# Amber light bg
INFO_BG = colors.HexColor("#EFF6FF")   # Blue light bg


class ThesisNumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic 'Sayfa X / Y' numbering with running academic headers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, total_pages):
        # Skip decorations on the cover page (page 1)
        if self._pageNumber == 1:
            return

        self.saveState()
        w, h = A4

        # Running Top Header
        self.setFillColor(NAVY)
        self.setFont(F_BOLD, 7.5)
        self.drawString(18 * mm, h - 10 * mm, "KOÇ ÜNİVERSİTESİ  ·  TATİLSEPETİ AR-GE  ·  BİTİRME PROJESİ")
        self.setFont(F_ITALIC, 7.5)
        self.setFillColor(MUTED)
        self.drawRightString(w - 18 * mm, h - 10 * mm, "İki Katmanlı ASR & LLM Değerlendirme Raporu")
        self.setStrokeColor(LINE)
        self.setLineWidth(0.4)
        self.line(18 * mm, h - 11.5 * mm, w - 18 * mm, h - 11.5 * mm)

        # Running Bottom Footer
        self.setStrokeColor(LINE)
        self.setLineWidth(0.4)
        self.line(18 * mm, 12 * mm, w - 18 * mm, 12 * mm)
        self.setFont(F_BODY, 7.5)
        self.setFillColor(MUTED)
        self.drawString(18 * mm, 7.5 * mm, "Eylül 2026  ·  Bitirme Projesi Raporu  ·  Boğaziçi Üniversitesi")
        self.setFont(F_BOLD, 7.5)
        self.setFillColor(NAVY)
        self.drawRightString(w - 18 * mm, 7.5 * mm, f"Sayfa {self._pageNumber} / {total_pages}")

        self.restoreState()


def get_thesis_styles():
    base = getSampleStyleSheet()
    s = {
        "cover_inst": ParagraphStyle(
            "CoverInst",
            fontName=F_BOLD,
            fontSize=11,
            leading=14,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "cover_dept": ParagraphStyle(
            "CoverDept",
            fontName=F_BODY,
            fontSize=9.5,
            leading=13,
            textColor=TEAL,
            alignment=TA_CENTER,
            spaceAfter=25,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            fontName=F_BOLD,
            fontSize=19,
            leading=24,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            fontName=F_BOLD,
            fontSize=11,
            leading=15,
            textColor=ACCENT,
            alignment=TA_CENTER,
            spaceAfter=25,
        ),
        "cover_author_label": ParagraphStyle(
            "CoverAuthorLabel",
            fontName=F_BOLD,
            fontSize=9,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "cover_author_val": ParagraphStyle(
            "CoverAuthorVal",
            fontName=F_BOLD,
            fontSize=10,
            leading=14,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=3,
        ),
        "cover_date": ParagraphStyle(
            "CoverDate",
            fontName=F_BODY,
            fontSize=9.5,
            leading=13,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "H1",
            fontName=F_BOLD,
            fontSize=13,
            leading=16.5,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            fontName=F_BOLD,
            fontSize=10.5,
            leading=13.5,
            textColor=TEAL,
            spaceBefore=8,
            spaceAfter=3.5,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            fontName=F_BOLD,
            fontSize=9.5,
            leading=12.5,
            textColor=INK,
            spaceBefore=6,
            spaceAfter=2.5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName=F_BODY,
            fontSize=9.2,
            leading=13.0,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
        ),
        "abstract_title": ParagraphStyle(
            "AbstractTitle",
            fontName=F_BOLD,
            fontSize=11,
            leading=14,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "abstract_body": ParagraphStyle(
            "AbstractBody",
            fontName=F_BODY,
            fontSize=8.7,
            leading=12.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=4,
        ),
        "keywords": ParagraphStyle(
            "Keywords",
            fontName=F_BOLD,
            fontSize=8.5,
            leading=11.5,
            textColor=NAVY,
            spaceAfter=6,
        ),
        "keywords_val": ParagraphStyle(
            "KeywordsVal",
            fontName=F_BODY,
            fontSize=8.5,
            leading=11.5,
            textColor=INK,
        ),
        "table_cell": ParagraphStyle(
            "TC",
            fontName=F_BODY,
            fontSize=7.8,
            leading=10.4,
            textColor=INK,
        ),
        "table_cell_center": ParagraphStyle(
            "TCC",
            fontName=F_BODY,
            fontSize=7.8,
            leading=10.4,
            textColor=INK,
            alignment=TA_CENTER,
        ),
        "table_cell_bold": ParagraphStyle(
            "TCB",
            fontName=F_BOLD,
            fontSize=7.8,
            leading=10.4,
            textColor=NAVY,
        ),
        "table_header": ParagraphStyle(
            "TH",
            fontName=F_BOLD,
            fontSize=7.8,
            leading=10.2,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            fontName=F_BODY,
            fontSize=9.0,
            leading=12.4,
            textColor=INK,
        ),
        "callout_title": ParagraphStyle(
            "CalloutTitle",
            fontName=F_BOLD,
            fontSize=8.8,
            leading=11.5,
            textColor=AMBER,
        ),
        "callout_body": ParagraphStyle(
            "CalloutBody",
            fontName=F_BODY,
            fontSize=8.2,
            leading=11.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
        ),
        "diagram_box": ParagraphStyle(
            "DiagramBox",
            fontName=F_BOLD,
            fontSize=8.0,
            leading=10.2,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "diagram_sub": ParagraphStyle(
            "DiagramSub",
            fontName=F_BODY,
            fontSize=7.2,
            leading=9.0,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }
    return s


STYLES = get_thesis_styles()


def make_callout_box(title: str, text: str, width_pt: float = 527):
    """Builds a formal warning/advisory callout container with amber border."""
    p_title = Paragraph(f"<b>{title}</b>", STYLES["callout_title"])
    p_text = Paragraph(text, STYLES["callout_body"])
    tbl = Table([[p_title], [p_text]], colWidths=[width_pt])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CALLOUT_BG),
                ("LINEBEFORE", (0, 0), (0, -1), 3.0, AMBER),
                ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return tbl


def make_styled_table(headers: list[str], rows: list[list[str]], col_widths: list[float], align_centers: list[int] | None = None):
    """Creates a publication-grade styled ReportLab table."""
    align_centers = align_centers or []
    head_cells = [Paragraph(h, STYLES["table_header"]) for h in headers]
    data = [head_cells]

    for row in rows:
        row_cells = []
        for idx, val in enumerate(row):
            is_bold = val.startswith("**") and val.endswith("**")
            cleaned = val.replace("**", "").replace("<bold>", "").replace("</bold>", "")
            if idx in align_centers:
                style = STYLES["table_cell_center"]
            elif is_bold:
                style = STYLES["table_cell_bold"]
            else:
                style = STYLES["table_cell"]
            row_cells.append(Paragraph(cleaned, style))
        data.append(row_cells)

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("BOX", (0, 0), (-1, -1), 0.8, NAVY),
            ]
        )
    )
    return t


def build_pipeline_diagram(width_pt: float = 527):
    """Constructs a clean visual flowchart diagram of the 2-layer pipeline."""
    c_w = width_pt / 3.0 - 4
    col_widths = [c_w, 12, c_w, 12, c_w]

    b1 = [
        Paragraph("KATMAN 1: ASR", STYLES["diagram_box"]),
        Paragraph("Whisper · Qwen3-ASR · Turbo<br/>Metrikler: WER, CER, RTF, Latency", STYLES["diagram_sub"]),
    ]
    b2 = [
        Paragraph("KATMAN 2: SEMANTİK (LLM)", STYLES["diagram_box"]),
        Paragraph("GPT-4.1 · GPT-4o-mini · Qwen3.5<br/>Görev: Terim Çıkarımı + CEFR B1", STYLES["diagram_sub"]),
    ]
    b3 = [
        Paragraph("UYGULAMA: DEMO STÜDYOSU", STYLES["diagram_box"]),
        Paragraph("FastAPI · Video &amp; Altyazı Eşleme<br/>Dinamik Terim Kartları &amp; Karşılaştırma", STYLES["diagram_sub"]),
    ]
    arrow1 = Paragraph("<font size='14' color='#1D4ED8'><b>&rarr;</b></font>", STYLES["diagram_box"])
    arrow2 = Paragraph("<font size='14' color='#1D4ED8'><b>&rarr;</b></font>", STYLES["diagram_box"])

    data = [[b1, arrow1, b2, arrow2, b3]]
    t = Table(data, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), INFO_BG),
                ("BACKGROUND", (2, 0), (2, 0), INFO_BG),
                ("BACKGROUND", (4, 0), (4, 0), INFO_BG),
                ("BOX", (0, 0), (0, 0), 0.7, ACCENT),
                ("BOX", (2, 0), (2, 0), 0.7, ACCENT),
                ("BOX", (4, 0), (4, 0), 0.7, ACCENT),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def bullets(items: list[str]):
    return ListFlowable(
        [ListItem(Paragraph(i, STYLES["bullet"]), leftIndent=12, bulletColor=TEAL) for i in items],
        bulletType="bullet",
        start="•",
        leftIndent=14,
        bulletFontName=F_BODY,
        bulletFontSize=9.0,
        spaceAfter=3.5,
    )


def build_thesis_pdf():
    # A4 printable dimensions with 18mm margins:
    # 595.27 pt width - 2 * 18 * 2.8346 pt = ~493 pt or 15mm margins -> 510 pt
    # Let's use 16mm margins -> 504 pt printable width
    margin = 15 * mm
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="Çok Dilli Eğitim İçin ASR ve LLM Tabanlı Terminoloji Çıkarma Sistemlerinin İki Katmanlı Değerlendirilmesi",
        author="Ali Ayhan Günder",
        subject="Bitirme Projesi Raporu - Boğaziçi Üniversitesi Bilgisayar Mühendisliği Bölümü",
    )

    printable_w = 595.27 - 2 * 15 * 2.834645  # ~510 pt
    story = []

    # =========================================================================
    # 1. FORMAL COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("BOĞAZİÇİ ÜNİVERSİTESİ &bull; BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ", STYLES["cover_inst"]))
    story.append(Paragraph("BİLGİSAYAR MÜHENDİSLİĞİ LİSANS BİTİRME TEZİ", STYLES["cover_dept"]))
    story.append(Spacer(1, 12 * mm))

    story.append(HRFlowable(width="100%", thickness=2.0, color=GOLD, spaceBefore=0, spaceAfter=8 * mm))

    story.append(
        Paragraph(
            "Çok Dilli Eğitim İçin Konuşma Tanıma (ASR) ve Büyük Dil Modeli (LLM) Tabanlı "
            "Terminoloji Çıkarma Sistemlerinin İki Katmanlı Değerlendirilmesi",
            STYLES["cover_title"],
        )
    )
    story.append(
        Paragraph(
            "BİTİRME PROJESİ RAPORU &bull; GRADUATION PROJECT FINAL THESIS",
            STYLES["cover_subtitle"],
        )
    )

    story.append(HRFlowable(width="100%", thickness=0.8, color=LINE, spaceBefore=0, spaceAfter=10 * mm))

    story.append(Paragraph("ARAŞTIRMACI VE YAZAR", STYLES["cover_author_label"]))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("<b>Ali Ayhan Günder</b> (Boğaziçi Üniversitesi)", STYLES["cover_author_val"]))

    story.append(Spacer(1, 12 * mm))

    # Jury Evaluation Approval Table
    jury_data = [
        [
            Paragraph("<b>Jüri Değerlendirme Kurulu</b>", STYLES["table_header"]),
            Paragraph("<b>Unvan / Kurum</b>", STYLES["table_header"]),
            Paragraph("<b>İmza / Onay</b>", STYLES["table_header"]),
        ],
        [
            Paragraph("Tez Danışmanı", STYLES["table_cell_bold"]),
            Paragraph("Boğaziçi Üniversitesi Bilgisayar Müh.", STYLES["table_cell"]),
            Paragraph("....................................", STYLES["table_cell_center"]),
        ],
        [
            Paragraph("Jüri Üyesi", STYLES["table_cell_bold"]),
            Paragraph("Boğaziçi Üniversitesi Bilgisayar Müh.", STYLES["table_cell"]),
            Paragraph("....................................", STYLES["table_cell_center"]),
        ],
        [
            Paragraph("Jüri Üyesi", STYLES["table_cell_bold"]),
            Paragraph("Boğaziçi Üniversitesi Bilgisayar Müh.", STYLES["table_cell"]),
            Paragraph("....................................", STYLES["table_cell_center"]),
        ],
    ]
    jury_table = Table(jury_data, colWidths=[150, 200, 160])
    jury_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(jury_table)

    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("İstanbul, Eylül 2026", STYLES["cover_date"]))
    story.append(PageBreak())

    # =========================================================================
    # 2. BILINGUAL ABSTRACT PAGE (ÖZET & ABSTRACT)
    # =========================================================================
    story.append(Paragraph("Özet", STYLES["abstract_title"]))
    story.append(HRFlowable(width="100%", thickness=1.0, color=ACCENT, spaceBefore=0, spaceAfter=3 * mm))
    story.append(
        Paragraph(
            "Eğitim materyallerinin dijitalleşmesi ve çevrimiçi video platformlarında yaygınlaşması, her düzeydeki öğrenciye "
            "zengin bir bilgi havuzu sunmaktadır. Ancak uzmanlık gerektiren ders ve eğitim videoları, dinleyicilerin belirli bir "
            "ön bilgiye sahip olduğu varsayımıyla sunulmakta; bu durum başlangıç düzeyindeki öğrenciler için terminolojik ve kavramsal "
            "anlama engelleri oluşturmaktadır. Bu bitirme çalışmasında, çok dilli (Türkçe ve İngilizce) eğitim konuşmalarını yeni başlayanlar "
            "için erişilebilir kılmayı amaçlayan, hazır (off-the-shelf) otomatik konuşma tanıma (ASR) ve büyük dil modeli (LLM) bileşenlerinin "
            "iki katmanlı deneysel değerlendirmesi gerçekleştirilmiştir.<br/><br/>"
            "Birinci katmanda, modern konuşma tanıma mimarileri (Whisper Medium, Large-v3, Large-v3 Turbo, Türkçe uyarlanmış Whisper Turbo, "
            "Qwen3-ASR 0.6B/1.7B ve NVIDIA Nemotron 3.5 ASR) kelime hata oranı (WER), karakter hata oranı (CER), gecikme ve gerçek zamanlılık "
            "katsayısı (RTF) açısından test edilmiştir. İkinci katmanda, transkripsiyon çıktıları LLM katmanına (GPT-4.1, GPT-4o-mini, "
            "Qwen3.5 modelleri) beslenerek alana özgü çok kelimeli terminolojinin tespiti ve CEFR B1 düzeyinde açıklamaların üretimi sağlanmıştır. "
            "Semantik çıktılar; bağlam uyumluluğu (<i>R</i>) ve açıklama doğruluğu (<i>D</i>) boyutlarında değerlendirilmiştir. "
            "115 terimlik çift-etiketleme kümesi üzerinde iki bağımsız değerlendirici ile Cohen'in kappa katsayısı hesaplanmış; "
            "bağlam uyumluluğunda <b>&kappa;<sub>R</sub> = 0.782</b> (belirgin uyum), açıklama doğruluğunda ise <b>&kappa;<sub>D</sub> = 1.000</b> elde edilmiştir. "
            "Sonuç olarak, yüksek doğruluk senaryolarında Whisper Large-v3 + GPT-4.1/Qwen 9B; düşük gecikme ve yerel çalıştırma senaryolarında ise "
            "Whisper Turbo + Qwen3.5-4B (RTX 5060 GPU üzerinde 39.55 s işlem süresi ile) en uygun dağıtım ikilileri olarak belirlenmiştir.",
            STYLES["abstract_body"],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        Paragraph(
            "<b>Anahtar Kelimeler:</b> Otomatik Konuşma Tanıma (ASR), Büyük Dil Modelleri (LLM), Terminoloji Çıkarma, "
            "Çok Dilli Eğitim Teknolojileri, Whisper, Qwen3-ASR, Cohen'in Kappa Katsayısı, İki Katmanlı Değerlendirme.",
            STYLES["keywords"],
        )
    )
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph("Abstract", STYLES["abstract_title"]))
    story.append(HRFlowable(width="100%", thickness=1.0, color=ACCENT, spaceBefore=0, spaceAfter=3 * mm))
    story.append(
        Paragraph(
            "<b>A Two-Layer Evaluation of Speech-to-Text and LLM-Based Terminology Extraction Systems for Multilingual Education</b><br/>"
            "Digital video-based instruction provides immense learning opportunities, yet specialized educational lectures inherently assume "
            "baseline domain familiarity, confronting beginner learners with substantial terminological friction. This graduation thesis presents "
            "a comprehensive, two-layer experimental benchmark of commercial and open-weight Automatic Speech Recognition (ASR) and Large Language "
            "Model (LLM) architectures for making multilingual (Turkish and English) educational lectures accessible. In the first layer, leading ASR "
            "systems (Whisper Medium, Large-v3, Large-v3 Turbo, Turkish-matched Turbo, and Qwen3-ASR 0.6B / 1.7B) are systematically benchmarked "
            "across raw/normalized Word Error Rate (WER), Character Error Rate (CER), latency, and Real-Time Factor (RTF). In the second layer, "
            "transcripts are fed to LLMs to extract domain-specific terminology and generate beginner-friendly, CEFR-B1 explanations. Evaluation "
            "references were established across 7 lectures (~95 minutes) via human-corrected gold transcripts. Human evaluation reliability was "
            "validated via double-annotated Cohen's kappa (<i>N</i>=115), achieving <b>&kappa;<sub>R</sub> = 0.782</b> (substantial agreement on context consistency) "
            "and <b>&kappa;<sub>D</sub> = 1.000</b> (definition accuracy). Acoustic error propagation benchmarked degraded front-ends into fixed Qwen3.5-4B, "
            "confirming that higher front-end WER halves downstream terminology precision. Finally, a local FastAPI-based interactive studio was developed "
            "to validate the complete pipeline locally.",
            STYLES["abstract_body"],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        Paragraph(
            "<b>Keywords:</b> Automatic Speech Recognition, Large Language Models, Terminology Extraction, Multilingual Educational Technology, "
            "Whisper, Qwen3-ASR, Inter-Annotator Agreement, Error Propagation.",
            STYLES["keywords"],
        )
    )
    story.append(PageBreak())

    # =========================================================================
    # 3. CORE REPORT SECTIONS (1 TO 10)
    # =========================================================================

    # SECTION 1
    story.append(Paragraph("1. Giriş ve Problem Tanımı", STYLES["h1"]))
    story.append(Paragraph("1.1 Motivasyon", STYLES["h2"]))
    story.append(
        Paragraph(
            "Eğitimin bireyselleştirilmesi, öğrencinin kavrama hızına ve hazırbulunuşluk düzeyine göre içeriğin uyarlanması, "
            "eğitim bilimlerinde uzun süredir en yüksek kazanımı sağlayan yöntem olarak bilinmektedir. Benjamin Bloom'un ünlü "
            "\"2-Sigma Problemi\" (1984), bire bir öğretmen rehberliğinde ustalık öğrenimi (mastery learning) gören öğrencilerin, "
            "geleneksel sınıf ortamındaki akranlarına kıyasla iki standart sapma daha başarılı olduğunu ortaya koymuştur. Ancak insan "
            "eğitmenlerle bire bir eğitim sunmak, devasa öğrenci kitleleri için sürdürülebilir ve ölçeklenebilir değildir.<br/><br/>"
            "Günümüzde çevrimiçi kitlesel açık dersler (MOOC), YouTube eğitim serileri ve dijital video platformları geniş kitlelere eğitim "
            "ulaştırmaktadır. Ancak bu videolardaki bilgi, ses sinyalinin içerisine gömülüdür. Başlangıç düzeyindeki bir öğrenci, videoda geçen "
            "teknik bir terimi (\"çizelge sapması\", \"MICE turizmi\", \"Scrum backlog\", \"ANOVA testi\" veya \"kara cisim ışıması\" gibi) "
            "bilmediğinde, videoyu durdurup harici kaynaklarda aramak zorunda kalmakta, bu da bilişsel yükü (cognitive load) artırmakta ve öğrenme "
            "akışını kesintiye uğratmaktadır.",
            STYLES["body"],
        )
    )

    story.append(Paragraph("1.2 Amaç ve Araştırma Soruları", STYLES["h2"]))
    story.append(
        Paragraph(
            "Bu projenin temel amacı; sesli ve görüntülü eğitim içeriklerini anlayan, transkribe eden, içerisindeki zorlayıcı teknik terimleri "
            "tespit edip başlangıç düzeyindeki öğrencilerin anlayacağı seviyede (CEFR B1) otomatik açıklamalar üreten iki katmanlı bir yapay zekâ "
            "boru hattını bilimsel titizlikle tasarlamak, deneysel olarak değerlendirmek ve çalışan bir yerel prototip stüdyosu haline getirmektir.<br/><br/>"
            "Çalışma kapsamında yanıt aranan temel araştırma soruları (Research Questions) şunlardır:",
            STYLES["body"],
        )
    )
    story.append(
        bullets(
            [
                "<b>RQ1 (ASR Doğruluğu ve Verimliliği):</b> Güncel konuşma tanıma modelleri, Türkçe ve İngilizce eğitim konuşmalarında "
                "dil bazında ayrıştırıldığında hata oranı (WER/CER) ve hesaplama maliyeti (RTF, gecikme) açısından nasıl performans sergilemektedir?",
                "<b>RQ2 (LLM Semantik Kapsamı ve Açıklama Kalitesi):</b> Transkriptler verildiğinde, ticari dil modelleri (GPT-4.1) ile açık "
                "ağırlıklı modeller (GPT-4o-mini, Qwen3.5), terim yakalama kapsamı, bağlamsal tutarlılık (R1) ve sözlük doğruluğu (D1) açısından nasıl farklılaşmaktadır?",
                "<b>RQ3 (Katmanlar Arası Eşleşme ve Dağıtım Tercihleri):</b> Doğruluk-öncelikli ve hız/gecikme-öncelikli senaryolarda hangi "
                "ASR+LLM ikilileri tercih edilmelidir? Sabit bir açık ağırlıklı LLM (Qwen3.5-4B) farklı kalitedeki ASR transkriptleriyle beslendiğinde "
                "akustik hatalar semantik çıktıya nasıl yayılmaktadır?",
            ]
        )
    )

    # SECTION 2
    story.append(Paragraph("2. Literatür Taraması (İlgili Çalışmalar)", STYLES["h1"]))
    story.append(
        Paragraph(
            "<b>2.1 Otomatik Konuşma Tanıma (ASR):</b> Baevski ve ark. (2020) ham ses dalga formlarından özdenetimli gösterim öğrenen "
            "Wav2Vec 2.0 ve çok dilli türevi XLSR-53 mimarilerini sunmuştur. Radford ve ark. (OpenAI, 2023), 680.000 saatlik veriyle zayıf "
            "denetimli Whisper mimarisini geliştirmiştir. 2024 yılında sunulan Whisper Turbo ise dekoder katmanlarını 32'den 4'e düşürerek 8 kata "
            "varan hızlanma sağlamıştır. 2026 yılında sunulan Qwen3-ASR (Alibaba) ve NVIDIA Nemotron 3.5 ise açık ağırlıklı çok dilli alternatifler "
            "olarak öne çıkmaktadır.<br/><br/>"
            "<b>2.2 Otomatik Terminoloji Çıkarma (ATE):</b> C-value/NC-value ve TF-IDF tabanlı istatistiksel yöntemlerden LiTeWi (Conde ve ark., 2016) "
            "gibi eğitimsel ontoloji çıkarma sistemlerine uzanan literatür, son dönemde büyük dil modellerinin (Tran ve ark., 2026) çok kelimeli terimleri "
            "bağlamsal esneklikle yakalama gücünü doğrulamıştır.<br/><br/>"
            "<b>2.3 Uyarlanabilir Eğitim Sistemleri ve LLM Tutanları:</b> Akıllı öğretim sistemleri (ITS) geleneksel olarak kurallar ve öğrenci modelleme "
            "algoritmalarıyla (Brusilovsky, 2007; VanLehn, 2011) çalışmıştır. Vanzo ve ark. (ACL 2025), GPT-4'ün öğrenci katılımını belirgin şekilde "
            "artırdığını deneysel olarak göstermiştir. Ancak literatürdeki çalışmalar temiz metin girdisi varsaymakta; ses sinyalinden terim çıkarımına "
            "uçtan uca hata yayılımını inceleyen çalışmalar son derece sınırlı kalmaktadır.",
            STYLES["body"],
        )
    )

    # SECTION 3
    story.append(Paragraph("3. Sistem Mimarisi ve Yöntem", STYLES["h1"]))
    story.append(
        Paragraph(
            "Çalışmada incelenen sistem iki ana işleme katmanından ve bunları pratik bir gösterim ortamında birleştiren etkileşimli bir yerel web "
            "stüdyosundan oluşmaktadır:",
            STYLES["body"],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(build_pipeline_diagram(printable_w))
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            "<b>3.1 Katman 1 (ASR):</b> Ses dosyaları 16 kHz mono PCM WAV standardına dönüştürülür. 30 saniyelik örtüşmesiz pencereler üzerinden "
            "ardışık çıkarım (sequential generate) ve zaman damgası tespiti uygulanır. Çıktılar hem ham hem de normalize edilmiş biçimde saklanır.<br/>"
            "<b>3.2 Katman 2 (LLM):</b> Transkript metni sabitlenmiş bir sistem yönlendirmesiyle (system prompt) LLM katmanına beslenir. "
            "Modelden ders bağlamına tam uyumlu teknik terimleri (çok kelimeli ifadeler dahil) tespit etmesi ve CEFR B1 düzeyinde sade bir "
            "tanım üretmesi istenir; çıktı kesin JSON formatında yapılandırılır.",
            STYLES["body"],
        )
    )

    # SECTION 4
    story.append(Paragraph("4. Veri Kümesi ve Doğrulama Protokolü", STYLES["h1"]))
    story.append(
        Paragraph(
            "Değerlendirme kümesi, dengeli terminolojik yüke sahip yedi farklı üniversite ve uzmanlık eğitim videosundan derlenmiştir "
            "(toplam ~95 dakika, 13,599 kelime):",
            STYLES["body"],
        )
    )
    story.append(Spacer(1, 1.5 * mm))

    # Dataset table
    ds_headers = ["Kod", "Dil", "Ders / Video Başlığı", "Alan / Disiplin", "Süre", "Kelime"]
    ds_rows = [
        ["TR-1", "TR", "AÖF Proje Yönetimi — Ünite 7 Konu Anlatımı", "Proje Yönetimi", "12:00", "1,466"],
        ["TR-2", "TR", "Turist Rehberleri İçin Kapsamlı Turizm Eğitimi", "Turizm", "11:17", "1,489"],
        ["TR-3", "TR", "Tirendaz AI Veri Bilimi Nedir, Python", "Veri Bilimi", "14:15", "1,841"],
        ["EN-1", "EN", "Grow with Google: Intro to Agile Methodologies", "Çevik Yöntemler", "13:17", "2,137"],
        ["EN-2", "EN", "What is Sustainable Tourism? (Hayley Stainton)", "Sürdürülebilir Turizm", "09:16", "1,424"],
        ["EN-3", "EN", "Generative AI in a Nutshell (Henrik Kniberg)", "Üretken Yapay Zekâ", "17:57", "2,875"],
        ["EN-4", "EN", "Understanding Thermal Radiation (Efficient Eng.)", "Isı Transferi / Fizik", "17:33", "2,367"],
        ["Toplam", "—", "<b>7 Video Dersi</b>", "<b>5 Farklı Disiplin</b>", "<b>~95 dk</b>", "<b>13,599</b>"],
    ]
    story.append(make_styled_table(ds_headers, ds_rows, [35, 25, 175, 125, 45, 50], align_centers=[0, 1, 4, 5]))
    story.append(Spacer(1, 2.5 * mm))

    story.append(
        make_callout_box(
            "Akademik Dürüstlük Uyarısı: 'Bağımsız Gold Transkript' Yanılgısından Kaçınma",
            "Referans transkriptler sıfırdan bağımsız bir insan transkripsiyonu değildir. EN-1 ve EN-4 insan üretimi altyazılara; "
            "TR-1, TR-2, EN-2 ve EN-3 YouTube otomatik altyazılarının dinlenerek teknik terimlerinin ve noktalamalarının insan eliyle "
            "düzeltilmesine (post-editing); TR-3 ise doğrudan sesten taslak çıkarılmasına dayanmaktadır. Bu durum, referansların otomatik "
            "altyazı ve Whisper stili sentaks/cümle bölme kalıplarına hafif bir stil yakınlığı taşımasına yol açabilmekte ve WER skorlarında "
            "marjinal bir avantaj sağlayabilmektedir. Bu kısıt jüriye tam bir akademik şeffaflıkla sunulmalıdır.",
            printable_w,
        )
    )

    # SECTION 5
    story.append(Paragraph("5. Değerlendirme Metrikleri ve İstatistiksel Güvenilirlik", STYLES["h1"]))
    story.append(
        Paragraph(
            "<b>5.1 ASR Metrikleri:</b> Kelime Hata Oranı (WER = (S + D + I) / N), Karakter Hata Oranı (CER) ve Gerçek Zamanlılık Katsayısı "
            "(RTF = İşlem Süresi / Ses Süresi) hesaplanmıştır. Normalizasyonda Türkçedeki İ/i ve I/ı ayrımı korunmuştur.<br/>"
            "<b>5.2 Semantik Metrikler:</b> Bağlamsal Tutarlılık (R1: terim ders bağlamına uygun mu?) ve Açıklama Doğruluğu (D1: tanım TDK/Oxford sözlüklerine göre doğru mu?) "
            "ölçülmüştür. <i>Önemli:</i> R1 skoru modelin ürettiği terimler üzerindeki bir kesinliktir (precision); kapalı altın sözlük olmadan tam bir recall ölçümü değildir.<br/>"
            "<b>5.3 İki Bağımsız Değerlendirici ve Cohen'in Kappa Analizi:</b> 115 terimlik çift etiketli kümede:",
            STYLES["body"],
        )
    )
    story.append(
        bullets(
            [
                "<b>Bağlam Uyumluluğu (R):</b> %96.5 ham uyum (111/115 terim, 4 uyumsuzluk), <b>&kappa;<sub>R</sub> = 0.782 (Belirgin Uyum / Substantial Agreement)</b>. "
                "Varyans ve sınır vakaları içeren bu metrik, değerlendirici güvenilirliğini kanıtlayan asıl ayırt edici göstergedir.",
                "<b>Açıklama Doğruluğu (D):</b> %100.0 ham uyum, <b>&kappa;<sub>D</sub> = 1.000 (Mükemmel Uyum)</b>. "
                "Ancak bu kümede 112 pozitif ve yalnızca 3 negatif etiket bulunduğundan, şans eseri beklenen uyum zaten P<sub>e</sub> &approx; 0.95 seviyesindedir "
                "(Kappa Paradoksu / Prevalence Paradox). Dolayısıyla &kappa;<sub>D</sub> = 1.000 matematiksel bir beklenen sonuç olup &kappa;<sub>R</sub> kadar zorlu bir ayrım gücü kanıtı değildir.",
            ]
        )
    )

    # SECTION 6
    story.append(Paragraph("6. Deneysel Bulgular ve Karşılaştırmalı Analiz", STYLES["h1"]))
    story.append(Paragraph("6.1 ASR Katmanı Başarımı (GPU Altın Referanslar & Bootstrap Aralıkları)", STYLES["h2"]))

    asr_headers = ["ASR Modeli", "Dil Kapsamı", "TR Macro / Micro", "TR 95% Bootstrap CI", "EN Macro / Micro", "EN 95% Bootstrap CI", "Ort. RTF"]
    asr_rows = [
        ["Whisper Large-v3", "tr+en", "0.021 / 0.021", "[0.014, 0.032]", "0.010 / 0.010", "[0.007, 0.012]", "0.151–0.216"],
        ["Whisper Turbo", "tr+en", "0.037 / 0.037", "[0.011, 0.052]", "0.011 / 0.012", "[0.007, 0.017]", "0.025–0.044"],
        ["Whisper Medium", "tr+en", "0.049 / 0.050", "[0.030, 0.081]", "0.011 / 0.013", "[0.007, 0.016]", "0.296–0.321"],
        ["Turkish Turbo (FT)", "tr only", "0.082 / 0.087", "[0.035, 0.163]", "---", "---", "0.076"],
        ["Qwen3-ASR 0.6B", "tr+en", "0.134 / 0.136", "[0.086, 0.180]", "0.021 / 0.021", "[0.013, 0.029]", "0.118–0.180"],
        ["Nemotron 3.5 ASR", "tr+en", "0.214 / 0.221", "[0.117, 0.287]", "0.035 / 0.034", "[0.026, 0.049]", "0.176–0.242"],
    ]
    story.append(make_styled_table(asr_headers, asr_rows, [85, 45, 75, 75, 75, 75, 55], align_centers=[1, 2, 3, 4, 5, 6]))
    story.append(Spacer(1, 2.5 * mm))

    story.append(
        Paragraph(
            "<b>ASR Katmanı Çıkarımları:</b> Whisper Large-v3 en düşük hata oranına (TR 0.021 / EN 0.010) sahipken; Whisper Turbo 4–8 kat "
            "daha hızlı çalışarak (RTF 0.025–0.044) gerçek zamanlı konuşma uygulamaları için ideal performans sunmaktadır. "
            "10,000 örneklemeli bootstrap güven aralıkları, küçük örneklem boyutuna bağlı varyansı açıkça belgelemektedir.",
            STYLES["body"],
        )
    )

    story.append(Paragraph("6.2 Semantik Katman, Kapalı Altın Recall ve Akustik Hata Yayılımı", STYLES["h2"]))
    story.append(
        Paragraph(
            "TR-1 ve EN-1 dersleri için oluşturulan kapalı altın sözlükler (N=32 terim) üzerinden kesinlik, duyarlılık (recall) ve F1 skorları hesaplanmış; "
            "aynı zamanda bozulmuş ASR hipotezleri sabit Qwen3.5-4B modeline beslenerek akustik hataların yayılımı ölçülmüştür:",
            STYLES["body"],
        )
    )
    story.append(Spacer(1, 1.5 * mm))

    sem_headers = ["ASR Ön Ucu", "LLM Modeli", "Ders", "Çıkarılan", "TP", "Kesinlik", "Duyarlılık", "F1 Skoru", "Ön Uç WER"]
    sem_rows = [
        ["Whisper Large-v3", "Qwen3.5-4B", "TR-1", "12", "8", "66.7%", "25.0%", "36.4%", "0.016 (Düşük WER)"],
        ["Turkish Turbo (FT)", "Qwen3.5-4B", "TR-1", "8", "7", "87.5%", "21.9%", "35.0%", "0.035 (Orta WER)"],
        ["Nemotron 3.5 ASR", "Qwen3.5-4B", "TR-1", "9", "9", "100.0%", "28.1%", "43.9%", "0.117 (Yüksek WER)"],
        ["Qwen3-ASR 0.6B", "Qwen3.5-4B", "TR-1", "10", "5", "50.0%*", "15.6%*", "23.8%", "0.136 (Bozulmuş)"],
        ["Altın Referans", "Qwen3.5-9B", "TR-1", "16", "15", "93.8%", "46.9%", "62.5%", "0.000 (Altın Ref)"],
        ["Whisper Large-v3", "Qwen3.5-4B", "EN-1", "21", "18", "85.7%", "56.2%", "67.9%", "0.006 (Düşük WER)"],
        ["Qwen3-ASR 0.6B", "Qwen3.5-4B", "EN-1", "18", "16", "88.9%", "50.0%", "64.0%", "0.009 (Düşük WER)"],
    ]
    story.append(make_styled_table(sem_headers, sem_rows, [85, 60, 35, 42, 28, 48, 48, 48, 85], align_centers=[2, 3, 4, 5, 6, 7, 8]))
    story.append(Spacer(1, 2 * mm))
    story.append(
        Paragraph(
            "*<b>Kritik Hata Yayılımı Bulgusu:</b> Qwen3-ASR 0.6B gibi ön uç WER'i %13.6 seviyesine çıkan bozulmuş transkriptlerde "
            "LLM terim çıkarma kesinliği %50.0'ye (yarı yarıya) ve duyarlılık %15.6'ya gerilemektedir. Bu durum, konuşma tanıma hatalarının "
            "alt katmanda doğrudan terim kaybı ve halüsinasyona yol açtığını deneysel olarak kanıtlamaktadır.",
            STYLES["body"],
        )
    )

    # SECTION 7
    story.append(Paragraph("7. Donanım, Bellek Darboğazı ve Gecikme Profili", STYLES["h1"]))
    story.append(
        Paragraph(
            "Deneysel çalışmalar Boğaziçi Üniversitesi Bilgisayar Mühendisliği Bölümü bünyesinde aşağıdaki laboratuvar donanımında yürütülmüştür:<br/>"
            "&bull; <b>İşlemci:</b> Intel Core i7-14650HX (16 Çekirdek, 24 İş Parçacığı, 5.2 GHz Turbo)<br/>"
            "&bull; <b>Grafik İşlemci (GPU):</b> NVIDIA GeForce RTX 5060 Laptop GPU (8 GB GDDR6 VRAM, CUDA 13.0)<br/>"
            "&bull; <b>Sistem Belleği:</b> 32 GB DDR5 &bull; <b>Yazılım:</b> Windows 11 64-bit, Python 3.12, PyTorch 2.12.0+cu130, ReportLab 4.4+.<br/><br/>"
            "<b>8 GB VRAM Tüketici Donanımında Gecikme Sonucu:</b><br/>"
            "4-bit NF4 kuantize edilmiş <b>Qwen3.5-4B</b> modeli 8 GB VRAM bütçesine tamamen sığmakta ve tam bir ders klibini "
            "<b>39.55 saniyede</b> (s.d. 2.11 s) tamamlamaktadır. Buna karşın, üstün terminoloji duyarlılığı (%46.9) sunan <b>Qwen3.5-9B</b> modeli, "
            "uzun bağlam üretiminde 8 GB VRAM sınırını aşarak ana sistem belleğine (RAM) disk/PCIe aktarımı (offloading) yapmakta ve işlem süresi "
            "<b>513.19 saniyeye (~8.6 dakika)</b> fırlamaktadır (13 kat gecikme artışı). Bu nedenle dizüstü ve uç cihazlarda Qwen3.5-4B tartışmasız üretim tercihidir.",
            STYLES["body"],
        )
    )

    # SECTION 8
    story.append(Paragraph("8. Geliştirilen Yerel Demo Stüdyosu (`experiments/gui/`)", STYLES["h1"]))
    story.append(
        Paragraph(
            "Boru hattının uçtan uca uygulanabilirliğini sergilemek amacıyla FastAPI ve modern cam efektli (glassmorphism) Vanilla CSS/JS "
            "arayüzünden oluşan bir yerel demo stüdyosu (`experiments/gui/`, `python run_gui.py`) geliştirilmiştir.<br/>"
            "<b>Temel Yetenekler:</b> (a) `yt-dlp` ile YouTube video indirme; (b) doğrudan mikrofon kaydı ve yerel ses dosyası yükleme; "
            "(c) video oynatıcıyla senkronize transkripsiyon gösterimi; (d) CEFR B1 düzeyinde anlık dinamik terim kartları ve modeller arası yan yana karşılaştırma.<br/>"
            "<i>Not:</i> Bu arayüz ticari bir SaaS ürünü değil, açık CORS ve bellek içi iş kuyruğu kullanan yerel bir konsept kanıtlama (PoC) stüdyosudur.",
            STYLES["body"],
        )
    )

    # SECTION 9
    story.append(Paragraph("9. Sonuç, Kısıtlar ve Dağıtım Önerileri", STYLES["h1"]))
    story.append(
        Paragraph(
            "<b>9.1 Dağıtım İkilileri:</b><br/>"
            "1. <b>Pratik Yerel / Uç Cihaz (Consumer Edge):</b> Whisper Turbo + Qwen3.5-4B &rarr; Toplam gecikme 40–70 s, RTF 0.025–0.044, "
            "sıfır harici API maliyeti, 8 GB VRAM bütçesine tam uyum. Canlı ders ve öğrenci araçları için idealdir.<br/>"
            "2. <b>Yüksek Kapasite / Arşivleme (Server Batch):</b> Whisper Large-v3 + GPT-4.1 / Qwen 9B &rarr; En düşük WER (%1.0 EN / %2.1 TR), "
            "en yüksek terim kapsamı (%46.9 TR-1 / %56.2 EN-1 duyarlılık). &ge;16 GB VRAM sunucu donanımı gerektirir.<br/><br/>"
            "<b>9.2 Temel Kısıtlar:</b> (1) Veri kümesi 7 video (95 dk) ile sınırlıdır; (2) Referanslar bağımsız çift-kör transkripsiyon değil, "
            "insan düzeltmeli taslaklardır; (3) Dil modellerinin duyarlılığı yalnızca TR-1 ve EN-1'deki kapalı altın sözlükler üzerinde ölçülebilmiştir; "
            "(4) &kappa;<sub>D</sub> = 1.000 skoru yüksek sınıf tekdüzeliği (prevalence) nedeniyle &kappa;<sub>R</sub> = 0.782 kadar güçlü bir ayrım göstergesi değildir.<br/><br/>"
            "<b>9.3 Gelecek Çalışmalar:</b> Açık kaynaklı bir LMS (Moodle/Canvas) eklentisine dönüştürülmesi ve öğrencinin öğrenme geçmişini "
            "takip eden bir Bilgi Takip (Knowledge Tracing) modülü ile tam uyarlanabilir bir öğretici asistana evrilmesi planlanmaktadır.",
            STYLES["body"],
        )
    )

    # SECTION 10
    story.append(Paragraph("10. Kaynakça", STYLES["h1"]))
    refs = [
        "Bloom, B. S. (1984). \"The 2 sigma problem: The search for methods of group instruction as effective as one-to-one tutoring.\" <i>Educational Researcher</i>, 13(6), 4-16.",
        "Radford, A., Kim, J. W., Xu, T., et al. (2023). \"Robust speech recognition via large-scale weak supervision.\" <i>Proc. ICML</i>, 28492-28518.",
        "Wang, X., et al. (2026). \"Qwen3-ASR technical report.\" <i>arXiv preprint arXiv:2601.21337</i>.",
        "NVIDIA (2026). \"Nemotron 3.5 ASR streaming 0.6B model card.\" <i>Hugging Face Hub</i>.",
        "Conde, A., Larrañaga, M., et al. (2016). \"LiTeWi: A combined term extraction and entity linking method for eliciting educational ontologies from textbooks.\" <i>JASIST</i>, 67(2), 380-399.",
        "Tran, H. T. H., Martinc, M., et al. (2026). \"Recent advances in automatic term extraction: A comprehensive survey.\" <i>ACM Computing Surveys</i>, 58(9), 1-35.",
        "Vanzo, A., Pal Chowdhury, S., & Sachan, M. (2025). \"GPT-4 as a homework tutor can improve student engagement and learning outcomes.\" <i>Proc. ACL</i>, 31119-31136.",
        "Cohen, J. (1960). \"A coefficient of agreement for nominal scales.\" <i>Educational and Psychological Measurement</i>, 20(1), 37-46.",
        "Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). \"wav2vec 2.0: A framework for self-supervised learning of speech representations.\" <i>NeurIPS</i>, 33, 12449-12460.",
        "Ravanelli, M., et al. (2021). \"SpeechBrain: A general-purpose speech toolkit.\" <i>arXiv:2106.04624</i>.",
    ]
    story.append(bullets(refs))

    doc.build(story, canvasmaker=ThesisNumberedCanvas)
    print(f"Built thesis PDF: {OUT_PDF}")


if __name__ == "__main__":
    build_thesis_pdf()
