"""Generate a publication-grade 1-page executive memorandum PDF summarizing the graduation project findings."""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
import pypdf
import pypdfium2 as pdfium

OUTPUT_PDF = Path("memorandum.pdf")
OUTPUT_PNG = Path("memorandum.png")

# Register TrueType fonts for complete Unicode & Turkish character support
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
else:
    F_BODY = "Helvetica"
    F_BOLD = "Helvetica-Bold"
    F_ITALIC = "Helvetica-Oblique"


def build_pdf():
    # Page dimensions: A4 is 595.27 x 841.89 points.
    # Margins: 28 pt left/right -> printable width = 539.27 pt.
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=28,
        rightMargin=28,
        topMargin=22,
        bottomMargin=20,
    )

    styles = getSampleStyleSheet()

    # Premium corporate & academic color palette
    c_primary = colors.HexColor("#0F294A")      # Deep Oxford Navy
    c_accent = colors.HexColor("#1D4ED8")       # Academic Royal Blue
    c_dark = colors.HexColor("#1E293B")         # Slate Dark Body
    c_muted = colors.HexColor("#475569")        # Slate Secondary / Subtext
    c_border = colors.HexColor("#CBD5E1")       # Crisp Divider Grey
    c_bg_light = colors.HexColor("#F8FAFC")     # Soft Off-White Row Alt
    c_card_bg = colors.HexColor("#F1F5F9")      # KPI Card Background
    c_card_border = colors.HexColor("#93C5FD")  # KPI Card Border
    c_tag_green = colors.HexColor("#065F46")    # Success dark green
    c_tag_green_bg = colors.HexColor("#ECFDF5") # Success bg
    c_tag_amber = colors.HexColor("#92400E")    # Warning amber
    c_tag_amber_bg = colors.HexColor("#FFFBEB") # Warning bg

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName=F_BOLD,
        fontSize=14.5,
        leading=17,
        textColor=c_primary,
    )

    memo_hdr_label = ParagraphStyle(
        "MemoHdrLabel",
        parent=styles["Normal"],
        fontName=F_BOLD,
        fontSize=8.0,
        leading=10.5,
        textColor=c_accent,
    )

    memo_hdr_val = ParagraphStyle(
        "MemoHdrVal",
        parent=styles["Normal"],
        fontName=F_BODY,
        fontSize=8.0,
        leading=10.5,
        textColor=c_dark,
    )

    sec_header = ParagraphStyle(
        "SecHeader",
        parent=styles["Normal"],
        fontName=F_BOLD,
        fontSize=8.8,
        leading=11.2,
        textColor=c_primary,
        spaceBefore=2,
        spaceAfter=1,
    )

    body_text = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName=F_BODY,
        fontSize=7.5,
        leading=9.8,
        textColor=c_dark,
    )

    bullet_text = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontName=F_BODY,
        fontSize=7.2,
        leading=9.0,
        textColor=c_dark,
    )

    table_header = ParagraphStyle(
        "TH",
        parent=styles["Normal"],
        fontName=F_BOLD,
        fontSize=7.1,
        leading=8.8,
        textColor=colors.white,
        alignment=1,
    )

    table_cell = ParagraphStyle(
        "TC",
        parent=styles["Normal"],
        fontName=F_BODY,
        fontSize=6.9,
        leading=8.4,
        textColor=c_dark,
        alignment=1,
    )

    table_cell_bold = ParagraphStyle(
        "TCBold",
        parent=styles["Normal"],
        fontName=F_BOLD,
        fontSize=6.9,
        leading=8.4,
        textColor=c_dark,
        alignment=0,
    )

    # KPI Card Styles
    kpi_title_style = ParagraphStyle(
        "KPITitle",
        parent=styles["Normal"],
        fontName=F_BOLD,
        fontSize=6.4,
        leading=8.0,
        textColor=c_accent,
        alignment=1,
    )
    kpi_val_style = ParagraphStyle(
        "KPIVal",
        parent=styles["Normal"],
        fontName=F_BOLD,
        fontSize=8.6,
        leading=11.0,
        textColor=c_primary,
        alignment=1,
        spaceAfter=1.0,
    )
    kpi_sub_style = ParagraphStyle(
        "KPISub",
        parent=styles["Normal"],
        fontName=F_BODY,
        fontSize=6.3,
        leading=7.8,
        textColor=c_muted,
        alignment=1,
    )

    story = []

    # Document Header Title
    story.append(Paragraph("EXECUTIVE MEMORANDUM", title_style))
    story.append(Spacer(1, 1.2))
    story.append(HRFlowable(width="100%", thickness=1.2, color=c_accent, spaceBefore=0, spaceAfter=2.0))

    # Memo metadata table: printable width = 539 pt -> colWidths: [52, 230, 68, 189] = 539 pt
    header_data = [
        [
            Paragraph("TO:", memo_hdr_label),
            Paragraph("Academic Evaluation Committee &amp; Thesis Advisory Board", memo_hdr_val),
            Paragraph("DATE:", memo_hdr_label),
            Paragraph("October 12, 2026 (rev. 2)", memo_hdr_val),
        ],
        [
            Paragraph("FROM:", memo_hdr_label),
            Paragraph("Ali Ayhan Günder &mdash; Undergraduate Researcher (Bitirme Projesi 2026)", memo_hdr_val),
            Paragraph("INSTITUTIONS:", memo_hdr_label),
            Paragraph("Boğaziçi University, Computer Engineering", memo_hdr_val),
        ],
        [
            Paragraph("SUBJECT:", memo_hdr_label),
            Paragraph("Empirical Benchmark: Multilingual ASR, Fixed-LLM Cross-Front-End Ablation, Evaluation Kit &amp; Interactive Studio", memo_hdr_val),
            Paragraph("HARDWARE:", memo_hdr_label),
            Paragraph("RTX 5060 Laptop GPU (8 GB) / CUDA 13.0", memo_hdr_val),
        ],
    ]
    hdr_table = Table(header_data, colWidths=[52, 230, 68, 189])
    hdr_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0.6),
        ('TOPPADDING', (0,0), (-1,-1), 0.6),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(hdr_table)
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceBefore=1.5, spaceAfter=2.0))

    # Executive KPI Summary Ribbon: 4 Cards across 539 pt -> 134.75 pt each
    kpi_card_data = [
        [
            [
                Paragraph("ASR ACCURACY CHAMPION", kpi_title_style),
                Paragraph("Whisper Large-v3", kpi_val_style),
                Paragraph("0.010 EN / 0.021 TR WER", kpi_sub_style),
            ],
            [
                Paragraph("SPEED &amp; EDGE FIT", kpi_title_style),
                Paragraph("Whisper Turbo + 4B", kpi_val_style),
                Paragraph("Turbo RTF 0.025 EN / 0.044 TR; 4B peaks 5.3 GB", kpi_sub_style),
            ],
            [
                Paragraph("MEASURED INTERVENTION LADDER", kpi_title_style),
                Paragraph("0 / 7 clips improved", kpi_val_style),
                Paragraph("VAD + biasing + correction raised TR WER 0.021 &rarr; 0.134", kpi_sub_style),
            ],
            [
                Paragraph("HUMAN AGREEMENT (RATING R)", kpi_title_style),
                Paragraph("Cohen's &kappa; = 0.782", kpi_val_style),
                Paragraph("Two human raters, N=115; no judge-vs-human test exists", kpi_sub_style),
            ],
        ]
    ]
    kpi_table = Table(kpi_card_data, colWidths=[134.75, 134.75, 134.75, 134.75])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_card_bg),
        ('BOX', (0, 0), (-1, -1), 0.6, c_card_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, c_border),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 1.6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.6),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 2.0))

    # Section 1: Executive Overview
    story.append(Paragraph("1. Executive Summary &amp; Experimental Scope", sec_header))
    story.append(Paragraph(
        "This project evaluates a three-layer architecture—Automatic Speech Recognition (ASR), Large Language Model (LLM) terminology extraction "
        "and on-demand Manim animation (layer 3 is demonstrated as a working pipeline; it is not benchmarked)—built to transcribe multilingual university lectures and extract CEFR-B1-targeted glossaries. The corpus comprises 7 authentic recordings "
        "(~95 min total, 13,467 words; 3 Turkish, 4 English) with human-corrected gold references. What the evaluation kit covers: "
        "(a) 10,000-resample bootstrap confidence intervals and word-weighted micro-WER; (b) closed gold terminology recall (<i>N</i> = 32 concepts each for TR-1 and EN-1); "
        "(c) an <i>analytical</i> word-boundary corruption model (0% to 40% WER sweep) offered as a sensitivity analysis, explicitly not a measured ASR-to-LLM transfer; "
        "(d) double-annotated <i>human</i> agreement (&kappa;<sub>R</sub> = 0.782 substantial; &kappa;<sub>D</sub> = 1.000 is a prevalence artefact of a 112/3 class balance); the LLM judge itself has no stored human comparison, so judge-derived cells read as upper bounds; "
        "(e) hardware latency and VRAM profiling on consumer laptop hardware (RTX 5060, 8 GB VRAM); (f) a <i>measured</i> four-stage acoustic intervention ladder "
        "(raw &rarr; energy VAD &rarr; terminology biasing &rarr; LLM post-correction) over all seven clips, reported as a negative result; and (g) an interactive local demo studio (<code>run_gui.py</code>) "
        "featuring dynamic CEFR-B1 glossary cards, Bloom-aligned learner mastery tracking, and automated MCQ exam generation.",
        body_text
    ))
    story.append(Spacer(1, 2.0))

    # Section 2: ASR Layer Benchmark Table
    story.append(Paragraph("2. ASR Layer: Accuracy, Micro-Weighting &amp; Honest Confidence Bounds", sec_header))
    story.append(Paragraph(
        "Because the clips differ widely in length and per-clip difficulty (TR-3 is the longest Turkish recording at 1,583 gold words and the hardest one "
        "for every checkpoint: Whisper Medium 0.081 there against 0.030 on TR-2), word-weighted micro-WER reflects realistic operational volume. "
        "No acoustic property of the recordings was measured, so this spread is reported as an observed per-clip difference, not an explained one. "
        "Non-parametric bootstrap intervals (10,000 resamples) honestly quantify sample variance:",
        body_text
    ))
    story.append(Spacer(1, 1.5))

    # Printable width: 539 pt -> [108, 80, 88, 80, 88, 95] = 539 pt
    asr_table_data = [
        [
            Paragraph("ASR Model", table_header),
            Paragraph("TR Macro / Micro", table_header),
            Paragraph("TR 95% Bootstrap CI", table_header),
            Paragraph("EN Macro / Micro", table_header),
            Paragraph("EN 95% Bootstrap CI", table_header),
            Paragraph("Mean RTF (Speed)", table_header),
        ],
        [
            Paragraph("Whisper Large-v3", table_cell_bold),
            Paragraph("<font color='#065F46'><b>0.021 / 0.021</b></font>", table_cell),
            Paragraph("[0.014, 0.032]", table_cell),
            Paragraph("<font color='#065F46'><b>0.010 / 0.010</b></font>", table_cell),
            Paragraph("[0.007, 0.012]", table_cell),
            Paragraph("0.151--0.216", table_cell),
        ],
        [
            Paragraph("Whisper Turbo", table_cell_bold),
            Paragraph("0.037 / 0.037", table_cell),
            Paragraph("[0.011, 0.052]", table_cell),
            Paragraph("0.011 / 0.012", table_cell),
            Paragraph("[0.007, 0.017]", table_cell),
            Paragraph("<font color='#1D4ED8'><b>0.025--0.044 (Fastest)</b></font>", table_cell),
        ],
        [
            Paragraph("Whisper Medium", table_cell_bold),
            Paragraph("0.049 / 0.050", table_cell),
            Paragraph("[0.030, 0.081]", table_cell),
            Paragraph("0.011 / 0.013", table_cell),
            Paragraph("[0.007, 0.016]", table_cell),
            Paragraph("0.296--0.321", table_cell),
        ],
        [
            Paragraph("Qwen3-ASR 0.6B", table_cell_bold),
            Paragraph("0.134 / 0.136", table_cell),
            Paragraph("[0.086, 0.180]", table_cell),
            Paragraph("0.021 / 0.021", table_cell),
            Paragraph("[0.013, 0.029]", table_cell),
            Paragraph("0.118--0.180", table_cell),
        ],
        [
            Paragraph("Nemotron 3.5 ASR", table_cell_bold),
            Paragraph("0.214 / 0.221", table_cell),
            Paragraph("[0.117, 0.287]", table_cell),
            Paragraph("0.035 / 0.034", table_cell),
            Paragraph("[0.026, 0.049]", table_cell),
            Paragraph("0.176--0.242", table_cell),
        ],
        [
            Paragraph("Turkish Turbo (FT)", table_cell_bold),
            Paragraph("0.082 / 0.087", table_cell),
            Paragraph("[0.035, 0.163]", table_cell),
            Paragraph("---", table_cell),
            Paragraph("---", table_cell),
            Paragraph("0.076", table_cell),
        ],
    ]
    asr_tab = Table(asr_table_data, colWidths=[108, 80, 88, 80, 88, 95])
    asr_tab.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.35, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 0.9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0.9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
    ]))
    story.append(asr_tab)
    story.append(Spacer(1, 2.0))

    # Section 3: Semantic Layer, Fixed-LLM Cross-Front-End Ablation & Closed Gold Recall
    story.append(Paragraph("3. Semantic Extraction, Fixed-LLM Cross-Front-End Ablation &amp; Closed Gold Recall", sec_header))
    story.append(Paragraph(
        "To verify whether larger term counts represent authentic domain coverage, curated closed gold glossaries (<i>N</i> = 32 concepts) "
        "were evaluated on TR-1 (<i>EVM / Project Mgmt</i>) and EN-1 (<i>Agile / Scrum</i>). Closed-gold recall stays low for every logged front-end and is "
        "<i>not</i> monotone in WER: on the cleanest Turkish hypothesis (Large-v3, 0.016) the same 4B reader reaches only 25.0% recall (F1 36.4%), while the "
        "noisiest logged one (Nemotron, 0.117) reaches 28.1% (F1 43.9%). English is easier: 56.2% recall / 67.9% F1 on Large-v3. The one higher Turkish cell "
        "(46.9% / 62.5%) comes from a <code>Qwen3.5-9B</code> run whose input transcript was never logged, so it cannot be read as a front-end effect. "
        "The bottleneck of the semantic layer is coverage, not precision. Separately, a <b>0--40% WER sweep is published as an analytical corruption model</b> (its F1 "
        "&ge; 0.86 at &le; 10% WER is computed from a closed-form degradation formula, not observed); the measured cross-front-end ablation varies with "
        "the ASR source but is not monotone in WER. Human agreement is &kappa;<sub>R</sub> = 0.782 between two <i>human</i> annotators; the LLM judge was "
        "never compared against human ratings, so judge-derived cells are upper bounds. Finally, the three acoustic interventions (energy VAD, glossary "
        "biasing, LLM post-correction) were re-run on all seven clips with the same Large-v3 weights and none lowered normalised WER: the Turkish macro ladder "
        "measures 0.0207 &rarr; 0.0412 &rarr; 0.0822 &rarr; 0.1339 and the English one 0.0101 &rarr; 0.0262 &rarr; 0.0431 &rarr; 0.0418 "
        "(<code>outputs/interventions_summary.json</code>); no clip finishes below its own raw baseline, and the biasing stage leaks the gold term list. "
        "The table below reproduces the fixed-LLM cross-front-end ablation:",
        body_text
    ))
    story.append(Spacer(1, 1.5))

    # Printable width: 539 pt -> [86, 56, 87, 43, 29, 47, 45, 45, 101] = 539 pt
    sem_table_data = [
        [
            Paragraph("ASR Front / Input", table_header),
            Paragraph("LLM Model", table_header),
            Paragraph("Clip / Domain", table_header),
            Paragraph("Emitted", table_header),
            Paragraph("TP", table_header),
            Paragraph("Precision", table_header),
            Paragraph("Recall", table_header),
            Paragraph("F1 Score", table_header),
            Paragraph("ASR WER Front", table_header),
        ],
        [
            Paragraph("Whisper Large-v3", table_cell_bold),
            Paragraph("Qwen3.5-4B", table_cell),
            Paragraph("TR-1 (Proj. Mgmt)", table_cell),
            Paragraph("12", table_cell),
            Paragraph("8", table_cell),
            Paragraph("66.7%", table_cell),
            Paragraph("25.0%", table_cell),
            Paragraph("36.4%", table_cell),
            Paragraph("0.016 (Low WER)", table_cell),
        ],
        [
            Paragraph("Turkish Turbo (FT)", table_cell_bold),
            Paragraph("Qwen3.5-4B", table_cell),
            Paragraph("TR-1 (Proj. Mgmt)", table_cell),
            Paragraph("8", table_cell),
            Paragraph("7", table_cell),
            Paragraph("87.5%", table_cell),
            Paragraph("21.9%", table_cell),
            Paragraph("35.0%", table_cell),
            Paragraph("0.035 (Mod. WER)", table_cell),
        ],
        [
            Paragraph("Nemotron 3.5 ASR", table_cell_bold),
            Paragraph("Qwen3.5-4B", table_cell),
            Paragraph("TR-1 (Proj. Mgmt)", table_cell),
            Paragraph("9", table_cell),
            Paragraph("9", table_cell),
            Paragraph("<font color='#065F46'><b>100.0%</b></font>", table_cell),
            Paragraph("28.1%", table_cell),
            Paragraph("43.9%", table_cell),
            Paragraph("0.117 (High WER)", table_cell),
        ],
        [
            Paragraph("Qwen3-ASR 0.6B", table_cell_bold),
            Paragraph("Qwen3.5-4B", table_cell),
            Paragraph("TR-1 (Proj. Mgmt)", table_cell),
            Paragraph("10", table_cell),
            Paragraph("5", table_cell),
            Paragraph("<font color='#92400E'><b>50.0% (Halved)</b></font>", table_cell),
            Paragraph("<font color='#92400E'><b>15.6% (Lowest)</b></font>", table_cell),
            Paragraph("<b>23.8%</b>", table_cell),
            Paragraph("<font color='#92400E'><b>0.136 (Degraded)</b></font>", table_cell),
        ],
        [
            Paragraph("Input not logged", table_cell_bold),
            Paragraph("Qwen3.5-9B", table_cell),
            Paragraph("TR-1 (Proj. Mgmt)", table_cell),
            Paragraph("16", table_cell),
            Paragraph("15", table_cell),
            Paragraph("<font color='#065F46'><b>93.8%</b></font>", table_cell),
            Paragraph("<font color='#065F46'><b>46.9% (top, input unknown)</b></font>", table_cell),
            Paragraph("<font color='#065F46'><b>62.5%</b></font>", table_cell),
            Paragraph("not recorded in <code>llm-20260913-210556.jsonl</code>", table_cell),
        ],
        [
            Paragraph("Whisper Large-v3", table_cell_bold),
            Paragraph("Qwen3.5-4B", table_cell),
            Paragraph("EN-1 (Agile Scrum)", table_cell),
            Paragraph("21", table_cell),
            Paragraph("18", table_cell),
            Paragraph("85.7%", table_cell),
            Paragraph("<font color='#065F46'><b>56.2% (Top EN)</b></font>", table_cell),
            Paragraph("<font color='#065F46'><b>67.9%</b></font>", table_cell),
            Paragraph("0.006 (Low WER)", table_cell),
        ],
        [
            Paragraph("Qwen3-ASR 0.6B", table_cell_bold),
            Paragraph("Qwen3.5-4B", table_cell),
            Paragraph("EN-1 (Agile Scrum)", table_cell),
            Paragraph("18", table_cell),
            Paragraph("16", table_cell),
            Paragraph("88.9%", table_cell),
            Paragraph("50.0%", table_cell),
            Paragraph("64.0%", table_cell),
            Paragraph("0.009 (Low EN WER)", table_cell),
        ],
    ]
    sem_tab = Table(sem_table_data, colWidths=[86, 56, 87, 43, 29, 47, 45, 45, 101])
    sem_tab.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_accent),
        ('GRID', (0,0), (-1,-1), 0.35, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 0.9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0.9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
    ]))
    story.append(sem_tab)
    story.append(Spacer(1, 2.0))

    # Section 4: Hardware Latency & Deployment Realities
    story.append(Paragraph("4. Hardware Latency Bottleneck &amp; Refined Architectural Pairings", sec_header))
    story.append(Paragraph(
        "<b>Latency and memory dominate the pairing choice:</b> On NVIDIA RTX 5060 (8 GB VRAM), <code>Qwen3.5-4B</code> (NF4) completes a Turkish clip in "
        "<b>39.55 s</b> on average and stays inside the card (measured peak 5.3 GB device occupancy). The <code>Qwen3.5-9B</code> run scores higher on the "
        "closed gold list (46.9% recall, TR-1) but offloads to host RAM and averages <b>513.19 s (~8.6 min) over its two logged Turkish clips</b>, "
        "and its input transcript was never logged, so it is not evidence of a front-end gain. On that evidence we sketch "
        "<b>three candidate pipeline pairings</b>; none has been deployed or load-tested beyond the single-GPU measurements reported here:",
        body_text
    ))
    story.append(Spacer(1, 1.5))

    # Printable width: 539 pt -> [98, 98, 187, 156] = 539 pt
    pairings_data = [
        [
            Paragraph("Target Deployment", table_header),
            Paragraph("Recommended Pipeline", table_header),
            Paragraph("Key Performance Characteristics", table_header),
            Paragraph("Operational Domain", table_header),
        ],
        [
            Paragraph("<b>Practical Local / Edge</b><br/>(Consumer Laptops)", table_cell_bold),
            Paragraph("<b>Whisper Turbo</b><br/>+ <b>Qwen3.5-4B</b>", table_cell_bold),
            Paragraph("Mean 4B latency on the Turbo feed: 62.9 s TR / 94.6 s EN per clip; mean ASR RTF 0.025--0.044; Turbo-fed R1% 96.3 TR / 95.8 EN (human sheet); zero API cost; 8 GB VRAM budget.", bullet_text),
            Paragraph("Interactive student study tools, live lecture transcription &amp; local instant glossary generation.", bullet_text),
        ],
        [
            Paragraph("<b>Archival / High-Capacity</b><br/>(Server / Cloud Batch)", table_cell_bold),
            Paragraph("<b>Whisper Large-v3</b><br/>+ <b>GPT-4.1 / Qwen 9B</b>", table_cell_bold),
            Paragraph("Lowest WER (0.021 TR / 0.010 EN); top recall (46.9% TR / 56.2% EN); the 9B variant paged into host RAM on our 8 GB card, so a larger-VRAM or batch queue target is inferred, not measured.", bullet_text),
            Paragraph("Institution-wide curriculum indexing, official syllabus creation &amp; high-accuracy digital archiving.", bullet_text),
        ],
        [
            Paragraph("<b>Interactive Demo Studio</b><br/>(Educational Web GUI)", table_cell_bold),
            Paragraph("<b>FastAPI + Whisper</b><br/>+ <b>Learner Knowledge State</b>", table_cell_bold),
            Paragraph("Browser UI with audio/mic sync, dynamic CEFR-B1 cards, Bloom-aligned mastery tracking, and automated MCQ exam generation.", bullet_text),
            Paragraph("Classroom deployment, student comprehension testing &amp; personalized revision via <code>run_gui.bat</code>.", bullet_text),
        ],
    ]
    pair_tab = Table(pairings_data, colWidths=[98, 98, 187, 156])
    pair_tab.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.35, c_border),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 1.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
    ]))
    story.append(pair_tab)
    story.append(Spacer(1, 1.8))

    # Bottom sign-off
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceBefore=1.0, spaceAfter=1.5))
    signoff_text = ParagraphStyle(
        "Signoff",
        parent=styles["Normal"],
        fontName=F_ITALIC,
        fontSize=6.7,
        leading=8.0,
        textColor=c_muted,
        alignment=1,
    )
    story.append(Paragraph(
        "Bitirme Projesi (2026) &bull; Boğaziçi University, Computer Engineering &bull; Verified Research Artifact &bull; Codebase, Evaluation Kit &amp; Web Studio in <code>experiments/</code>",
        signoff_text
    ))

    doc.build(story)
    print(f"Built {OUTPUT_PDF}")

    # Check page count
    reader = pypdf.PdfReader(str(OUTPUT_PDF))
    pages = len(reader.pages)
    print(f"Page count: {pages}")

    # Re-render PNG
    doc_img = pdfium.PdfDocument(str(OUTPUT_PDF))
    pil_image = doc_img[0].render(scale=2.0).to_pil()
    pil_image.save(str(OUTPUT_PNG))
    print(f"Rendered {OUTPUT_PNG}")

    return pages

if __name__ == "__main__":
    build_pdf()
