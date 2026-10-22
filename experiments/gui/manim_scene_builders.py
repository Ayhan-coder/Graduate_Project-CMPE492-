"""Modular Parametric Scene Builders for Manimator AI Video Generation.

This module provides deterministic, error-free Manim Community v0.21+ scene builders
for educational concept scaffolding:
  1. DAG / Concept Graph (`dag_concept`)
  2. Process Flow / Pipeline (`process_flow`)
  3. Comparison Table (`comparison_table`)
  4. Equation & Rule Analysis (`equation_step`)
  5. Concept Card / Pedagogical Takeaway (`concept_card`)

Each builder receives structured parameters and timing budgets, producing safe,
reproducible Manim Python code with automatic text wrapping, dynamic element counts,
and dark-mode aesthetics.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, List, Optional


def safe_text(val: Any, limit: int = 60) -> str:
    """Sanitize strings for inclusion in Manim Text mobjects."""
    s = str(val or "").strip()
    s = re.sub(r'["\\]', "", s)
    s = s.replace("\n", " ").strip()
    return s[:limit]


def safe_latex(formula: str) -> str:
    """Sanitize LaTeX formula string preserving math syntax (slashes, brackets, exponents)."""
    f = str(formula or "").strip()
    f = re.sub(r"[\r\n]+", " ", f)
    f = f.replace('"', "'")
    return f[:140]


def latex_to_unicode_math(formula: str) -> str:
    """Convert common LaTeX math commands to readable Unicode symbols for Text fallback."""
    replacements = [
        (r"\cdot", "·"),
        (r"\times", "×"),
        (r"\nabla", "∇"),
        (r"\partial", "∂"),
        (r"\sum", "∑"),
        (r"\infty", "∞"),
        (r"\approx", "≈"),
        (r"\neq", "≠"),
        (r"\leq", "≤"),
        (r"\geq", "≥"),
        (r"\pm", "±"),
        (r"\in", "∈"),
        (r"\alpha", "α"),
        (r"\beta", "β"),
        (r"\gamma", "γ"),
        (r"\delta", "δ"),
        (r"\epsilon", "ε"),
        (r"\eta", "η"),
        (r"\theta", "θ"),
        (r"\lambda", "λ"),
        (r"\mu", "μ"),
        (r"\pi", "π"),
        (r"\sigma", "σ"),
        (r"\omega", "ω"),
        (r"\Delta", "Δ"),
    ]
    s = str(formula or "").strip()
    for k, v in replacements:
        s = s.replace(k, v)
    # Remove braces around subscripts and superscripts
    s = re.sub(r"([_~^])\{([^}]+)\}", r"\1\2", s)
    # Remove leftover backslashes and double quotes
    s = re.sub(r'["\\]', "", s)
    s = s.replace("\n", " ").strip()
    return s[:70]


def wrap_text_lines(text: str, width: int = 40, max_lines: int = 4) -> str:
    """Wrap text to prevent canvas overflow, using clean \\n joins."""
    cleaned = safe_text(text, limit=width * max_lines + 20)
    lines = textwrap.wrap(cleaned, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rsplit(" ", 1)[0].rstrip(",.;:!?") + "…"
    return "\\n".join(lines) if lines else ""


def detect_scene_type(scene_data: Dict[str, Any], default: str = "concept_card") -> str:
    """Heuristically infer scene_type if LLM did not provide one."""
    explicit = str(scene_data.get("scene_type") or "").strip().lower()
    valid_types = {"dag_concept", "process_flow", "comparison_table", "equation_step", "concept_card"}
    if explicit in valid_types:
        # Guard: if equation_step was declared but no formula exists, fallback to concept_card
        if explicit == "equation_step" and not scene_data.get("formula"):
            # Check if narration or title has a formula-like equation
            combined = f"{scene_data.get('title', '')} {scene_data.get('narration', '')}"
            if not any(sym in combined for sym in ["=", "\\cdot", "\\times", " - ", " + ", "/"]):
                return "concept_card"
        return explicit

    title = str(scene_data.get("title") or "").lower()
    narration = str(scene_data.get("narration") or "").lower()
    combined = f"{title} {narration}"

    # Only match equation if formula keywords or math symbols are explicitly present (never 'kural')
    if any(k in combined for k in ["formül", "formul", "equation", "denklem", "özdeşlik", "ozdeslik"]) or (
        "=" in combined and any(c.isupper() for c in combined)
    ):
        return "equation_step"
    if any(k in combined for k in ["karşılaştır", "fark", "vs", "versus", "kıyas", "tablo", "avantaj"]):
        return "comparison_table"
    if any(k in combined for k in ["adım", "aşama", "süreç", "akış", "pipeline", "sıra", "algoritma"]):
        return "process_flow"
    if any(k in combined for k in ["önkoşul", "bağlantı", "hiyerarşi", "dag", "ağaç", "graf"]):
        return "dag_concept"

    return default


def build_dag_scene_code(
    topic: str,
    nodes: List[Dict[str, str]],
    edges: Optional[List[tuple]] = None,
    duration_s: float = 12.0,
    lang: str = "tr",
) -> str:
    """Generate Manim code showing a concept DAG / hierarchy with directed arrows."""
    clean_topic = safe_text(topic, 45)
    raw_nodes = nodes if nodes and len(nodes) >= 2 else [
        {"name": "Önkoşul Bilgi", "desc": "Temel zemin"},
        {"name": clean_topic, "desc": "Hedef kavram"},
        {"name": "Uygulama Alanı", "desc": "Pratik yansıma"},
    ]
    node_items = raw_nodes[:3]

    root_title = safe_text(node_items[0].get("name", "Önkoşul"), 26)
    root_desc = wrap_text_lines(node_items[0].get("desc", "Temel kavram"), 24, 2)

    child1_title = safe_text(node_items[1].get("name", clean_topic), 26)
    child1_desc = wrap_text_lines(node_items[1].get("desc", "Hedef konu"), 24, 2)

    has_third = len(node_items) > 2
    child2_title = safe_text(node_items[2].get("name", "Uygulama"), 26) if has_third else ""
    child2_desc = wrap_text_lines(node_items[2].get("desc", "Sonuç"), 24, 2) if has_third else ""

    # Exact timing calculation: total animations ~ 4.5s
    anim_time = 4.5
    hold_s = max(0.5, duration_s - anim_time)

    third_node_code = ""
    third_anim_code = ""
    third_cleanup = ""
    if has_third:
        third_node_code = f'''
        c2_box = RoundedRectangle(corner_radius=0.18, height=1.6, width=3.8, color="#F59E0B")
        c2_box.set_fill("#78350F", opacity=0.9)
        c2_box.move_to(DOWN * 1.5 + RIGHT * 2.6)
        c2_t = Text("{child2_title}", font_size=19, weight=BOLD, color="#FDE68A")
        c2_t.next_to(c2_box.get_top(), DOWN, buff=0.18)
        c2_d = Text("{child2_desc}", font_size=14, color="#E2E8F0")
        c2_d.next_to(c2_t, DOWN, buff=0.12)
        c2_grp = VGroup(c2_box, c2_t, c2_d)
        arrow2 = Arrow(root_box.get_bottom(), c2_box.get_top(), buff=0.15, color="#A5B4FC", stroke_width=3.5)
        '''
        third_anim_code = "GrowArrow(arrow2), FadeIn(c2_grp, shift=LEFT),"
        third_cleanup = ", c2_grp, arrow2"

    c1_x = "-2.6" if has_third else "0.0"

    code = f'''
        # --- Sahne: DAG / Kavram Hiyerarşisi ---
        dag_title = Text("{clean_topic}: Kavram Bağıntıları", font_size=28, weight=BOLD, color="#38BDF8")
        dag_title.to_edge(UP, buff=0.45)
        self.play(FadeIn(dag_title, shift=DOWN), run_time=0.8)

        # Root Node (Prerequisite)
        root_box = RoundedRectangle(corner_radius=0.18, height=1.6, width=4.0, color="#818CF8")
        root_box.set_fill("#1E1B4B", opacity=0.9)
        root_box.move_to(UP * 1.0)
        root_t = Text("{root_title}", font_size=19, weight=BOLD, color="#C7D2FE")
        root_t.next_to(root_box.get_top(), DOWN, buff=0.18)
        root_d = Text("{root_desc}", font_size=14, color="#E2E8F0")
        root_d.next_to(root_t, DOWN, buff=0.12)
        root_grp = VGroup(root_box, root_t, root_d)
        self.play(FadeIn(root_grp, shift=UP), run_time=0.9)

        # Child Node 1
        c1_box = RoundedRectangle(corner_radius=0.18, height=1.6, width=3.8, color="#34D399")
        c1_box.set_fill("#064E3B", opacity=0.9)
        c1_box.move_to(DOWN * 1.5 + RIGHT * {c1_x})
        c1_t = Text("{child1_title}", font_size=19, weight=BOLD, color="#A7F3D0")
        c1_t.next_to(c1_box.get_top(), DOWN, buff=0.18)
        c1_d = Text("{child1_desc}", font_size=14, color="#E2E8F0")
        c1_d.next_to(c1_t, DOWN, buff=0.12)
        c1_grp = VGroup(c1_box, c1_t, c1_d)

        arrow1 = Arrow(root_box.get_bottom(), c1_box.get_top(), buff=0.15, color="#A5B4FC", stroke_width=3.5)
{third_node_code}
        self.play(GrowArrow(arrow1), FadeIn(c1_grp, shift=RIGHT), {third_anim_code} run_time=1.2)

        # Highlight target concept
        focus_box = SurroundingRectangle(c1_grp, color="#38BDF8", buff=0.1, stroke_width=2.5)
        self.play(Create(focus_box), run_time=0.8)
        self.wait({hold_s:.2f})

        self.play(FadeOut(VGroup(dag_title, root_grp, c1_grp, arrow1, focus_box{third_cleanup})), run_time=0.8)
'''
    return code


def build_process_flow_code(
    topic: str,
    steps: List[Dict[str, str]],
    duration_s: float = 14.0,
    lang: str = "tr",
) -> str:
    """Generate Manim code showing a multi-step sequential process or algorithm pipeline."""
    clean_topic = safe_text(topic, 45)
    default_steps = [
        {"title": "1. Girdi / Tespit", "desc": "Gerekli verilerin toplanması"},
        {"title": "2. İşlem / Dönüşüm", "desc": "Kural ve algoritmanın işletilmesi"},
        {"title": "3. Çıktı / Doğrulama", "desc": "Nihai pedagojik sonuç"},
    ]
    raw_steps = steps if steps and len(steps) >= 2 else default_steps
    step_items = raw_steps[:3]

    s1_t = safe_text(step_items[0].get("title", "1. Girdi"), 22)
    s1_d = wrap_text_lines(step_items[0].get("desc", "Aşama detayı"), 18, 2)

    s2_t = safe_text(step_items[1].get("title", "2. İşlem"), 22)
    s2_d = wrap_text_lines(step_items[1].get("desc", "Aşama detayı"), 18, 2)

    has_s3 = len(step_items) > 2
    s3_t = safe_text(step_items[2].get("title", "3. Çıktı"), 22) if has_s3 else ""
    s3_d = wrap_text_lines(step_items[2].get("desc", "Sonuç"), 18, 2) if has_s3 else ""

    # Animation timing
    anim_time = 4.2 if has_s3 else 3.4
    hold_s = max(0.5, duration_s - anim_time)

    if has_s3:
        step_setup = f'''
        b1 = RoundedRectangle(corner_radius=0.15, height=2.4, width=3.3, color="#0284C7")
        b1.set_fill("#0C4A6E", opacity=0.9).move_to(LEFT * 4.0 + DOWN * 0.4)
        t1 = Text("{s1_t}", font_size=18, weight=BOLD, color="#7DD3FC").next_to(b1.get_top(), DOWN, buff=0.22)
        d1 = Text("{s1_d}", font_size=14, color="#F0F9FF").next_to(t1, DOWN, buff=0.18)
        step1 = VGroup(b1, t1, d1)

        b2 = RoundedRectangle(corner_radius=0.15, height=2.4, width=3.3, color="#6366F1")
        b2.set_fill("#312E81", opacity=0.9).move_to(DOWN * 0.4)
        t2 = Text("{s2_t}", font_size=18, weight=BOLD, color="#A5B4FC").next_to(b2.get_top(), DOWN, buff=0.22)
        d2 = Text("{s2_d}", font_size=14, color="#EEF2FF").next_to(t2, DOWN, buff=0.18)
        step2 = VGroup(b2, t2, d2)

        b3 = RoundedRectangle(corner_radius=0.15, height=2.4, width=3.3, color="#059669")
        b3.set_fill("#064E3B", opacity=0.9).move_to(RIGHT * 4.0 + DOWN * 0.4)
        t3 = Text("{s3_t}", font_size=18, weight=BOLD, color="#6EE7B7").next_to(b3.get_top(), DOWN, buff=0.22)
        d3 = Text("{s3_d}", font_size=14, color="#ECFDF5").next_to(t3, DOWN, buff=0.18)
        step3 = VGroup(b3, t3, d3)

        arr1 = Arrow(b1.get_right(), b2.get_left(), buff=0.12, color="#94A3B8", stroke_width=4)
        arr2 = Arrow(b2.get_right(), b3.get_left(), buff=0.12, color="#94A3B8", stroke_width=4)

        self.play(FadeIn(step1, shift=RIGHT), run_time=0.8)
        self.play(GrowArrow(arr1), FadeIn(step2, shift=RIGHT), run_time=0.9)
        self.play(GrowArrow(arr2), FadeIn(step3, shift=RIGHT), run_time=0.9)

        pulse = SurroundingRectangle(step3, color="#10B981", buff=0.12, stroke_width=3)
        self.play(Create(pulse), run_time=0.8)
        self.wait({hold_s:.2f})
        self.play(FadeOut(VGroup(flow_title, step1, arr1, step2, arr2, step3, pulse)), run_time=0.8)
        '''
    else:
        step_setup = f'''
        b1 = RoundedRectangle(corner_radius=0.15, height=2.5, width=4.5, color="#0284C7")
        b1.set_fill("#0C4A6E", opacity=0.9).move_to(LEFT * 3.0 + DOWN * 0.4)
        t1 = Text("{s1_t}", font_size=20, weight=BOLD, color="#7DD3FC").next_to(b1.get_top(), DOWN, buff=0.25)
        d1 = Text("{s1_d}", font_size=15, color="#F0F9FF").next_to(t1, DOWN, buff=0.2)
        step1 = VGroup(b1, t1, d1)

        b2 = RoundedRectangle(corner_radius=0.15, height=2.5, width=4.5, color="#059669")
        b2.set_fill("#064E3B", opacity=0.9).move_to(RIGHT * 3.0 + DOWN * 0.4)
        t2 = Text("{s2_t}", font_size=20, weight=BOLD, color="#6EE7B7").next_to(b2.get_top(), DOWN, buff=0.25)
        d2 = Text("{s2_d}", font_size=15, color="#ECFDF5").next_to(t2, DOWN, buff=0.2)
        step2 = VGroup(b2, t2, d2)

        arr1 = Arrow(b1.get_right(), b2.get_left(), buff=0.15, color="#94A3B8", stroke_width=4)

        self.play(FadeIn(step1, shift=RIGHT), run_time=0.9)
        self.play(GrowArrow(arr1), FadeIn(step2, shift=RIGHT), run_time=0.9)

        pulse = SurroundingRectangle(step2, color="#10B981", buff=0.12, stroke_width=3)
        self.play(Create(pulse), run_time=0.8)
        self.wait({hold_s:.2f})
        self.play(FadeOut(VGroup(flow_title, step1, arr1, step2, pulse)), run_time=0.8)
        '''

    code = f'''
        # --- Sahne: Süreç Akışı / Boru Hattı ---
        flow_title = Text("{clean_topic}: Adım Adım İşleyiş", font_size=28, weight=BOLD, color="#38BDF8")
        flow_title.to_edge(UP, buff=0.45)
        self.play(FadeIn(flow_title, shift=DOWN), run_time=0.8)
{step_setup}
'''
    return code


def build_comparison_table_code(
    topic: str,
    left_title: str,
    right_title: str,
    left_points: List[str],
    right_points: List[str],
    duration_s: float = 14.0,
    lang: str = "tr",
) -> str:
    """Generate Manim code showing a two-column comparison card with distinct badges."""
    clean_topic = safe_text(topic, 45)
    c1_name = safe_text(left_title or "Yöntem A", 25)
    c2_name = safe_text(right_title or "Yöntem B", 25)

    lp = [wrap_text_lines(p, 28, 2) for p in (left_points or ["Temel Özellik 1", "Temel Özellik 2"])[:3]]
    rp = [wrap_text_lines(p, 28, 2) for p in (right_points or ["Karşıt Özellik 1", "Karşıt Özellik 2"])[:3]]

    anim_time = 3.6
    hold_s = max(0.5, duration_s - anim_time)

    # Generate bullet items dynamically
    l_items = []
    prev_id = "div1"
    for idx, pt in enumerate(lp):
        cur_id = f"pt{idx+1}"
        l_items.append(f'{cur_id} = Text("• {pt}", font_size=15, color="#E2E8F0").next_to({prev_id}, DOWN, buff=0.2, aligned_edge=LEFT)')
        prev_id = cur_id
    l_setup = "\n        ".join(l_items)
    l_grp = ", ".join(f"pt{i+1}" for i in range(len(lp)))

    r_items = []
    prev_id = "div2"
    for idx, pt in enumerate(rp):
        cur_id = f"qt{idx+1}"
        r_items.append(f'{cur_id} = Text("✓ {pt}", font_size=15, color="#F0FDF4").next_to({prev_id}, DOWN, buff=0.2, aligned_edge=LEFT)')
        prev_id = cur_id
    r_setup = "\n        ".join(r_items)
    r_grp = ", ".join(f"qt{i+1}" for i in range(len(rp)))

    code = f'''
        # --- Sahne: Karşılaştırma Tablosu ---
        comp_title = Text("{clean_topic}: Karşılaştırma", font_size=28, weight=BOLD, color="#38BDF8")
        comp_title.to_edge(UP, buff=0.45)
        self.play(FadeIn(comp_title, shift=DOWN), run_time=0.8)

        # Left Column Card (Concept A)
        col1_bg = RoundedRectangle(corner_radius=0.18, height=3.8, width=5.2, color="#0284C7")
        col1_bg.set_fill("#082F49", opacity=0.9).move_to(LEFT * 3.0 + DOWN * 0.5)

        h1 = Text("{c1_name}", font_size=20, weight=BOLD, color="#38BDF8").next_to(col1_bg.get_top(), DOWN, buff=0.22)
        div1 = Line(col1_bg.get_left() + RIGHT * 0.3, col1_bg.get_right() + LEFT * 0.3, color="#0284C7", stroke_width=2).next_to(h1, DOWN, buff=0.12)
        {l_setup}
        col1 = VGroup(col1_bg, h1, div1, {l_grp})

        # Right Column Card (Concept B)
        col2_bg = RoundedRectangle(corner_radius=0.18, height=3.8, width=5.2, color="#059669")
        col2_bg.set_fill("#064E3B", opacity=0.9).move_to(RIGHT * 3.0 + DOWN * 0.5)

        h2 = Text("{c2_name}", font_size=20, weight=BOLD, color="#34D399").next_to(col2_bg.get_top(), DOWN, buff=0.22)
        div2 = Line(col2_bg.get_left() + RIGHT * 0.3, col2_bg.get_right() + LEFT * 0.3, color="#059669", stroke_width=2).next_to(h2, DOWN, buff=0.12)
        {r_setup}
        col2 = VGroup(col2_bg, h2, div2, {r_grp})

        # VS Badge
        vs_circ = Circle(radius=0.45, color="#F59E0B").set_fill("#78350F", opacity=0.95).move_to(DOWN * 0.5)
        vs_t = Text("VS", font_size=18, weight=BOLD, color="#FDE68A").move_to(vs_circ.get_center())
        vs_grp = VGroup(vs_circ, vs_t)

        self.play(FadeIn(col1, shift=RIGHT), FadeIn(col2, shift=LEFT), run_time=1.0)
        self.play(GrowFromCenter(vs_grp), run_time=0.6)
        self.wait({hold_s:.2f})

        self.play(FadeOut(VGroup(comp_title, col1, col2, vs_grp)), run_time=0.8)
'''
    return code


def build_equation_scene_code(
    topic: str,
    formula: str,
    variables: List[Dict[str, str]],
    duration_s: float = 14.0,
    lang: str = "tr",
) -> str:
    """Generate Manim code showing a key formula/equation with explanatory parameter cards."""
    clean_topic = safe_text(topic, 45)
    latex_form = safe_latex(formula or "CV = EV - AC")
    fallback_text_form = latex_to_unicode_math(formula or "CV = EV - AC")

    vars_items = variables if variables and len(variables) >= 1 else [
        {"symbol": "Terim 1", "meaning": "Girdi faktörü"},
        {"symbol": "Terim 2", "meaning": "Sonuç katsayısı"},
    ]

    v1_name = safe_text(vars_items[0].get("symbol", "Terim 1"), 20)
    v1_desc = wrap_text_lines(vars_items[0].get("meaning", "Tanım"), 24, 2)

    has_v2 = len(vars_items) > 1
    v2_name = safe_text(vars_items[1].get("symbol", "Terim 2"), 20) if has_v2 else ""
    v2_desc = wrap_text_lines(vars_items[1].get("meaning", "Tanım"), 24, 2) if has_v2 else ""

    anim_time = 3.6
    hold_s = max(0.5, duration_s - anim_time)

    v1_x = "-2.7" if has_v2 else "0.0"
    v2_setup = ""
    v2_play = ""
    v2_cleanup = ""
    if has_v2:
        v2_setup = f'''
        p2_box = RoundedRectangle(corner_radius=0.15, height=2.2, width=4.8, color="#F59E0B")
        p2_box.set_fill("#78350F", opacity=0.85).move_to(DOWN * 1.6 + RIGHT * 2.7)
        p2_sym = Text("{v2_name}", font_size=20, weight=BOLD, color="#FBBF24").next_to(p2_box.get_top(), DOWN, buff=0.18)
        p2_lbl = Text("{v2_desc}", font_size=14, color="#E2E8F0").next_to(p2_sym, DOWN, buff=0.12)
        p2_grp = VGroup(p2_box, p2_sym, p2_lbl)
        '''
        v2_play = ", FadeIn(p2_grp, shift=LEFT)"
        v2_cleanup = ", p2_grp"

    code = f'''
        # --- Sahne: Denklem ve Kural Analizi ---
        eq_title = Text("{clean_topic}: Matematiksel Model", font_size=28, weight=BOLD, color="#38BDF8")
        eq_title.to_edge(UP, buff=0.45)
        self.play(FadeIn(eq_title, shift=DOWN), run_time=0.8)

        # Formula Hero Box
        f_box = RoundedRectangle(corner_radius=0.2, height=1.6, width=10.0, color="#6366F1")
        f_box.set_fill("#1E1B4B", opacity=0.9).move_to(UP * 0.8)

        # Attempt MathTex with fallback to Text
        try:
            f_text = MathTex(r"{latex_form}", font_size=32, color="#F8FAFC")
        except Exception:
            f_text = Text("{fallback_text_form}", font_size=24, weight=BOLD, color="#F8FAFC")
        f_text.move_to(f_box.get_center())
        f_group = VGroup(f_box, f_text)
        self.play(FadeIn(f_group, shift=UP), run_time=0.9)

        # Parameter 1 Card
        p1_box = RoundedRectangle(corner_radius=0.15, height=2.2, width=4.8, color="#0284C7")
        p1_box.set_fill("#082F49", opacity=0.85).move_to(DOWN * 1.6 + RIGHT * {v1_x})
        p1_sym = Text("{v1_name}", font_size=20, weight=BOLD, color="#38BDF8").next_to(p1_box.get_top(), DOWN, buff=0.18)
        p1_lbl = Text("{v1_desc}", font_size=14, color="#E2E8F0").next_to(p1_sym, DOWN, buff=0.12)
        p1_grp = VGroup(p1_box, p1_sym, p1_lbl)
{v2_setup}
        self.play(FadeIn(p1_grp, shift=RIGHT){v2_play}, run_time=0.9)
        self.wait({hold_s:.2f})

        self.play(FadeOut(VGroup(eq_title, f_group, p1_grp{v2_cleanup})), run_time=0.8)
'''
    return code


def build_concept_card_code(
    topic: str,
    analogy: str,
    terms: List[Dict[str, str]],
    takeaway: str = "",
    duration_s: float = 14.0,
    lang: str = "tr",
) -> str:
    """Generate Manim code for a polished concept card with pedagogical takeaway."""
    clean_topic = safe_text(topic, 45)
    clean_analogy = wrap_text_lines(analogy or "Pedagojik benzetme ve somut örnek", 48, 3)

    t1_name = safe_text(terms[0].get("term", "Temel İlke") if terms else "Önemli Kural", 26)
    t1_desc = wrap_text_lines(terms[0].get("explanation", "B1 düzeyinde sade tanım") if terms else "Açıklama", 26, 3)

    t2_name = safe_text(terms[1].get("term", "Pratik Çıkarım") if len(terms) > 1 else "Kullanım İpucu", 26)
    t2_desc = wrap_text_lines(terms[1].get("explanation", "Sınav ve uygulama ipucu") if len(terms) > 1 else "Özet", 26, 3)

    takeaway_txt = wrap_text_lines(
        takeaway or "Temel mantığı kavra, adımları ve kuralları bağlamında uygula.",
        52,
        2
    )

    anim_time = 4.8
    hold_s = max(0.5, duration_s - anim_time)

    code = f'''
        # --- Sahne: Kavram Kartı ve Pedagojik Benzetme ---
        card_title = Text("{clean_topic}", font_size=30, weight=BOLD, color="#38BDF8")
        card_title.to_edge(UP, buff=0.45)
        underline = Line(LEFT * 4.0, RIGHT * 4.0, color="#818CF8", stroke_width=2.5)
        underline.next_to(card_title, DOWN, buff=0.15)
        self.play(Write(card_title), GrowFromCenter(underline), run_time=0.9)

        # Analogy / Hook Banner
        hook_box = RoundedRectangle(corner_radius=0.18, height=1.5, width=10.0, color="#6366F1")
        hook_box.set_fill("#1E1B4B", opacity=0.88).next_to(underline, DOWN, buff=0.25)
        hook_text = Text("{clean_analogy}", font_size=17, line_spacing=0.85, color="#F8FAFC")
        hook_text.move_to(hook_box.get_center())
        hook_grp = VGroup(hook_box, hook_text)
        self.play(FadeIn(hook_grp, shift=UP), run_time=0.8)

        # Dual Concept Cards
        cd1 = RoundedRectangle(corner_radius=0.15, height=2.3, width=4.8, color="#059669")
        cd1.set_fill("#064E3B", opacity=0.85).move_to(LEFT * 2.65 + DOWN * 1.5)
        t1_t = Text("{t1_name}", font_size=19, weight=BOLD, color="#34D399").next_to(cd1.get_top(), DOWN, buff=0.18)
        t1_d = Text("{t1_desc}", font_size=14, color="#E2E8F0").next_to(t1_t, DOWN, buff=0.12)
        cgrp1 = VGroup(cd1, t1_t, t1_d)

        cd2 = RoundedRectangle(corner_radius=0.15, height=2.3, width=4.8, color="#D97706")
        cd2.set_fill("#78350F", opacity=0.85).move_to(RIGHT * 2.65 + DOWN * 1.5)
        t2_t = Text("{t2_name}", font_size=19, weight=BOLD, color="#FBBF24").next_to(cd2.get_top(), DOWN, buff=0.18)
        t2_d = Text("{t2_desc}", font_size=14, color="#E2E8F0").next_to(t2_t, DOWN, buff=0.12)
        cgrp2 = VGroup(cd2, t2_t, t2_d)

        self.play(FadeIn(cgrp1, shift=RIGHT), FadeIn(cgrp2, shift=LEFT), run_time=0.9)
        self.wait({hold_s * 0.6:.2f})

        self.play(FadeOut(VGroup(card_title, underline, hook_grp, cgrp1, cgrp2)), run_time=0.7)

        # Takeaway / Altın Kural (Clean geometric badge without font emoji dependencies)
        tk_box = RoundedRectangle(corner_radius=0.18, height=1.7, width=10.2, color="#10B981")
        tk_box.set_fill("#064E3B", opacity=0.94).move_to(ORIGIN)

        tk_badge = RoundedRectangle(corner_radius=0.08, height=0.45, width=2.4, color="#10B981")
        tk_badge.set_fill("#047857", opacity=0.95).next_to(tk_box.get_top(), DOWN, buff=0.16)
        tk_badge_t = Text("ALTIN ÇIKARIM", font_size=14, weight=BOLD, color="#D1FAE5").move_to(tk_badge.get_center())
        tk_badge_grp = VGroup(tk_badge, tk_badge_t)

        tk_body = Text("{takeaway_txt}", font_size=16, color="#FFFFFF").next_to(tk_badge_grp, DOWN, buff=0.16)
        takeaway_grp = VGroup(tk_box, tk_badge_grp, tk_body)

        self.play(GrowFromCenter(takeaway_grp), run_time=0.8)
        self.wait({max(0.5, hold_s * 0.4):.2f})
        self.play(FadeOut(takeaway_grp), run_time=0.6)
'''
    return code


def generate_modular_manim_code(
    spec: Dict[str, Any],
    lang: str = "tr",
    target_duration_s: float = 0.0,
    scene_durations: Optional[List[float]] = None,
) -> str:
    """Master generator: inspects spec['scenes'], dispatches to scene builders, and stitches code."""
    topic = spec.get("title") or spec.get("target_concept") or "Teknik Konu"
    hook = spec.get("pedagogical_hook", "")
    terms = spec.get("terms") or []
    scenes = spec.get("scenes") or []

    # If no scenes in spec, create a baseline 2-scene structure
    if not scenes:
        scenes = [
            {"id": 1, "title": topic, "scene_type": "concept_card", "duration_s": 15.0},
            {"id": 2, "title": "Özet ve Çıkarım", "scene_type": "dag_concept", "duration_s": 15.0},
        ]

    # Calculate time budget per scene
    num_scenes = len(scenes)
    total_budget = target_duration_s if target_duration_s > 0 else sum(float(s.get("duration_s", 12.0)) for s in scenes)
    avg_per_scene = max(4.0, total_budget / max(1, num_scenes))

    scene_codes: List[str] = []

    for idx, sc in enumerate(scenes):
        dur = scene_durations[idx] if (scene_durations and idx < len(scene_durations)) else float(sc.get("duration_s") or avg_per_scene)
        stype = detect_scene_type(sc, default="concept_card")
        sc_title = sc.get("title") or f"{topic} Bölüm {idx+1}"
        sc_narration = sc.get("narration") or ""

        if stype == "dag_concept":
            nodes_data = sc.get("nodes")
            if not nodes_data:
                # Derive from terms dynamically
                if len(terms) >= 2:
                    nodes_data = [
                        {"name": terms[0].get("term", "Önkoşul"), "desc": terms[0].get("explanation", "Ön bilgi")},
                        {"name": topic, "desc": "Hedef kavram"},
                        {"name": terms[1].get("term", "Uygulama"), "desc": terms[1].get("explanation", "Pratik yansıma")},
                    ]
                else:
                    nodes_data = [
                        {"name": "Ders Zemin Bilgisi", "desc": "Önceki konular"},
                        {"name": topic, "desc": sc_title},
                        {"name": "Sınav ve Uygulama", "desc": "Çözüm aşaması"},
                    ]
            scene_codes.append(build_dag_scene_code(sc_title, nodes_data, duration_s=dur, lang=lang))

        elif stype == "process_flow":
            steps_data = sc.get("steps")
            if not steps_data:
                if len(terms) >= 3:
                    steps_data = [
                        {"title": f"1. {safe_text(terms[0].get('term', 'Girdi'), 15)}", "desc": terms[0].get("explanation", "")},
                        {"title": f"2. {safe_text(terms[1].get('term', 'İşlem'), 15)}", "desc": terms[1].get("explanation", "")},
                        {"title": f"3. {safe_text(terms[2].get('term', 'Çıktı'), 15)}", "desc": terms[2].get("explanation", "")},
                    ]
                else:
                    steps_data = [
                        {"title": f"1. {safe_text(sc_title, 14)}", "desc": safe_text(sc_narration, 35) or "Girdi ve problem"},
                        {"title": "2. Temel Mekanizma", "desc": "Kuralların işletilmesi"},
                        {"title": "3. Çözüm ve Çıktı", "desc": "Nihai pedagojik sonuç"},
                    ]
            scene_codes.append(build_process_flow_code(sc_title, steps_data, duration_s=dur, lang=lang))

        elif stype == "comparison_table":
            left_t = sc.get("left_title")
            right_t = sc.get("right_title")
            left_p = sc.get("left_points")
            right_p = sc.get("right_points")

            if not left_t or not right_t:
                if len(terms) >= 2:
                    left_t = terms[0].get("term", "Kavram A")
                    right_t = terms[1].get("term", "Kavram B")
                    left_p = left_p or [terms[0].get("explanation", "Temel özellik")]
                    right_p = right_p or [terms[1].get("explanation", "Karşıt özellik")]
                else:
                    left_t = left_t or "Klasik Yaklaşım"
                    right_t = right_t or f"{topic} Modeli"
                    left_p = left_p or ["Daha yüksek karmaşıklık", "Bağlamdan kopuk ezber"]
                    right_p = right_p or ["Doğrudan amaca yönelik", "Pedagojik netlik"]

            scene_codes.append(build_comparison_table_code(sc_title, left_t, right_t, left_p, right_p, duration_s=dur, lang=lang))

        elif stype == "equation_step":
            formula = sc.get("formula")
            vars_data = sc.get("variables")
            if not formula:
                # Check terms for formula
                formula = next((t["term"] for t in terms if "=" in t.get("term", "")), None)
            if not formula:
                # Fall back to concept_card when no genuine formula exists for the topic
                scene_takeaway = sc.get("takeaway") or sc_narration or f"{topic} için temel kural ve çıkarım."
                scene_hook = sc.get("pedagogical_hook") or f"{sc_title} analizi ve kural özeti."
                scene_terms = sc.get("terms") or terms
                scene_codes.append(build_concept_card_code(sc_title, scene_hook, scene_terms, takeaway=scene_takeaway, duration_s=dur, lang=lang))
                continue
            if not vars_data:
                vars_data = [
                    {"symbol": "EV", "meaning": "Kazanılmış Değer (Earned Value)"},
                    {"symbol": "AC", "meaning": "Gerçekleşen Maliyet (Actual Cost)"},
                ]
            scene_codes.append(build_equation_scene_code(sc_title, formula, vars_data, duration_s=dur, lang=lang))

        else:  # concept_card fallback
            # Distinct takeaway per scene so multiple concept cards do not repeat the hook
            scene_takeaway = sc.get("takeaway") or sc_narration or f"{topic} için temel çıkarım: Kuralları doğru sırayla uygulayın."
            scene_hook = sc.get("pedagogical_hook") or (hook if idx == 0 else f"{sc_title} analizi ve kilit noktaları.")
            scene_terms = sc.get("terms") or (terms if idx == 0 else (terms[1:] if len(terms) > 1 else terms))
            scene_codes.append(build_concept_card_code(sc_title, scene_hook, scene_terms, takeaway=scene_takeaway, duration_s=dur, lang=lang))

    body = "\n".join(scene_codes)

    full_code = f'''from manim import *

class ExplainerScene(Scene):
    def construct(self):
        # Professional midnight obsidian background
        self.camera.background_color = "#0B0F19"
{body}
'''
    return full_code
