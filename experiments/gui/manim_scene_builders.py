"""Modular Parametric Scene Builders for Manimator AI Video Generation.

This module provides deterministic, error-free Manim Community v0.21+ scene builders
for educational concept scaffolding:
  1. DAG / Concept Graph (`dag_concept`)
  2. Process Flow / Pipeline (`process_flow`)
  3. Comparison Table (`comparison_table`)
  4. Equation & Rule Analysis (`equation_step`)
  5. Concept Card / Pedagogical Takeaway (`concept_card`)

Each builder receives structured parameters and timing budgets, producing safe,
reproducible Manim Python code with automatic text wrapping and dark-mode aesthetics.
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
        return explicit

    title = str(scene_data.get("title") or "").lower()
    narration = str(scene_data.get("narration") or "").lower()
    combined = f"{title} {narration}"

    if any(k in combined for k in ["formül", "equation", "denklem", "hesap", "matematik", "kural"]):
        return "equation_step"
    if any(k in combined for k in ["karşılaştır", "fark", "vs", "versus", "kıyas", "tablo", "avantaj"]):
        return "comparison_table"
    if any(k in combined for k in ["adım", "aşama", "süreç", "akış", "pipeline", "sıra", "algoritma"]):
        return "process_flow"
    if any(k in combined for k in ["ilişki", "önkoşul", "bağlantı", "hiyerarşi", "dag", "ağaç", "graf"]):
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
    scale = max(0.5, min(4.0, duration_s / 10.0))
    t_write = f"{0.9 * scale:.2f}"
    t_grow = f"{0.7 * scale:.2f}"
    t_hold = f"{1.5 * scale:.2f}"

    clean_topic = safe_text(topic, 45)
    root_title = safe_text(nodes[0].get("name", "Önkoşul Kavram") if nodes else "Temel Kavram", 28)
    root_desc = wrap_text_lines(nodes[0].get("desc", "Gerekli temel bilgi") if nodes else "Önkoşul tanımı", 22, 2)

    child1_title = safe_text(nodes[1].get("name", "Hedef Kavram") if len(nodes) > 1 else clean_topic, 28)
    child1_desc = wrap_text_lines(nodes[1].get("desc", "Yeni öğrenilen konu") if len(nodes) > 1 else "Hedef tanım", 22, 2)

    child2_title = safe_text(nodes[2].get("name", "Uygulama Alanı") if len(nodes) > 2 else "Pratik Kullanım", 28)
    child2_desc = wrap_text_lines(nodes[2].get("desc", "Pratik yansıma") if len(nodes) > 2 else "Çıkarım", 22, 2)

    code = f'''
        # --- Sahne: DAG / Kavram Hiyerarşisi ---
        dag_title = Text("{clean_topic}: Kavram Bağıntıları", font_size=28, weight=BOLD, color="#38BDF8")
        dag_title.to_edge(UP, buff=0.45)
        self.play(FadeIn(dag_title, shift=DOWN), run_time={t_write})

        # Root Node (Prerequisite)
        root_box = RoundedRectangle(corner_radius=0.18, height=1.6, width=3.8, color="#818CF8")
        root_box.set_fill("#1E1B4B", opacity=0.9)
        root_box.move_to(UP * 1.0)
        root_t = Text("{root_title}", font_size=20, weight=BOLD, color="#C7D2FE")
        root_t.next_to(root_box.get_top(), DOWN, buff=0.2)
        root_d = Text("{root_desc}", font_size=15, color="#E2E8F0")
        root_d.next_to(root_t, DOWN, buff=0.15)
        root_grp = VGroup(root_box, root_t, root_d)
        self.play(FadeIn(root_grp, shift=UP), run_time={t_write})

        # Child Node 1
        c1_box = RoundedRectangle(corner_radius=0.18, height=1.6, width=3.8, color="#34D399")
        c1_box.set_fill("#064E3B", opacity=0.9)
        c1_box.move_to(DOWN * 1.5 + LEFT * 2.6)
        c1_t = Text("{child1_title}", font_size=20, weight=BOLD, color="#A7F3D0")
        c1_t.next_to(c1_box.get_top(), DOWN, buff=0.2)
        c1_d = Text("{child1_desc}", font_size=15, color="#E2E8F0")
        c1_d.next_to(c1_t, DOWN, buff=0.15)
        c1_grp = VGroup(c1_box, c1_t, c1_d)

        # Child Node 2
        c2_box = RoundedRectangle(corner_radius=0.18, height=1.6, width=3.8, color="#F59E0B")
        c2_box.set_fill("#78350F", opacity=0.9)
        c2_box.move_to(DOWN * 1.5 + RIGHT * 2.6)
        c2_t = Text("{child2_title}", font_size=20, weight=BOLD, color="#FDE68A")
        c2_t.next_to(c2_box.get_top(), DOWN, buff=0.2)
        c2_d = Text("{child2_desc}", font_size=15, color="#E2E8F0")
        c2_d.next_to(c2_t, DOWN, buff=0.15)
        c2_grp = VGroup(c2_box, c2_t, c2_d)

        # Directed Edges
        arrow1 = Arrow(root_box.get_bottom(), c1_box.get_top(), buff=0.15, color="#A5B4FC", stroke_width=3.5)
        arrow2 = Arrow(root_box.get_bottom(), c2_box.get_top(), buff=0.15, color="#A5B4FC", stroke_width=3.5)

        self.play(GrowArrow(arrow1), GrowArrow(arrow2), run_time={t_grow})
        self.play(FadeIn(c1_grp, shift=RIGHT), FadeIn(c2_grp, shift=LEFT), run_time={t_write})

        # Highlight target concept
        focus_box = SurroundingRectangle(c1_grp, color="#38BDF8", buff=0.1, stroke_width=2.5)
        self.play(Create(focus_box), run_time={t_grow})
        self.wait({t_hold})

        self.play(FadeOut(VGroup(dag_title, root_grp, c1_grp, c2_grp, arrow1, arrow2, focus_box)), run_time=0.8)
'''
    return code


def build_process_flow_code(
    topic: str,
    steps: List[Dict[str, str]],
    duration_s: float = 14.0,
    lang: str = "tr",
) -> str:
    """Generate Manim code showing a multi-step sequential process or algorithm pipeline."""
    scale = max(0.5, min(4.0, duration_s / 12.0))
    t_step = f"{0.8 * scale:.2f}"
    t_arrow = f"{0.5 * scale:.2f}"
    t_hold = f"{1.4 * scale:.2f}"

    clean_topic = safe_text(topic, 45)
    default_steps = [
        {"title": "1. Girdi / Tespit", "desc": "Problemin saptanması"},
        {"title": "2. İşlem / Dönüşüm", "desc": "Algoritmanın işletilmesi"},
        {"title": "3. Çıktı / Sonuç", "desc": "Nihai doğrulanmış çözüm"},
    ]
    raw_steps = steps if len(steps) >= 2 else default_steps
    step_items = raw_steps[:3]

    s1_t = safe_text(step_items[0].get("title", "Adım 1"), 20)
    s1_d = wrap_text_lines(step_items[0].get("desc", "Aşama detayı"), 18, 2)

    s2_t = safe_text(step_items[1].get("title", "Adım 2"), 20)
    s2_d = wrap_text_lines(step_items[1].get("desc", "Aşama detayı"), 18, 2)

    s3_t = safe_text(step_items[2].get("title", "Adım 3") if len(step_items) > 2 else "Tamamlandı", 20)
    s3_d = wrap_text_lines(step_items[2].get("desc", "Aşama detayı") if len(step_items) > 2 else "Çıktı", 18, 2)

    code = f'''
        # --- Sahne: Süreç Akışı / Boru Hattı ---
        flow_title = Text("{clean_topic}: Adım Adım İşleyiş", font_size=28, weight=BOLD, color="#38BDF8")
        flow_title.to_edge(UP, buff=0.45)
        self.play(FadeIn(flow_title, shift=DOWN), run_time={t_step})

        # Step 1 Card
        b1 = RoundedRectangle(corner_radius=0.15, height=2.4, width=3.3, color="#0284C7")
        b1.set_fill("#0C4A6E", opacity=0.9)
        b1.move_to(LEFT * 4.0 + DOWN * 0.4)
        t1 = Text("{s1_t}", font_size=19, weight=BOLD, color="#7DD3FC")
        t1.next_to(b1.get_top(), DOWN, buff=0.25)
        d1 = Text("{s1_d}", font_size=15, color="#F0F9FF")
        d1.next_to(t1, DOWN, buff=0.2)
        step1 = VGroup(b1, t1, d1)

        # Step 2 Card
        b2 = RoundedRectangle(corner_radius=0.15, height=2.4, width=3.3, color="#6366F1")
        b2.set_fill("#312E81", opacity=0.9)
        b2.move_to(DOWN * 0.4)
        t2 = Text("{s2_t}", font_size=19, weight=BOLD, color="#A5B4FC")
        t2.next_to(b2.get_top(), DOWN, buff=0.25)
        d2 = Text("{s2_d}", font_size=15, color="#EEF2FF")
        d2.next_to(t2, DOWN, buff=0.2)
        step2 = VGroup(b2, t2, d2)

        # Step 3 Card
        b3 = RoundedRectangle(corner_radius=0.15, height=2.4, width=3.3, color="#059669")
        b3.set_fill("#064E3B", opacity=0.9)
        b3.move_to(RIGHT * 4.0 + DOWN * 0.4)
        t3 = Text("{s3_t}", font_size=19, weight=BOLD, color="#6EE7B7")
        t3.next_to(b3.get_top(), DOWN, buff=0.25)
        d3 = Text("{s3_d}", font_size=15, color="#ECFDF5")
        d3.next_to(t3, DOWN, buff=0.2)
        step3 = VGroup(b3, t3, d3)

        # Sequence arrows
        arr1 = Arrow(b1.get_right(), b2.get_left(), buff=0.12, color="#94A3B8", stroke_width=4)
        arr2 = Arrow(b2.get_right(), b3.get_left(), buff=0.12, color="#94A3B8", stroke_width=4)

        self.play(FadeIn(step1, shift=RIGHT), run_time={t_step})
        self.play(GrowArrow(arr1), FadeIn(step2, shift=RIGHT), run_time={t_step})
        self.play(GrowArrow(arr2), FadeIn(step3, shift=RIGHT), run_time={t_step})

        # Pipeline progress pulse
        pulse = SurroundingRectangle(step3, color="#10B981", buff=0.12, stroke_width=3)
        self.play(Create(pulse), run_time={t_arrow})
        self.wait({t_hold})

        self.play(FadeOut(VGroup(flow_title, step1, arr1, step2, arr2, step3, pulse)), run_time=0.8)
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
    scale = max(0.5, min(4.0, duration_s / 12.0))
    t_in = f"{1.0 * scale:.2f}"
    t_hold = f"{1.8 * scale:.2f}"

    clean_topic = safe_text(topic, 45)
    c1_name = safe_text(left_title or "Yöntem A", 25)
    c2_name = safe_text(right_title or "Yöntem B", 25)

    p1 = wrap_text_lines(left_points[0] if left_points else "Klasik yaklaşım", 28, 2)
    p2 = wrap_text_lines(left_points[1] if len(left_points) > 1 else "Daha yüksek karmaşıklık", 28, 2)

    q1 = wrap_text_lines(right_points[0] if right_points else "Önerilen çözüm", 28, 2)
    q2 = wrap_text_lines(right_points[1] if len(right_points) > 1 else "Düşük kaynak tüketimi", 28, 2)

    code = f'''
        # --- Sahne: Karşılaştırma Tablosu ---
        comp_title = Text("{clean_topic}: Karşılaştırma", font_size=28, weight=BOLD, color="#38BDF8")
        comp_title.to_edge(UP, buff=0.45)
        self.play(FadeIn(comp_title, shift=DOWN), run_time=0.8)

        # Left Column Card (Concept A)
        col1_bg = RoundedRectangle(corner_radius=0.18, height=3.8, width=5.2, color="#0284C7")
        col1_bg.set_fill("#082F49", opacity=0.9)
        col1_bg.move_to(LEFT * 3.0 + DOWN * 0.5)

        h1 = Text("{c1_name}", font_size=22, weight=BOLD, color="#38BDF8")
        h1.next_to(col1_bg.get_top(), DOWN, buff=0.25)
        div1 = Line(col1_bg.get_left() + RIGHT * 0.4, col1_bg.get_right() + LEFT * 0.4, color="#0284C7", stroke_width=2)
        div1.next_to(h1, DOWN, buff=0.15)

        pt1 = Text("• {p1}", font_size=16, color="#E2E8F0")
        pt1.next_to(div1, DOWN, buff=0.25, aligned_edge=LEFT)
        pt2 = Text("• {p2}", font_size=16, color="#E2E8F0")
        pt2.next_to(pt1, DOWN, buff=0.25, aligned_edge=LEFT)
        col1 = VGroup(col1_bg, h1, div1, pt1, pt2)

        # Right Column Card (Concept B)
        col2_bg = RoundedRectangle(corner_radius=0.18, height=3.8, width=5.2, color="#059669")
        col2_bg.set_fill("#064E3B", opacity=0.9)
        col2_bg.move_to(RIGHT * 3.0 + DOWN * 0.5)

        h2 = Text("{c2_name}", font_size=22, weight=BOLD, color="#34D399")
        h2.next_to(col2_bg.get_top(), DOWN, buff=0.25)
        div2 = Line(col2_bg.get_left() + RIGHT * 0.4, col2_bg.get_right() + LEFT * 0.4, color="#059669", stroke_width=2)
        div2.next_to(h2, DOWN, buff=0.15)

        qt1 = Text("✓ {q1}", font_size=16, color="#F0FDF4")
        qt1.next_to(div2, DOWN, buff=0.25, aligned_edge=LEFT)
        qt2 = Text("✓ {q2}", font_size=16, color="#F0FDF4")
        qt2.next_to(qt1, DOWN, buff=0.25, aligned_edge=LEFT)
        col2 = VGroup(col2_bg, h2, div2, qt1, qt2)

        # VS Badge
        vs_circ = Circle(radius=0.45, color="#F59E0B").set_fill("#78350F", opacity=0.95)
        vs_circ.move_to(DOWN * 0.5)
        vs_t = Text("VS", font_size=18, weight=BOLD, color="#FDE68A").move_to(vs_circ.get_center())
        vs_grp = VGroup(vs_circ, vs_t)

        self.play(FadeIn(col1, shift=RIGHT), FadeIn(col2, shift=LEFT), run_time={t_in})
        self.play(GrowFromCenter(vs_grp), run_time=0.6)
        self.wait({t_hold})

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
    scale = max(0.5, min(4.0, duration_s / 12.0))
    t_write = f"{1.0 * scale:.2f}"
    t_hold = f"{1.8 * scale:.2f}"

    clean_topic = safe_text(topic, 45)
    clean_formula = safe_text(formula or "w_{t+1} = w_t - \\eta \\cdot \\nabla L", 50)

    v1_name = safe_text(variables[0].get("symbol", "Parametre 1") if variables else "w", 20)
    v1_desc = wrap_text_lines(variables[0].get("meaning", "Güncellenen ağırlık vektörü") if variables else "Ağırlık", 26, 2)

    v2_name = safe_text(variables[1].get("symbol", "Parametre 2") if len(variables) > 1 else "\\eta", 20)
    v2_desc = wrap_text_lines(variables[1].get("meaning", "Öğrenme oranı (adım büyüklüğü)") if len(variables) > 1 else "Öğrenme oranı", 26, 2)

    code = f'''
        # --- Sahne: Denklem ve Kural Analizi ---
        eq_title = Text("{clean_topic}: Matematiksel Model", font_size=28, weight=BOLD, color="#38BDF8")
        eq_title.to_edge(UP, buff=0.45)
        self.play(FadeIn(eq_title, shift=DOWN), run_time=0.8)

        # Formula Hero Box
        f_box = RoundedRectangle(corner_radius=0.2, height=1.6, width=10.0, color="#6366F1")
        f_box.set_fill("#1E1B4B", opacity=0.9)
        f_box.move_to(UP * 0.8)

        # Safe formula display with Text
        f_text = Text("{clean_formula}", font_size=28, weight=BOLD, color="#F8FAFC")
        f_text.move_to(f_box.get_center())
        f_group = VGroup(f_box, f_text)
        self.play(FadeIn(f_group, shift=UP), run_time={t_write})

        # Parameter 1 Card
        p1_box = RoundedRectangle(corner_radius=0.15, height=2.2, width=4.8, color="#0284C7")
        p1_box.set_fill("#082F49", opacity=0.85)
        p1_box.move_to(DOWN * 1.6 + LEFT * 2.7)
        p1_sym = Text("{v1_name}", font_size=22, weight=BOLD, color="#38BDF8")
        p1_sym.next_to(p1_box.get_top(), DOWN, buff=0.2)
        p1_lbl = Text("{v1_desc}", font_size=15, color="#E2E8F0")
        p1_lbl.next_to(p1_sym, DOWN, buff=0.15)
        p1_grp = VGroup(p1_box, p1_sym, p1_lbl)

        # Parameter 2 Card
        p2_box = RoundedRectangle(corner_radius=0.15, height=2.2, width=4.8, color="#F59E0B")
        p2_box.set_fill("#78350F", opacity=0.85)
        p2_box.move_to(DOWN * 1.6 + RIGHT * 2.7)
        p2_sym = Text("{v2_name}", font_size=22, weight=BOLD, color="#FBBF24")
        p2_sym.next_to(p2_box.get_top(), DOWN, buff=0.2)
        p2_lbl = Text("{v2_desc}", font_size=15, color="#E2E8F0")
        p2_lbl.next_to(p2_sym, DOWN, buff=0.15)
        p2_grp = VGroup(p2_box, p2_sym, p2_lbl)

        self.play(FadeIn(p1_grp, shift=RIGHT), FadeIn(p2_grp, shift=LEFT), run_time={t_write})
        self.wait({t_hold})

        self.play(FadeOut(VGroup(eq_title, f_group, p1_grp, p2_grp)), run_time=0.8)
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
    scale = max(0.5, min(4.0, duration_s / 12.0))
    t_write = f"{1.0 * scale:.2f}"
    t_hold = f"{1.6 * scale:.2f}"

    clean_topic = safe_text(topic, 45)
    clean_analogy = wrap_text_lines(analogy or "Pedagojik benzetme ve somut örnek", 48, 3)

    t1_name = safe_text(terms[0].get("term", "Temel Kural") if terms else "Önemli Nokta", 26)
    t1_desc = wrap_text_lines(terms[0].get("explanation", "B1 düzeyinde sade tanım") if terms else "Açıklama", 26, 3)

    t2_name = safe_text(terms[1].get("term", "İpucu") if len(terms) > 1 else "Pratik Çıkarım", 26)
    t2_desc = wrap_text_lines(terms[1].get("explanation", "Sınav ve uygulama ipucu") if len(terms) > 1 else "Özet", 26, 3)

    takeaway_txt = wrap_text_lines(
        takeaway or "Altın Kural: Ezberleme, mantığı kavra ve bağlamı izle.",
        52,
        2
    )

    code = f'''
        # --- Sahne: Kavram Kartı ve Pedagojik Benzetme ---
        card_title = Text("{clean_topic}", font_size=32, weight=BOLD, color="#38BDF8")
        card_title.to_edge(UP, buff=0.45)
        underline = Line(LEFT * 4.0, RIGHT * 4.0, color="#818CF8", stroke_width=2.5)
        underline.next_to(card_title, DOWN, buff=0.15)
        self.play(Write(card_title), GrowFromCenter(underline), run_time={t_write})

        # Analogy / Hook Banner
        hook_box = RoundedRectangle(corner_radius=0.18, height=1.5, width=10.0, color="#6366F1")
        hook_box.set_fill("#1E1B4B", opacity=0.88)
        hook_box.next_to(underline, DOWN, buff=0.25)
        hook_text = Text("{clean_analogy}", font_size=18, line_spacing=0.85, color="#F8FAFC")
        hook_text.move_to(hook_box.get_center())
        hook_grp = VGroup(hook_box, hook_text)
        self.play(FadeIn(hook_grp, shift=UP), run_time=0.8)

        # Dual Concept Cards
        cd1 = RoundedRectangle(corner_radius=0.15, height=2.3, width=4.8, color="#059669")
        cd1.set_fill("#064E3B", opacity=0.85)
        cd1.move_to(LEFT * 2.65 + DOWN * 1.5)
        t1_t = Text("{t1_name}", font_size=20, weight=BOLD, color="#34D399")
        t1_t.next_to(cd1.get_top(), DOWN, buff=0.2)
        t1_d = Text("{t1_desc}", font_size=15, color="#E2E8F0")
        t1_d.next_to(t1_t, DOWN, buff=0.15)
        cgrp1 = VGroup(cd1, t1_t, t1_d)

        cd2 = RoundedRectangle(corner_radius=0.15, height=2.3, width=4.8, color="#D97706")
        cd2.set_fill("#78350F", opacity=0.85)
        cd2.move_to(RIGHT * 2.65 + DOWN * 1.5)
        t2_t = Text("{t2_name}", font_size=20, weight=BOLD, color="#FBBF24")
        t2_t.next_to(cd2.get_top(), DOWN, buff=0.2)
        t2_d = Text("{t2_desc}", font_size=15, color="#E2E8F0")
        t2_d.next_to(t2_t, DOWN, buff=0.15)
        cgrp2 = VGroup(cd2, t2_t, t2_d)

        self.play(FadeIn(cgrp1, shift=RIGHT), FadeIn(cgrp2, shift=LEFT), run_time={t_write})
        self.wait({t_hold})

        self.play(FadeOut(VGroup(card_title, underline, hook_grp, cgrp1, cgrp2)), run_time=0.8)

        # Takeaway / Altın Kural (Replaces static 'Tebrikler' screen)
        tk_box = RoundedRectangle(corner_radius=0.18, height=1.8, width=10.2, color="#10B981")
        tk_box.set_fill("#064E3B", opacity=0.92)
        tk_box.move_to(ORIGIN)
        tk_title = Text("💡 Altın Çıkarım", font_size=22, weight=BOLD, color="#6EE7B7")
        tk_title.next_to(tk_box.get_top(), DOWN, buff=0.2)
        tk_body = Text("{takeaway_txt}", font_size=17, color="#FFFFFF")
        tk_body.next_to(tk_title, DOWN, buff=0.15)
        takeaway_grp = VGroup(tk_box, tk_title, tk_body)

        self.play(GrowFromCenter(takeaway_grp), run_time=0.9)
        self.wait(1.5)
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
    avg_per_scene = max(6.0, total_budget / max(1, num_scenes))

    scene_codes: List[str] = []

    for idx, sc in enumerate(scenes):
        dur = scene_durations[idx] if (scene_durations and idx < len(scene_durations)) else sc.get("duration_s", avg_per_scene)
        stype = detect_scene_type(sc, default="concept_card")

        if stype == "dag_concept":
            nodes_data = sc.get("nodes") or [
                {"name": terms[0].get("term", "Önkoşul") if terms else "Temel", "desc": "Ön Bilgi"},
                {"name": topic, "desc": "Hedef Kavram"},
                {"name": terms[1].get("term", "Uygulama") if len(terms) > 1 else "Kullanım", "desc": "Pratik"},
            ]
            scene_codes.append(build_dag_scene_code(topic, nodes_data, duration_s=dur, lang=lang))

        elif stype == "process_flow":
            steps_data = sc.get("steps") or [
                {"title": "1. " + safe_text(sc.get("title", "Aşama 1"), 15), "desc": safe_text(sc.get("narration", "Giriş"), 35)},
                {"title": "2. İşlem", "desc": "Çözümleme ve hesap"},
                {"title": "3. Çıktı", "desc": "Nihai pedagojik sonuç"},
            ]
            scene_codes.append(build_process_flow_code(topic, steps_data, duration_s=dur, lang=lang))

        elif stype == "comparison_table":
            left_t = sc.get("left_title") or (terms[0].get("term") if terms else "Geleneksel")
            right_t = sc.get("right_title") or (terms[1].get("term") if len(terms) > 1 else topic)
            left_p = sc.get("left_points") or ["Statik veya kural tabanlı", "Yüksek bellek ihtiyacı"]
            right_p = sc.get("right_points") or ["Dinamik ve pedagojik", "Hafif ve optimize"]
            scene_codes.append(build_comparison_table_code(topic, left_t, right_t, left_p, right_p, duration_s=dur, lang=lang))

        elif stype == "equation_step":
            formula = sc.get("formula") or "E = mc^2"
            vars_data = sc.get("variables") or [
                {"symbol": "Terim 1", "meaning": "Girdi faktörü"},
                {"symbol": "Terim 2", "meaning": "Sonuç katsayısı"},
            ]
            scene_codes.append(build_equation_scene_code(topic, formula, vars_data, duration_s=dur, lang=lang))

        else:  # concept_card fallback
            takeaway = sc.get("takeaway") or spec.get("pedagogical_hook") or "Konunun özü: Temel mantığı doğru bağlamda uygulamak."
            scene_codes.append(build_concept_card_code(topic, hook, terms, takeaway=takeaway, duration_s=dur, lang=lang))

    body = "\n".join(scene_codes)

    full_code = f'''from manim import *

class ExplainerScene(Scene):
    def construct(self):
        # Professional midnight obsidian background
        self.camera.background_color = "#0B0F19"
{body}
'''
    return full_code
