"""Socratic AI Tutor / Chat Module (Vanzo et al., ACL 2025).

Pedagogical Basis:
Research by Vanzo et al. (ACL 2025) demonstrates that providing guiding, Socratic
questions rather than spoon-feeding direct answers triggers deeper conceptual
understanding and active recall.

Key Pedagogical Behaviors:
1. Never give the direct numerical answer or formula derivation away on the first turn.
2. Formulate 1-2 guiding / scaffolding questions (yönlendirici sorular).
3. Connect the concept to intuitive real-world analogies (CEFR B1 friendly).
4. Provide suggested response pills for active student participation.
5. Synchronize with the video lecture timestamp for context-aware tutoring.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("EduTerm.Socratic")

SOCRATIC_SYSTEM_PROMPT = (
    "Sen üniversite düzeyinde ders dinleyen bir öğrenciye yardımcı olan 'Sokratik Yapay Zekâ Çalışma Arkadaşı'sın. "
    "Pedagojik İlke (Vanzo et al., ACL 2025): "
    "Öğrenciye ASLA doğrudan hazır cevabı verme! Bunun yerine öğrencinin cevabı kendi zihninde keşfetmesini sağlayacak "
    "1-2 yönlendirici, düşündürücü soru sor (Sokratik Yöntem). "
    "Dili CEFR B1 başlangıç seviyesinde, samimi ve teşvik edici tut. "
    "Ders bağlamı ve terimleri aşağıdadır:\n"
    "{CONTEXT}\n\n"
    "Öğrencinin sorusu: {QUESTION}\n"
    "Cevap kuralları:\n"
    "1. Öğrencinin sorusunu empatiyle karşıla.\n"
    "2. Günlük hayattan basit bir benzetme (analogi) kur.\n"
    "3. Öğrenciye cevabı bulduracak 1 veya en fazla 2 net yönlendirici soru sor.\n"
    "4. Cevabın sonunda [SEÇENEKLER] etiketi altında öğrencinin seçebileceği 2-3 kısa yanıt önerisi yaz (virgülle ayırarak)."
)


def _find_active_concept(user_msg: str, terms: list[dict], current_time_s: Optional[float] = None) -> Optional[dict]:
    """Identify which concept or formula the student is asking about."""
    msg_low = (user_msg or "").lower()
    
    # 1. Direct term match
    for t in (terms or []):
        t_name = (t.get("term") or "").lower()
        if t_name and (t_name in msg_low or any(w in msg_low for w in t_name.split() if len(w) >= 4)):
            return t

    # 2. Key phrases
    if any(k in msg_low for k in ["formül", "formula", "hesap", "cv", "sv", "cpi", "spi", "sapma"]):
        for t in (terms or []):
            t_low = (t.get("term") or "").lower()
            if any(k in t_low for k in ["maliyet sapması", "çizelge sapması", "performans", "kazanılmış değer"]):
                return t

    # 3. Match closest to playback time if available
    if current_time_s is not None and terms:
        closest = None
        min_diff = float("inf")
        for t in terms:
            if t.get("start_s") is not None:
                diff = abs(float(t["start_s"]) - current_time_s)
                if diff < min_diff:
                    min_diff = diff
                    closest = t
        if closest and min_diff < 120:  # within 2 minutes of playback
            return closest

    return terms[0] if terms else None


def _build_llm_context(
    active_term: Optional[dict],
    terms: list[dict],
    segments: list[dict],
    history: list[dict],
    current_time_s: Optional[float],
) -> str:
    """Assemble lecture context (active term, related terms, nearby transcript) for the LLM prompt."""
    parts = []
    if active_term:
        parts.append(f"Aktif kavram: {active_term.get('term', '')} — {active_term.get('explanation', '')}")
    related = [t for t in terms if t is not active_term][:4]
    if related:
        parts.append("Dersteki diğer terimler: " + "; ".join(
            f"{t.get('term', '')}: {t.get('explanation', '')}" for t in related if t.get("term")
        ))
    if segments:
        if current_time_s is not None:
            nearby = [
                s for s in segments
                if s.get("start") is not None and s.get("end") is not None
                and abs((float(s["start"]) + float(s["end"])) / 2 - float(current_time_s)) <= 90
            ]
        else:
            nearby = []
        nearby = nearby or segments[:3]
        transcript = " ".join((s.get("text") or "").strip() for s in nearby)[:600]
        if transcript:
            parts.append(f"Videonun ilgili bölümünden transkript: {transcript}")
    if history:
        recent = history[-4:]
        parts.append("Önceki sohbet:\n" + "\n".join(
            f"{'Öğrenci' if h.get('role') == 'user' else 'Tutor'}: {str(h.get('content', ''))[:200]}"
            for h in recent
        ))
    return "\n".join(parts) or "Bağlam bulunamadı; genel çalışma desteği ver."


def _parse_llm_reply(raw: str) -> dict:
    """Split model output into reply text and [SEÇENEKLER] suggestions."""
    text = raw.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    suggested = []
    if "[SEÇENEKLER]" in text:
        text, options_part = text.split("[SEÇENEKLER]", 1)
        suggested = [o.strip() for o in options_part.replace("\n", ",").split(",") if o.strip()][:3]
    text = text.strip()
    questions = [s.strip() + "?" for s in text.split("?") if s.strip()][-2:]
    if not suggested:
        suggested = [
            "Günlük hayattan bir örnek verir misin?",
            "Bu konuyu formülle açıklar mısın?",
            "Videoda bu kavram nerede anlatılıyor?",
        ]
    return {
        "status": "ok",
        "reply": text,
        "guiding_questions": questions,
        "suggested_replies": suggested,
        "related_term": "",
        "time_badge": None,
        "pedagogical_basis": "Vanzo et al. (ACL 2025) Socratic Scaffolding",
        "phase": "llm_socratic",
    }


def _try_llm_socratic_response(
    user_message: str,
    active_term: Optional[dict],
    terms: list[dict],
    segments: list[dict],
    history: list[dict],
    current_time_s: Optional[float],
    lang: str,
) -> Optional[dict]:
    """Attempt Socratic reply via local Qwen model; return None to trigger heuristic fallback."""
    try:
        from run_llm import generate_hf

        prompt = SOCRATIC_SYSTEM_PROMPT.format(
            CONTEXT=_build_llm_context(active_term, terms, segments, history, current_time_s),
            QUESTION=user_message,
        )
        if not str(lang).lower().startswith("tr"):
            prompt += "\n\n(Answer in English at CEFR B1 level.)"
        raw = generate_hf("Qwen/Qwen3.5-4B", prompt, max_new_tokens=350).strip()
        if not raw:
            return None
        parsed = _parse_llm_reply(raw)
        if active_term:
            parsed["related_term"] = active_term.get("term", "")
            term_start = active_term.get("start_s")
            term_time = active_term.get("timestamp", "--:--")
            if term_start is not None and term_time != "--:--":
                parsed["time_badge"] = {"start_s": term_start, "timestamp": term_time, "term": parsed["related_term"]}
        return parsed
    except Exception as exc:
        logger.warning("Socratic LLM inference fallback to heuristic: %s", exc)
        return None


def generate_socratic_response(
    user_message: str,
    history: Optional[list[dict]] = None,
    current_time_s: Optional[float] = None,
    terms: Optional[list[dict]] = None,
    segments: Optional[list[dict]] = None,
    lang: str = "tr",
    use_llm: bool = False,
) -> dict:
    """Generate a pedagogically scaffolded Socratic response.
    
    If use_llm is True and GPU/HF model is loaded, it attempts LLM inference,
    otherwise falls back to high-fidelity contextual Socratic reasoning.
    """
    msg_clean = (user_message or "").strip()
    msg_low = msg_clean.lower()
    active_term = _find_active_concept(msg_clean, terms or [], current_time_s)

    if use_llm:
        llm_result = _try_llm_socratic_response(
            msg_clean,
            active_term,
            terms or [],
            segments or [],
            history or [],
            current_time_s,
            lang,
        )
        if llm_result:
            return llm_result

    term_name = active_term.get("term", "Kazanılmış Değer") if active_term else "Kazanılmış Değer"
    term_expl = active_term.get("explanation", "") if active_term else ""
    term_time = active_term.get("timestamp", "--:--") if active_term else "--:--"
    term_start = active_term.get("start_s") if active_term else None

    # Check for student's direct answers to prior Socratic questions
    is_answering_negative = any(w in msg_low for w in ["negatif", "eksi", "< 0", "küçük", "zarar", "aşım", "aştı", "kötü", "bütçe aşımı"])
    is_answering_positive = any(w in msg_low for w in ["pozitif", "artı", "> 0", "büyük", "kâr", "tasarruf", "iyi"])
    is_asking_for_hint = any(w in msg_low for w in ["ipucu", "anlamadım", "yardım", "örnek", "nasıl", "neden"])
    is_asking_formula = any(w in msg_low for w in ["formül", "formula", "hesap", "cv", "sv", "cpi", "spi"])

    reply_text = ""
    guiding_questions = []
    suggested_replies = []
    pedagogical_phase = "scaffold"

    # Evaluation branch: Student answers "Negatif"
    if is_answering_negative and ("maliyet" in msg_low or "cv" in msg_low or "sapma" in msg_low or not is_asking_for_hint):
        reply_text = (
            f"🎯 **Tebrikler, harika bir akıl yürütme! Tam üzerine bastın.**\n\n"
            f"Eğer cebinden çıkan para (Gerçekleşen Maliyet / AC), tamamlanan işin değerinden (Kazanılmış Değer / EV) "
            f"büyükse, formüldeki çıkarma işlemi **$CV = EV - AC$ negatif çıkar**.\n\n"
            f"Bu da bütçenin aşıldığını ve kasadan planlanandan fazla para çıktığını gösterir."
        )
        guiding_questions = [
            "Peki aynı mantıkla düşünürsek, zaman çizelgesi sapmasında ($SV = EV - PV$) sonucun negatif çıkması sence ne anlama gelir?",
            "Proje takvimin ilerisinde midir, yoksa gerisinde mi kalmıştır?"
        ]
        suggested_replies = [
            "Takvimin gerisindedir (Gecikme)",
            "Takvimin ilerisindedir (Hızlı)",
            "Bu durumu düzeltmek için ne yapılır?"
        ]
        pedagogical_phase = "praise_and_deepen"

    # Evaluation branch: Student answers "Pozitif"
    elif is_answering_positive and not is_asking_for_hint:
        reply_text = (
            f"Birlikte tekrar test edelim: Diyelim ki elindeki tamamlanan işin değeri 100 TL ($EV = 100$), "
            f"fakat bu işi yaparken ustaya 130 TL ödedin ($AC = 130$).\n\n"
            f"Küçük sayıdan büyük sayıyı çıkardığımızda ($100 - 130$), sence sonuç pozitif mi olur yoksa negatif mi?"
        )
        guiding_questions = [
            "100'den 130'u çıkardığında cebindeki para azalmış mıdır, artmış mıdır?",
            "O halde maliyet sapması (CV) sence eksi (-) mi olmalı, artı (+) mı?"
        ]
        suggested_replies = [
            "Negatif olur (-30 TL), bütçe aşımı!",
            "Şimdi anladım, teşekkürler",
            "Bir ipucu daha verebilir misin?"
        ]
        pedagogical_phase = "gentle_reorientation"

    # Inquiring branch: Formula / Concept confusion
    elif is_asking_formula or "formül" in msg_low:
        reply_text = (
            f"Harika bir soru! Formülü ezberlemek yerine zihninde somutlaştıralım.\n\n"
            f"💡 **Düşünelim:** Bir inşaatta duvar ördüğünü hayal et. 100 tuğlalık iş yaptın (Kazanılmış Değer / EV = 100). "
            f"Ama işverenden 120 tuğlalık para harcadın (Gerçekleşen Maliyet / AC = 120).\n\n"
            f"Sence maliyet sapmasını ($CV = EV - AC$) hesapladığımızda:"
        )
        guiding_questions = [
            "Harcanan para yapılan işin değerini aştığında sonuç sence pozitif mi çıkar, negatif mi?",
            "Ve bir proje yöneticisi için sonucun eksi (-) çıkması sevindirici midir, yoksa alarm zili midir?"
        ]
        suggested_replies = [
            "Negatif çıkar, yani bütçe aşımı!",
            "Pozitif çıkar, tasarruf ettik",
            "Günlük hayattan başka bir örnek ver"
        ]
        pedagogical_phase = "socratic_inquiry"

    # Analogy or general request
    elif any(k in msg_low for k in ["örnek", "analoji", "benzetme", "günlük"]):
        reply_text = (
            f"🚗 **Günlük Hayattan Bir Benzetme Yapalım:**\n\n"
            f"Tatile çıkarken navigasyonun sana '3 saatte varacaksın ve 500 TL benzin yakacaksın' dediğini düşün (Bu senin Temel Çizgin / Baseline).\n"
            f"2 saat sonra baktın ki 600 TL harcamışsın ama yolun henüz üçte birindesin!\n\n"
            f"Burada Kazanılmış Değer analizi (EVM), hız göstergen ile benzin cüzdanını aynı anda kıyaslayan akıllı göstergedir."
        )
        guiding_questions = [
            "Sence bu yolculukta hedefe varmak için gaza daha mı çok basmalısın, yoksa harcamaları kısıp rotayı mı kontrol etmelisin?",
            "Projedeki 'Düzeltici Eylem' sence hangi adıma karşılık gelir?"
        ]
        suggested_replies = [
            "Rotayı kontrol edip tasarruf etmeli",
            "Peki CPI ve SPI neyi gösterir?",
            "Videoda formülün geçtiği saniyeye gidelim"
        ]
        pedagogical_phase = "analogy_scaffolding"

    # Default Socratic Guidance
    else:
        reply_text = (
            f"'{term_name}' kavramı bu dersin en kritik bilişsel bağlantılarından biri.\n\n"
            f"Doğrudan ezberlemek yerine şu soruyla başlayalım: Bir projede sadece 'ne kadar harcandığına' (AC) bakmak "
            f"neden o projenin başarılı olup olmadığını anlamaya yetmez?"
        )
        guiding_questions = [
            "Sence harcanan paranın ne kadarlık 'tamamlanmış iş' ürettiğini (EV) bilmek neden zorunludur?",
            "Bunu bilmeden bütçenin doğru kullanıldığını söyleyebilir miyiz?"
        ]
        suggested_replies = [
            "Yetmez, çünkü ne kadar iş bittiğini bilmeliyiz!",
            "Bu konuyu formülle açıklar mısın?",
            "Videoda bu kavram nerede anlatılıyor?"
        ]
        pedagogical_phase = "open_socratic"

    # Contextual Video Timestamp connection
    time_badge = None
    if term_start is not None and term_time != "--:--":
        time_badge = {
            "start_s": term_start,
            "timestamp": term_time,
            "term": term_name,
        }

    return {
        "status": "ok",
        "reply": reply_text,
        "guiding_questions": guiding_questions,
        "suggested_replies": suggested_replies,
        "related_term": term_name,
        "time_badge": time_badge,
        "pedagogical_basis": "Vanzo et al. (ACL 2025) Socratic Scaffolding",
        "phase": pedagogical_phase,
    }
