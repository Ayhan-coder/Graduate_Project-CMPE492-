"""LLM-authored practice exams (flashcards + MCQ) from lecture terms."""

from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from gui.learner import (
    build_learner_prompt_block,
    classify_mastery,
    get_term_record,
    normalize_term,
)

ROOT = Path(__file__).resolve().parent.parent

EXAM_PROMPT = (
    "You are a practice-exam author for {LANG} educational lecture content. "
    "Write a short practice exam from the domain terms below. "
    "Learner background (weight question choice toward weak/review terms):\n"
    "{LEARNER}\n"
    "Rules:\n"
    "- Write 6 to 10 questions. Prefer flashcard, mix in some mcq.\n"
    "- Prefer review-priority and still-learning terms when listed.\n"
    "- Use only facts consistent with the terms and transcript excerpt.\n"
    "- type must be \"flashcard\" or \"mcq\".\n"
    "- flashcard: fields id, type, term, front, back.\n"
    "- mcq: fields id, type, term, prompt, options (exactly 4 strings), answer_index (0-3).\n"
    "Return strictly a JSON array of objects.\n"
    "Terms:\n{TERMS}\n"
    "Transcript excerpt:\n{TRANSCRIPT}"
)

GENERIC_DISTRACTORS_EN = [
    "An unvalidated theoretical assumption with no empirical support",
    "A secondary implementation detail not covered in this curriculum",
    "A measurement index that quantifies unrelated external noise",
    "A preliminary hypothesis that was refuted in earlier experiments",
    "A deprecated procedure superseded by modern standards",
]
GENERIC_DISTRACTORS_TR = [
    "Ders kapsamında ele alınmayan ikincil bir uygulama ayrıntısı",
    "Deneysel desteği bulunmayan doğrulanmamış teorik bir varsayım",
    "Sistem dışı gürültüyü ölçen ilgisiz bir değerlendirme metriği",
    "Önceki aşamalarda elenmiş geçersiz bir başlangıç hipotezi",
    "Güncel standartların yerine geçmiş eski bir yöntem",
]


def _exams_dir() -> Path:
    override = os.environ.get("EDUTERM_EXAMS_DIR")
    if override:
        path = Path(override)
    else:
        path = ROOT / "gui" / "data" / "exams"
    path.mkdir(parents=True, exist_ok=True)
    return path


def exam_path(exam_id: str) -> Path:
    safe = re.sub(r"[^0-9A-Za-z_-]", "", exam_id or "")
    if not safe:
        raise ValueError("Invalid exam id")
    return _exams_dir() / f"{safe}.json"


def save_exam(exam: dict) -> None:
    path = exam_path(str(exam["exam_id"]))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(exam, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_exam(exam_id: str) -> Optional[dict]:
    try:
        path = exam_path(exam_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def parse_exam_questions(text: str) -> list[dict]:
    start, end = (text or "").find("["), (text or "").rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    cleaned: list[dict] = []
    for i, raw in enumerate(data, start=1):
        q = _normalize_question(raw, i)
        if q:
            cleaned.append(q)
    return cleaned


def _normalize_question(raw: object, index: int) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    qtype = str(raw.get("type") or "").strip().lower()
    term = str(raw.get("term") or "").strip()
    qid = str(raw.get("id") or f"q{index}").strip() or f"q{index}"
    if qtype == "mcq":
        options = raw.get("options")
        if not isinstance(options, list) or len(options) < 2:
            return None
        opts = [str(o).strip() for o in options if str(o).strip()]
        if len(opts) < 2:
            return None
        opts = opts[:4]
        while len(opts) < 4:
            opts.append(opts[-1])
        try:
            answer_index = int(raw.get("answer_index"))
        except (TypeError, ValueError):
            return None
        if answer_index < 0 or answer_index >= len(opts):
            return None
        prompt = str(raw.get("prompt") or raw.get("front") or "").strip()
        if not prompt:
            prompt = f'What does "{term}" mean?' if term else "Choose the best answer."
        return {
            "id": qid,
            "type": "mcq",
            "term": term or f"item-{index}",
            "prompt": prompt,
            "options": opts,
            "answer_index": answer_index,
        }
    if qtype in ("flashcard", "card", ""):
        front = str(raw.get("front") or raw.get("prompt") or term).strip()
        back = str(raw.get("back") or raw.get("explanation") or "").strip()
        if not front or not back:
            return None
        return {
            "id": qid,
            "type": "flashcard",
            "term": term or front,
            "front": front,
            "back": back,
            "prompt": front,
        }
    return None


def prioritize_terms(terms: list[dict], profile: Optional[dict], limit: int = 10) -> list[dict]:
    """Weak and unseen terms first so exams follow the learner background."""
    rank = {"review": 0, "learning": 1, "new": 2, "known": 3}
    scored = []
    for item in terms:
        if not isinstance(item, dict):
            continue
        term = (item.get("term") or "").strip()
        expl = (item.get("explanation") or "").strip()
        if not term or not expl:
            continue
        focus = item.get("focus")
        rec = get_term_record(profile or {}, term) if profile else None
        if rec:
            focus = classify_mastery(rec.get("mastery"), rec.get("seen", 0))
            mastery = float(rec.get("mastery", 0.5))
        else:
            focus = focus or "new"
            mastery = 0.5 if focus != "new" else 0.55
        scored.append((rank.get(focus, 2), mastery, item))
    scored.sort(key=lambda row: (row[0], row[1]))
    picked: list[dict] = []
    seen: set[str] = set()
    for _, _, item in scored:
        key = normalize_term(item.get("term") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        picked.append(item)
        if len(picked) >= limit:
            break
    return picked


def fallback_exam_from_terms(
    terms: list[dict],
    lang: str = "en",
    profile: Optional[dict] = None,
    limit: int = 8,
) -> list[dict]:
    """Deterministic flashcard + MCQ exam when the LLM is unavailable."""
    selected = prioritize_terms(terms, profile, limit=max(limit, 4))
    if not selected:
        return []
    is_tr = (lang or "").lower().startswith("tr")
    generics = GENERIC_DISTRACTORS_TR if is_tr else GENERIC_DISTRACTORS_EN
    questions: list[dict] = []
    for i, item in enumerate(selected, start=1):
        # Flashcards dominate; every 3rd item is an MCQ for an objective score.
        if i % 3 == 0 and len(selected) >= 3:
            questions.append(_mcq_from_term(item, selected, i, generics, is_tr))
        else:
            term = item["term"]
            questions.append(
                {
                    "id": f"q{i}",
                    "type": "flashcard",
                    "term": term,
                    "front": term,
                    "back": item.get("explanation") or "",
                    "prompt": term,
                }
            )
    return questions


def _mcq_from_term(item: dict, pool: list[dict], index: int, generics: list[str], is_tr: bool) -> dict:
    term = item["term"]
    correct = (item.get("explanation") or "").strip()
    if not correct:
        correct = f"{term} kavramının ders bağlamındaki pedagojik tanımı." if is_tr else f"The pedagogical definition of {term}."
    others = [
        (t.get("explanation") or "").strip()
        for t in pool
        if normalize_term(t.get("term") or "") != normalize_term(term)
        and (t.get("explanation") or "").strip()
        and (t.get("explanation") or "").strip() != correct
    ]
    distractors = list(dict.fromkeys(others))[:3]
    gi = 0
    while len(distractors) < 3 and gi < len(generics):
        g = generics[gi]
        if g != correct and g not in distractors:
            distractors.append(g)
        gi += 1
    # Fallback to ensure exactly 3 distinct distractors (4 options total)
    fallback_idx = 1
    while len(distractors) < 3:
        placeholder = f"Seçenek {fallback_idx}: İlgisiz kavram tanımı" if is_tr else f"Option {fallback_idx}: Alternate concept definition"
        if placeholder not in distractors and placeholder != correct:
            distractors.append(placeholder)
        fallback_idx += 1

    options = [correct] + distractors[:3]
    rng = random.Random(normalize_term(term) + str(index))
    order = list(range(len(options)))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    answer_index = shuffled.index(correct)
    if is_tr:
        prompt = f'Bu derste "{term}" ne anlama gelir?'
    else:
        prompt = f'What does "{term}" mean in this lecture?'
    return {
        "id": f"q{index}",
        "type": "mcq",
        "term": term,
        "prompt": prompt,
        "options": shuffled,
        "answer_index": answer_index,
    }


def generate_practice_exam(
    terms: list[dict],
    transcript: str = "",
    lang: str = "en",
    model_id: str = "Qwen/Qwen3.5-4B",
    profile: Optional[dict] = None,
    use_llm: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Build an exam. Tries the lecture LLM first; falls back to term cards."""
    usable = [
        t
        for t in (terms or [])
        if isinstance(t, dict) and (t.get("term") or "").strip() and (t.get("explanation") or "").strip()
    ]
    if not usable:
        raise ValueError("No extracted terms available to build a practice exam.")

    questions: list[dict] = []
    source = "fallback"
    raw = ""
    if use_llm:
        if progress_cb:
            progress_cb(f"Writing practice exam with {model_id}...")
        learner_block = build_learner_prompt_block(profile or {}, usable) or "No prior exam history."
        lang_label = "Turkish" if (lang or "").lower().startswith("tr") else "English"
        term_lines = "\n".join(
            f"- {t.get('term')}: {t.get('explanation')}" for t in prioritize_terms(usable, profile, limit=12)
        )
        excerpt = " ".join((transcript or "").split()[:280])
        prompt = EXAM_PROMPT.format(
            LANG=lang_label,
            LEARNER=learner_block,
            TERMS=term_lines,
            TRANSCRIPT=excerpt or "(no transcript excerpt)",
        )
        try:
            from run_llm import generate_hf

            raw = generate_hf(model_id, prompt, max_new_tokens=512)
            questions = parse_exam_questions(raw)
        except Exception:
            questions = []
        if questions:
            source = "llm"

    if not questions:
        if progress_cb:
            progress_cb("Building flashcard exam from extracted terms...")
        questions = fallback_exam_from_terms(usable, lang=lang, profile=profile)
        source = "fallback"

    if not questions:
        raise ValueError("Could not build a practice exam from these terms.")

    exam_id = uuid.uuid4().hex[:16]
    return {
        "exam_id": exam_id,
        "source": source,
        "lang": lang,
        "model_id": model_id if source == "llm" else None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "questions": questions,
        "raw": raw,
        "submitted": False,
        "result": None,
    }


def public_questions(questions: list[dict]) -> list[dict]:
    """Strip scoring keys before sending an exam to the browser."""
    out = []
    for q in questions:
        item = {
            "id": q.get("id"),
            "type": q.get("type"),
            "term": q.get("term"),
            "prompt": q.get("prompt") or q.get("front") or "",
        }
        if q.get("type") == "mcq":
            item["options"] = list(q.get("options") or [])
        else:
            item["front"] = q.get("front") or item["prompt"]
            item["back"] = q.get("back") or ""
        out.append(item)
    return out


def score_exam(questions: list[dict], answers: list[dict]) -> dict:
    by_id = {str(q.get("id")): q for q in questions if isinstance(q, dict) and q.get("id")}
    graded = []
    for raw in answers or []:
        if not isinstance(raw, dict):
            continue
        qid = str(raw.get("question_id") or raw.get("id") or "")
        q = by_id.get(qid)
        if not q:
            continue
        qtype = q.get("type")
        correct = False
        selected = None
        if qtype == "mcq":
            try:
                selected = int(raw.get("selected_index"))
            except (TypeError, ValueError):
                selected = None
            correct = selected is not None and selected == int(q.get("answer_index"))
        else:
            grade = str(raw.get("self_grade") or "").strip().lower()
            selected = grade
            correct = grade in ("know", "known", "yes", "correct")
        graded.append(
            {
                "question_id": qid,
                "type": qtype,
                "term": q.get("term"),
                "correct": correct,
                "selected": selected,
                "prompt": q.get("prompt") or q.get("front"),
                "expected": q.get("options", [None])[q["answer_index"]]
                if qtype == "mcq" and isinstance(q.get("answer_index"), int)
                else q.get("back"),
            }
        )

    total = len(graded)
    n_correct = sum(1 for g in graded if g["correct"])
    score = (n_correct / total) if total else 0.0
    weak = [g["term"] for g in graded if not g["correct"] and g.get("term")]
    strong = [g["term"] for g in graded if g["correct"] and g.get("term")]
    return {
        "answers": graded,
        "correct": n_correct,
        "total": total,
        "score": round(score, 4),
        "weak_terms": weak,
        "strong_terms": strong,
    }


def analysis_after_exam(scored: dict, adaptive: dict) -> str:
    total = int(scored.get("total") or 0)
    n_ok = int(scored.get("correct") or 0)
    pct = float(scored.get("score") or 0)
    weak = [t for t in (scored.get("weak_terms") or []) if t]
    if total == 0:
        return "No answers were recorded."
    parts = [f"You scored {n_ok}/{total} ({pct:.0%})."]
    if weak:
        parts.append("Missed terms (higher weight on the next analysis): " + ", ".join(weak[:8]) + ".")
    else:
        parts.append("All items were marked correct; known terms will be shorter next time.")
    if adaptive.get("exams_taken"):
        parts.append(
            f"Learner history is now {int(adaptive['exams_taken'])} exam(s), "
            f"average {float(adaptive.get('avg_score') or 0):.0%}."
        )
    return " ".join(parts)


# --------------------------------------------------------------------------
# Cognitive Science Research on Student Attention Span & Interpolated Testing
# --------------------------------------------------------------------------
ATTENTION_MILESTONES = [
    {
        "seconds": 180,
        "minute_label": "3. Dakika (180 sn)",
        "source": "Risko et al. (2012) & Bunce et al. (2010)",
        "citation": "Erken Zihin Kayması (Mind Wandering) Onset Noktası: Araştırmalar video dersin 3. dakikasından itibaren ilk dikkat dalgalanmasının başladığını gösterir.",
    },
    {
        "seconds": 300,
        "minute_label": "5. Dakika (300 sn)",
        "source": "Szpunar & Schacter (PNAS 2013 - Harvard)",
        "citation": "Optimal İnterpolated Test Eşiği: 5. dakikada verilen kısa pop-quiz, zihin kaymasını %40'tan %18'e indirir ve dikkati sıfırlar.",
    },
    {
        "seconds": 360,
        "minute_label": "6. Dakika (360 sn)",
        "source": "Guo, Kim, Rubin (ACM L@S 2014 - edX 6.9M Video)",
        "citation": "Kritik Dikkat Kırılma Sınırı: edX verilerine göre online ders videolarında medyan odak 6. dakikada tepe yapar ve ardından hızla çöker.",
    },
    {
        "seconds": 600,
        "minute_label": "10. Dakika (600 sn)",
        "source": "Bradbury (2016) / Wilson & Korn (2007)",
        "citation": "İkincil Bilişsel Yorgunluk Eşiği: Uzun derslerde 10. dakikadan sonra pasif dinlemede derin dikkat çöküşü yaşanır.",
    },
]

ATTENTION_RESEARCH_INFO = {
    "title": "Ders Videolarında Öğrenci Dikkat Dalgalanmaları ve İnterpolated Testing Araştırmaları",
    "sources": [
        {
            "id": "guo_2014",
            "authors": "Philip J. Guo, Juho Kim, Rob Rubin",
            "year": 2014,
            "title": "How Video Production Affects Student Engagement: An Empirical Study of MOOC Videos",
            "venue": "ACM Conference on Learning at Scale (L@S '14)",
            "key_finding": "6.9 milyon edX video izleme oturumu analizi: Öğrencilerin video derslere ortalama odaklanma süresi en fazla 6 dakikadır (360 sn). Video uzunluğu ne olursa olsun medyan etkileşim 6. dakikadan sonra hızla düşmektedir.",
            "critical_seconds": 360,
            "drop_range": "180s - 360s (3 - 6 dk)",
        },
        {
            "id": "szpunar_2013",
            "authors": "Karl K. Szpunar, Nova Z. Khan, Daniel L. Schacter",
            "year": 2013,
            "title": "Interpolated memory tests reduce mind wandering and improve learning of online lectures",
            "venue": "Proceedings of the National Academy of Sciences (PNAS)",
            "key_finding": "Ders videosu izlerken zihin kayması (mind wandering) 4-5. dakikalardan itibaren %40'ın üzerine çıkar. Her 5 dakikada bir (300 sn) araya serpiştirilen kısa bellek testleri (pop-quiz), zihin kaymasını %18'e indirmiş ve kalıcılığı ikiye katlamıştır.",
            "critical_seconds": 300,
            "drop_range": "240s - 300s (4 - 5 dk)",
        },
        {
            "id": "bunce_2010",
            "authors": "Diane M. Bunce, Elizabeth A. Flaherty, Kyle D. VandenPlas",
            "year": 2010,
            "title": "Measuring Student Attention in Lectures with Clickers",
            "venue": "Journal of Chemical Education",
            "key_finding": "Öğrencilerin saniye bazlı dikkat seviyeleri ilk 4.5-5. dakikada belirgin bir dalgalanma ve kopma yaşamakta, ardından interaktif müdahale olmazsa her 2-3 dakikada bir dikkat kopması tekrarlamaktadır.",
            "critical_seconds": 270,
            "drop_range": "240s - 300s (4 - 5 dk)",
        },
        {
            "id": "risko_2012",
            "authors": "Evan F. Risko, Nicola Anderson, Alyssa Sarwal, Megan Engelhardt, Alan Kingstone",
            "year": 2012,
            "title": "Everyday Attention: Variation in Mind Wandering & Memory in a Lecture",
            "venue": "Applied Cognitive Psychology",
            "key_finding": "Zaman ilerledikçe zihin kayması doğrusal olarak artar. İlk kaymalar 3. dakikadan (180 sn) itibaren gözlemlenir.",
            "critical_seconds": 180,
            "drop_range": "180s - 240s (3 - 4 dk)",
        },
        {
            "id": "bradbury_2016",
            "authors": "Neil A. Bradbury",
            "year": 2016,
            "title": "Attention span during lectures: 8 seconds, 10 minutes, or more?",
            "venue": "Advances in Physiology Education",
            "key_finding": "Pasif dinlemede 10-15. dakikada derin dikkat çöküşü yaşanır. Düzenli soru ve aktif hatırlama (retrieval practice) ile dikkat periyodu sıfırlanabilmektedir.",
            "critical_seconds": 600,
            "drop_range": "600s - 720s (10 - 12 dk)",
        },
    ],
    "default_thresholds_s": [180, 300, 360, 600],
}


def make_popquiz_question(
    terms: list[dict],
    current_time_s: float = 0.0,
    lang: str = "en",
    profile: Optional[dict] = None,
) -> Optional[dict]:
    """Build a single targeted MCQ pop-quiz based on terms taught up to current playback second."""
    if not terms:
        return None

    # Filter terms that occurred around or before current_time_s if timestamps are known
    earlier_terms = []
    for t in terms:
        if not isinstance(t, dict):
            continue
        start_s = t.get("start_s")
        if start_s is not None:
            try:
                if float(start_s) <= (current_time_s + 5.0):
                    earlier_terms.append(t)
            except (ValueError, TypeError):
                pass

    pool = earlier_terms if len(earlier_terms) >= 2 else terms
    selected = prioritize_terms(pool, profile, limit=4)
    if not selected:
        selected = prioritize_terms(terms, profile, limit=4)
    if not selected:
        return None

    is_tr = (lang or "").lower().startswith("tr")
    generics = GENERIC_DISTRACTORS_TR if is_tr else GENERIC_DISTRACTORS_EN
    target_item = selected[0]
    mcq = _mcq_from_term(target_item, terms, 1, generics, is_tr)

    closest_m = min(
        ATTENTION_MILESTONES,
        key=lambda m: abs(m["seconds"] - current_time_s),
    ) if ATTENTION_MILESTONES else None

    return {
        "question": mcq,
        "current_time_s": current_time_s,
        "term": target_item.get("term"),
        "explanation": target_item.get("explanation"),
        "milestone": closest_m,
    }

