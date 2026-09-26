"""Local learner profile: exam history, per-term mastery, and LLM prompt weight."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
_LOCK = threading.Lock()

REVIEW_THRESHOLD = 0.45
KNOWN_THRESHOLD = 0.75
CORRECT_GAIN = 0.30
INCORRECT_DECAY = 0.42
MAX_EXAM_SUMMARIES = 40


def _profile_path() -> Path:
    override = os.environ.get("EDUTERM_LEARNER_PATH")
    if override:
        path = Path(override)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path = ROOT / "gui" / "data" / "learner_profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalize_term(term: str) -> str:
    return " ".join((term or "").strip().lower().split())


def empty_profile() -> dict:
    return {
        "learner_id": "local",
        "updated_at": None,
        "terms": {},
        "exams": [],
        "stats": {
            "exams_taken": 0,
            "avg_score": 0.0,
            "last_score": None,
            "weak_count": 0,
        },
    }


def load_profile() -> dict:
    path = _profile_path()
    if not path.exists():
        return empty_profile()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_profile()
    if not isinstance(data, dict):
        return empty_profile()
    base = empty_profile()
    base.update({k: data.get(k, base[k]) for k in base})
    if not isinstance(base.get("terms"), dict):
        base["terms"] = {}
    if not isinstance(base.get("exams"), list):
        base["exams"] = []
    if not isinstance(base.get("stats"), dict):
        base["stats"] = empty_profile()["stats"]
    return base


def save_profile(profile: dict) -> None:
    profile = dict(profile)
    profile["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = _profile_path()
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(profile, ensure_ascii=False, indent=2)
    with _LOCK:
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)


def classify_mastery(mastery: Optional[float], seen: int = 0) -> str:
    if mastery is None or seen <= 0:
        return "new"
    if mastery < REVIEW_THRESHOLD:
        return "review"
    if mastery >= KNOWN_THRESHOLD:
        return "known"
    return "learning"


def update_mastery(profile: dict, term: str, correct: bool, video_id: str = "") -> dict:
    """Update one term record in-place and return it."""
    key = normalize_term(term)
    if not key:
        return {}
    rec = profile.setdefault("terms", {}).get(key) or {
        "term": (term or "").strip() or key,
        "seen": 0,
        "correct": 0,
        "incorrect": 0,
        "mastery": 0.5,
        "last_correct": None,
        "videos": [],
    }
    rec["term"] = (term or "").strip() or rec.get("term") or key
    rec["seen"] = int(rec.get("seen") or 0) + 1
    mastery = float(rec.get("mastery", 0.5))
    if correct:
        rec["correct"] = int(rec.get("correct") or 0) + 1
        rec["last_correct"] = True
        mastery = min(1.0, mastery + (1.0 - mastery) * CORRECT_GAIN)
    else:
        rec["incorrect"] = int(rec.get("incorrect") or 0) + 1
        rec["last_correct"] = False
        mastery = max(0.0, mastery * (1.0 - INCORRECT_DECAY))
    rec["mastery"] = round(mastery, 4)
    rec["focus"] = classify_mastery(rec["mastery"], rec["seen"])
    if video_id:
        videos = list(rec.get("videos") or [])
        if video_id not in videos:
            videos.append(video_id)
        rec["videos"] = videos[-12:]
    profile["terms"][key] = rec
    return rec


def _recompute_stats(profile: dict) -> None:
    exams = profile.get("exams") or []
    scores = [float(e["score"]) for e in exams if isinstance(e, dict) and e.get("score") is not None]
    terms = profile.get("terms") or {}
    weak = sum(1 for rec in terms.values() if classify_mastery(rec.get("mastery"), rec.get("seen", 0)) == "review")
    profile["stats"] = {
        "exams_taken": len(exams),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "last_score": scores[-1] if scores else None,
        "weak_count": weak,
    }


def record_exam(profile: dict, exam_result: dict) -> dict:
    """Append a compact exam summary and fold per-term answers into mastery."""
    video_id = str(exam_result.get("video_id") or "")
    for ans in exam_result.get("answers") or []:
        if not isinstance(ans, dict):
            continue
        term = ans.get("term") or ""
        if not term:
            continue
        update_mastery(profile, term, bool(ans.get("correct")), video_id=video_id)

    summary = {
        "exam_id": exam_result.get("exam_id"),
        "job_id": exam_result.get("job_id"),
        "video_id": video_id,
        "video_title": exam_result.get("video_title") or "",
        "score": exam_result.get("score"),
        "correct": exam_result.get("correct"),
        "total": exam_result.get("total"),
        "weak_terms": exam_result.get("weak_terms") or [],
        "created_at": exam_result.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    exams = list(profile.get("exams") or [])
    exams.append(summary)
    profile["exams"] = exams[-MAX_EXAM_SUMMARIES:]
    _recompute_stats(profile)
    return profile


def get_term_record(profile: dict, term: str) -> Optional[dict]:
    rec = (profile.get("terms") or {}).get(normalize_term(term))
    return rec if isinstance(rec, dict) else None


def annotate_terms(terms: list[dict], profile: dict) -> list[dict]:
    """Attach mastery / focus to extracted terms from the saved learner profile."""
    for item in terms:
        if not isinstance(item, dict):
            continue
        rec = get_term_record(profile, item.get("term") or "")
        if rec:
            item["mastery"] = rec.get("mastery")
            item["seen"] = rec.get("seen", 0)
            item["focus"] = classify_mastery(rec.get("mastery"), rec.get("seen", 0))
        else:
            item.setdefault("mastery", None)
            item.setdefault("seen", 0)
            item["focus"] = "new"
    return terms


def ranked_term_records(profile: dict) -> list[dict]:
    rows = [rec for rec in (profile.get("terms") or {}).values() if isinstance(rec, dict)]
    rows.sort(key=lambda r: (float(r.get("mastery", 0.5)), -int(r.get("incorrect") or 0)))
    return rows


def build_learner_prompt_block(profile: dict, current_terms: Optional[list[dict]] = None) -> str:
    """Compact student-background block injected into GUI LLM prompts."""
    stats = (profile.get("stats") or {})
    records = ranked_term_records(profile)
    if not records and not stats.get("exams_taken"):
        return ""

    current_keys = set()
    if current_terms:
        current_keys = {normalize_term(t.get("term", "")) for t in current_terms if isinstance(t, dict)}

    weak = [r for r in records if classify_mastery(r.get("mastery"), r.get("seen", 0)) == "review"][:8]
    learning = [r for r in records if classify_mastery(r.get("mastery"), r.get("seen", 0)) == "learning"][:6]
    strong = [r for r in records if classify_mastery(r.get("mastery"), r.get("seen", 0)) == "known"][:6]

    def _fmt(rec: dict) -> str:
        flag = ""
        if current_keys and normalize_term(rec.get("term", "")) in current_keys:
            flag = " [in this lecture]"
        return (
            f"- {rec.get('term')} (mastery {float(rec.get('mastery', 0)):.2f}, "
            f"missed {int(rec.get('incorrect') or 0)}x, seen {int(rec.get('seen') or 0)}x){flag}"
        )

    lines = [
        "Learner background — weight this when choosing what to emphasize.",
        (
            f"Practice exams taken: {int(stats.get('exams_taken') or 0)}. "
            f"Average score: {float(stats.get('avg_score') or 0):.0%}."
        ),
    ]
    last = stats.get("last_score")
    if last is not None:
        lines.append(f"Most recent exam score: {float(last):.0%}.")
    if weak:
        lines.append("Review-priority terms (HIGHER weight — clearer B1 explanation, do not skip):")
        lines.extend(_fmt(r) for r in weak)
    if learning:
        lines.append("Still-learning terms (medium weight):")
        lines.extend(_fmt(r) for r in learning)
    if strong:
        lines.append("Already-known terms (LOWER weight — keep explanations short):")
        lines.extend(_fmt(r) for r in strong)
    if not weak and not learning and not strong:
        return ""
    lines.append(
        "Give review-priority terms more care. Do not invent facts that are not in the transcript."
    )
    return "\n".join(lines)


def build_adaptive_analysis(profile: dict, current_terms: Optional[list[dict]] = None) -> dict:
    """Structured analysis used by the GUI and included in pipeline results."""
    stats = profile.get("stats") or empty_profile()["stats"]
    records = ranked_term_records(profile)
    exams_taken = int(stats.get("exams_taken") or 0)
    weak = [
        {"term": r.get("term"), "mastery": r.get("mastery"), "incorrect": r.get("incorrect", 0)}
        for r in records
        if classify_mastery(r.get("mastery"), r.get("seen", 0)) == "review"
    ][:10]
    strong = [
        {"term": r.get("term"), "mastery": r.get("mastery")}
        for r in records
        if classify_mastery(r.get("mastery"), r.get("seen", 0)) == "known"
    ][:10]

    video_focus = []
    if current_terms:
        for item in current_terms:
            if not isinstance(item, dict):
                continue
            focus = item.get("focus") or "new"
            if focus in ("review", "learning"):
                video_focus.append(
                    {
                        "term": item.get("term"),
                        "focus": focus,
                        "mastery": item.get("mastery"),
                    }
                )

    weight_applied = exams_taken > 0 and bool(weak or strong or video_focus)
    if not exams_taken:
        summary = "No practice exams yet. Take an exam after this lecture so later analyses can weight weak terms."
    elif video_focus:
        names = ", ".join(str(x["term"]) for x in video_focus[:5] if x.get("term"))
        summary = (
            f"{exams_taken} exam(s), average {float(stats.get('avg_score') or 0):.0%}. "
            f"This lecture overlaps review terms: {names}."
        )
    elif weak:
        names = ", ".join(str(x["term"]) for x in weak[:5] if x.get("term"))
        summary = (
            f"{exams_taken} exam(s), average {float(stats.get('avg_score') or 0):.0%}. "
            f"Review-priority background: {names}."
        )
    else:
        summary = (
            f"{exams_taken} exam(s), average {float(stats.get('avg_score') or 0):.0%}. "
            "No weak terms on record; explanations stay brief for known items."
        )

    return {
        "has_history": exams_taken > 0,
        "exams_taken": exams_taken,
        "avg_score": stats.get("avg_score") or 0.0,
        "last_score": stats.get("last_score"),
        "weak_count": int(stats.get("weak_count") or len(weak)),
        "weak_terms": weak,
        "strong_terms": strong,
        "video_focus": video_focus,
        "weight_applied": weight_applied,
        "summary": summary,
    }


def public_profile(profile: Optional[dict] = None) -> dict:
    profile = profile if profile is not None else load_profile()
    analysis = build_adaptive_analysis(profile)
    return {
        "learner_id": profile.get("learner_id") or "local",
        "updated_at": profile.get("updated_at"),
        "stats": profile.get("stats") or empty_profile()["stats"],
        "recent_exams": list(reversed(profile.get("exams") or []))[:12],
        "analysis": analysis,
    }
