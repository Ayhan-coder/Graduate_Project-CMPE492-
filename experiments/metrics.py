"""WER / CER with the paper's raw and normalized variants."""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str, lang: str | None = None) -> str:
    """Lowercase + strip punctuation. Turkish I/İ are mapped before .lower()."""
    text = unicodedata.normalize("NFKC", text or "")
    code = (lang or "").strip().lower()
    if code.startswith("tr"):
        # Default Unicode lower maps I→i and İ→i̇, which invents English-like
        # tokens and breaks dotted/dotless-I contrasts in Turkish.
        text = text.replace("I", "ı").replace("İ", "i")
    text = text.lower()
    text = re.sub(r"[^\w\sçğıöşüâîûÇĞİÖŞÜ]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _edit_distance(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Levenshtein with substitution / deletion / insertion counts."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "E"
            else:
                cand = [
                    (dp[i - 1][j - 1] + 1, "S"),
                    (dp[i - 1][j] + 1, "D"),
                    (dp[i][j - 1] + 1, "I"),
                ]
                dp[i][j], op[i][j] = min(cand, key=lambda x: x[0])
    s = d = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        o = op[i][j]
        if o == "E":
            i, j = i - 1, j - 1
        elif o == "S":
            s += 1
            i, j = i - 1, j - 1
        elif o == "D":
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins


def error_rate(
    reference: str,
    hypothesis: str,
    *,
    level: str = "word",
    do_norm: bool = False,
    lang: str | None = None,
) -> dict:
    ref_t = normalize(reference, lang) if do_norm else (reference or "")
    hyp_t = normalize(hypothesis, lang) if do_norm else (hypothesis or "")
    try:
        import jiwer
        if level == "word":
            res = jiwer.process_words(ref_t, hyp_t)
            n = res.hits + res.substitutions + res.deletions
            return {"S": res.substitutions, "D": res.deletions, "I": res.insertions, "N": n, "rate": res.wer}
        elif level == "char":
            ref_c = ref_t.replace(" ", "")
            hyp_c = hyp_t.replace(" ", "")
            res = jiwer.process_characters(ref_c, hyp_c)
            n = res.hits + res.substitutions + res.deletions
            return {"S": res.substitutions, "D": res.deletions, "I": res.insertions, "N": n, "rate": res.cer}
    except Exception:
        pass
    if level == "word":
        ref_u, hyp_u = ref_t.split(), hyp_t.split()
    elif level == "char":
        ref_u, hyp_u = list(ref_t.replace(" ", "")), list(hyp_t.replace(" ", ""))
    else:
        raise ValueError(level)
    n = max(len(ref_u), 1)
    s, d, ins = _edit_distance(ref_u, hyp_u)
    return {"S": s, "D": d, "I": ins, "N": len(ref_u), "rate": (s + d + ins) / n}


def pair_metrics(reference: str, hypothesis: str, lang: str | None = None) -> dict:
    return {
        "wer_raw": error_rate(reference, hypothesis, level="word", do_norm=False, lang=lang)["rate"],
        "cer_raw": error_rate(reference, hypothesis, level="char", do_norm=False, lang=lang)["rate"],
        "wer_norm": error_rate(reference, hypothesis, level="word", do_norm=True, lang=lang)["rate"],
        "cer_norm": error_rate(reference, hypothesis, level="char", do_norm=True, lang=lang)["rate"],
    }


def count_turkish_syllables(word: str) -> int:
    """Count syllables in a Turkish word. Each vowel corresponds to one syllable."""
    vowels = "aeıioöuüâîûAEIİOÖUÜÂÎÛ"
    c = sum(1 for ch in word if ch in vowels)
    return max(1, c) if any(ch.isalpha() for ch in word) else 0


def count_english_syllables(word: str) -> int:
    """Heuristic syllable count for English words."""
    w = word.lower().strip()
    if not w:
        return 0
    # Strip non-alpha
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    # Remove trailing silent e (unless ends in le)
    if w.endswith("e") and not w.endswith("le") and len(w) > 2 and w[-2] not in "aeiouy":
        w = w[:-1]
    # Count consecutive vowel groups
    matches = re.findall(r"[aeiouy]+", w)
    count = len(matches)
    return max(1, count)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences cleanly."""
    t = (text or "").strip()
    if not t:
        return []
    s = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    return s if s else [t]


def atesman_score(text: str) -> dict:
    """Calculate Ateşman Readability Index for Turkish educational text.

    Formula: S = 198.825 - 40.175 * (syllables / words) - 2.610 * (words / sentences)
    Standard categories:
      90-100: Çok Kolay (Very Easy)
      70-89:  Kolay (Easy)
      50-69:  Orta Güçlükte (Medium / CEFR B1 target)
      30-49:  Zor (Difficult)
      1-29:   Çok Zor (Very Difficult)
    """
    sentences = split_sentences(text)
    words = re.findall(r"\b[a-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛ0-9_-]+\b", text)
    if not words:
        return {
            "score": 0.0,
            "level": "Boş Metin",
            "syllables": 0,
            "words": 0,
            "sentences": 0,
            "asl": 0.0,
            "awl": 0.0,
        }

    n_sentences = max(1, len(sentences))
    n_words = len(words)
    total_syllables = sum(count_turkish_syllables(w) for w in words)

    syllables_per_word = total_syllables / n_words
    words_per_sentence = n_words / n_sentences

    score = 198.825 - (40.175 * syllables_per_word) - (2.610 * words_per_sentence)
    score_clamped = round(max(1.0, min(100.0, score)), 2)

    if score_clamped >= 90:
        level = "Çok Kolay"
    elif score_clamped >= 70:
        level = "Kolay"
    elif score_clamped >= 50:
        level = "Orta Güçlükte"
    elif score_clamped >= 30:
        level = "Zor"
    else:
        level = "Çok Zor"

    return {
        "score": score_clamped,
        "raw_score": round(score, 2),
        "level": level,
        "syllables": total_syllables,
        "words": n_words,
        "sentences": n_sentences,
        "syllables_per_word": round(syllables_per_word, 3),
        "words_per_sentence": round(words_per_sentence, 3),
    }


def flesch_score(text: str) -> dict:
    """Calculate Flesch Reading Ease & Flesch-Kincaid Grade Level for English."""
    sentences = split_sentences(text)
    words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text)
    if not words:
        return {
            "reading_ease": 0.0,
            "grade_level": 0.0,
            "level": "Empty",
            "words": 0,
            "sentences": 0,
            "syllables": 0,
        }

    n_sentences = max(1, len(sentences))
    n_words = len(words)
    total_syllables = sum(count_english_syllables(w) for w in words)

    asl = n_words / n_sentences
    asw = total_syllables / n_words

    fre = 206.835 - (1.015 * asl) - (84.6 * asw)
    fre_clamped = round(max(0.0, min(100.0, fre)), 2)

    fkgl = (0.39 * asl) + (11.8 * asw) - 15.59
    fkgl_clamped = round(max(0.0, fkgl), 2)

    if fre_clamped >= 90:
        level = "Very Easy"
    elif fre_clamped >= 80:
        level = "Easy"
    elif fre_clamped >= 70:
        level = "Fairly Easy"
    elif fre_clamped >= 60:
        level = "Standard"
    elif fre_clamped >= 50:
        level = "Fairly Difficult"
    elif fre_clamped >= 30:
        level = "Difficult"
    else:
        level = "Very Confusing"

    return {
        "reading_ease": fre_clamped,
        "grade_level": fkgl_clamped,
        "level": level,
        "words": n_words,
        "sentences": n_sentences,
        "syllables": total_syllables,
        "asl": round(asl, 2),
        "asw": round(asw, 2),
    }


def readability_score(text: str, lang: str = "tr") -> dict:
    """Unified readability interface for educational explanation evaluation."""
    l = (lang or "").strip().lower()
    if l.startswith("tr"):
        res = atesman_score(text)
        return {
            "lang": "tr",
            "metric": "atesman",
            "score": res["score"],
            "level": res["level"],
            "details": res,
        }
    else:
        res = flesch_score(text)
        return {
            "lang": "en",
            "metric": "flesch_reading_ease",
            "score": res["reading_ease"],
            "grade_level": res["grade_level"],
            "level": res["level"],
            "details": res,
        }

