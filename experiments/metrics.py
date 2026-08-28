"""WER / CER with the paper's raw and normalized variants."""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
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


def error_rate(reference: str, hypothesis: str, *, level: str = "word", do_norm: bool = False) -> dict:
    ref_t = normalize(reference) if do_norm else (reference or "")
    hyp_t = normalize(hypothesis) if do_norm else (hypothesis or "")
    if level == "word":
        ref_u, hyp_u = ref_t.split(), hyp_t.split()
    elif level == "char":
        ref_u, hyp_u = list(ref_t.replace(" ", "")), list(hyp_t.replace(" ", ""))
    else:
        raise ValueError(level)
    n = max(len(ref_u), 1)
    s, d, ins = _edit_distance(ref_u, hyp_u)
    return {"S": s, "D": d, "I": ins, "N": len(ref_u), "rate": (s + d + ins) / n}


def pair_metrics(reference: str, hypothesis: str) -> dict:
    return {
        "wer_raw": error_rate(reference, hypothesis, level="word", do_norm=False)["rate"],
        "cer_raw": error_rate(reference, hypothesis, level="char", do_norm=False)["rate"],
        "wer_norm": error_rate(reference, hypothesis, level="word", do_norm=True)["rate"],
        "cer_norm": error_rate(reference, hypothesis, level="char", do_norm=True)["rate"],
    }
