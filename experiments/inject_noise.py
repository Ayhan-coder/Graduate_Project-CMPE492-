"""inject_noise.py: Inject controlled speech recognition errors at target WER levels.

Generates realistic ASR-style acoustic and typographical corruptions (substitutions,
deletions, insertions) calibrated to fixed WER levels (e.g., 5%, 10%, 20%, 30%, 40%)
to test the central hypothesis: How does transcription error degrade downstream Term F1?
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import error_rate, normalize


# Common Turkish acoustic/homophone confusions and ASR insertion tokens
TR_ACOUSTIC_CONFUSIONS: dict[str, list[str]] = {
    "bir": ["bi", "ve", "bu"],
    "ve": ["de", "ile", "veya"],
    "ile": ["le", "ve", "gibi"],
    "için": ["çin", "işin", "içinde"],
    "olarak": ["olup", "olarak ta", "olaraktan"],
    "proje": ["proce", "projede", "parça"],
    "yönetimi": ["yönelimi", "yöntemi", "yönetim"],
    "yönetici": ["yöneltici", "yöneticiye"],
    "değer": ["değeri", "dere", "değerler"],
    "maliyet": ["mahliyet", "maliyeti", "malik"],
    "çizelge": ["çizge", "çizerge", "takvim"],
    "bütçe": ["bütçesi", "büçe", "güç"],
    "izleme": ["izlem", "işleme", "izler"],
    "kontrol": ["konturol", "kontrollü"],
    "veri": ["biri", "verim", "veriyi"],
    "model": ["modem", "modül", "modeli"],
    "analiz": ["anlat", "analizi", "araz"],
    "turizm": ["trizm", "turizim"],
    "turist": ["turistlik", "türis"],
}

# Common English acoustic/homophone confusions and ASR insertion tokens
EN_ACOUSTIC_CONFUSIONS: dict[str, list[str]] = {
    "the": ["a", "their", "they"],
    "a": ["the", "an", "uh"],
    "to": ["too", "two", "do"],
    "and": ["an", "in", "end"],
    "of": ["off", "have", "on"],
    "in": ["and", "an", "on"],
    "is": ["as", "his", "it's"],
    "that": ["dat", "this", "what"],
    "for": ["four", "fore", "far"],
    "agile": ["idle", "angle", "ajay"],
    "scrum": ["scum", "sum", "scram"],
    "sprint": ["spirit", "split", "print"],
    "backlog": ["black log", "back lot", "bag log"],
    "team": ["deem", "teen", "time"],
    "model": ["modal", "bottle", "module"],
    "learning": ["yearning", "earning", "learn in"],
    "training": ["draining", "chaining"],
    "feature": ["future", "feather"],
}


def _corrupt_word(word: str, lang: str, confusions: dict[str, list[str]]) -> str:
    """Corrupt a single word via acoustic/homophone dictionary or character mutation."""
    w_lower = word.lower()
    if w_lower in confusions:
        cand = random.choice(confusions[w_lower])
        return cand.capitalize() if word.istitle() else cand

    # Character level acoustic / typo perturbation
    chars = list(word)
    if len(chars) <= 1:
        return word

    idx = random.randint(0, len(chars) - 1)
    ch = chars[idx].lower()

    if lang.startswith("tr"):
        char_map = {"ı": "i", "i": "ı", "ş": "s", "s": "ş", "ç": "c", "c": "ç", "ğ": "g", "g": "ğ", "ö": "o", "ü": "u"}
    else:
        char_map = {"c": "k", "k": "c", "ph": "f", "f": "ph", "s": "z", "z": "s", "d": "t", "t": "d", "b": "p", "p": "b"}

    if ch in char_map:
        chars[idx] = char_map[ch]
    else:
        # Random deletion or duplicate
        if random.random() < 0.5:
            chars.pop(idx)
        else:
            chars.insert(idx, chars[idx])

    return "".join(chars)


def inject_noise_once(
    text: str,
    target_wer: float,
    lang: str = "tr",
    sub_ratio: float = 0.65,
    del_ratio: float = 0.20,
    ins_ratio: float = 0.15,
    seed: int | None = None,
) -> str:
    """Inject errors on a list of words targeting an approximate WER."""
    if target_wer <= 0.001:
        return text

    if seed is not None:
        random.seed(seed)

    words = text.split()
    if not words:
        return text

    n_words = len(words)
    # Total error operations to apply
    target_errors = max(1, int(round(n_words * target_wer)))

    confusions = TR_ACOUSTIC_CONFUSIONS if lang.startswith("tr") else EN_ACOUSTIC_CONFUSIONS
    common_insertions = list(confusions.keys()) + (["yani", "şey", "ııı", "fakat"] if lang.startswith("tr") else ["um", "uh", "like", "you know"])

    # Error budget distribution
    n_sub = int(round(target_errors * sub_ratio))
    n_del = int(round(target_errors * del_ratio))
    n_ins = target_errors - n_sub - n_del

    out_words = list(words)

    # 1. Substitutions
    sub_indices = random.sample(range(len(out_words)), min(n_sub, len(out_words)))
    for idx in sub_indices:
        out_words[idx] = _corrupt_word(out_words[idx], lang, confusions)

    # 2. Deletions (traverse descending to preserve indices)
    if n_del > 0 and len(out_words) > n_del + 2:
        del_indices = sorted(random.sample(range(len(out_words)), n_del), reverse=True)
        for idx in del_indices:
            out_words.pop(idx)

    # 3. Insertions
    for _ in range(n_ins):
        ins_pos = random.randint(0, len(out_words))
        ins_word = random.choice(common_insertions)
        out_words.insert(ins_pos, ins_word)

    return " ".join(out_words)


def calibrate_noise(
    text: str,
    target_wer: float,
    lang: str = "tr",
    max_iters: int = 15,
    tolerance: float = 0.015,
    seed: int | None = None,
) -> tuple[str, float, dict]:
    """Iteratively calibrate error injection until actual WER matches target within tolerance."""
    if target_wer <= 0.001:
        return text, 0.0, {"S": 0, "D": 0, "I": 0, "N": len(text.split()), "rate": 0.0}

    best_hyp = text
    best_wer = 0.0
    best_diff = float("inf")
    best_res: dict = {}

    current_target = target_wer
    base_seed = seed or 42

    for it in range(max_iters):
        hyp = inject_noise_once(text, target_wer=current_target, lang=lang, seed=base_seed + it * 7)
        res = error_rate(text, hyp, level="word", do_norm=True, lang=lang)
        actual_wer = res["rate"]
        diff = abs(actual_wer - target_wer)

        if diff < best_diff:
            best_diff = diff
            best_hyp = hyp
            best_wer = actual_wer
            best_res = res

        if diff <= tolerance:
            break

        # Adjust current target based on over/under shooting
        if actual_wer < target_wer:
            current_target += (target_wer - actual_wer) * 0.7
        else:
            current_target -= (actual_wer - target_wer) * 0.7
        current_target = max(0.01, min(1.0, current_target))

    return best_hyp, round(best_wer, 4), best_res


def generate_synthetic_noise_ladder(
    reference: str,
    lang: str = "tr",
    wer_levels: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.40),
    seed: int = 42,
) -> list[dict]:
    """Generate a progression of corrupted transcripts across fixed WER levels."""
    results = []
    # Include perfect gold reference as baseline (WER = 0.0)
    results.append({
        "target_wer": 0.0,
        "actual_wer": 0.0,
        "hypothesis": reference,
        "details": {"S": 0, "D": 0, "I": 0, "N": len(reference.split()), "rate": 0.0},
    })

    for target in wer_levels:
        hyp, actual, details = calibrate_noise(reference, target_wer=target, lang=lang, seed=seed)
        results.append({
            "target_wer": target,
            "actual_wer": actual,
            "hypothesis": hyp,
            "details": details,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject controlled ASR noise at target WER levels")
    parser.add_argument("--text", type=str, default=None, help="Direct input text")
    parser.add_argument("--file", type=Path, default=None, help="Input text file path")
    parser.add_argument("--wer", type=float, default=0.10, help="Target WER (e.g. 0.05, 0.10, 0.20)")
    parser.add_argument("--ladder", action="store_true", help="Generate full WER ladder (5%, 10%, 20%, 30%, 40%)")
    parser.add_argument("--lang", default="tr", choices=["tr", "en"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", help="Run self-test calibration")
    args = parser.parse_args()

    if args.test:
        print("Running noise calibration self-test...")
        sample_tr = (
            "Proje yönetimi belirli hedeflere ulaşmak için kaynakların planlanması ve kontrol edilmesidir. "
            "Kazanılmış değer yönetimi projenin bütçe ve takvim performansını ölçmede temel yöntemdir. "
            "Maliyet sapması ve çizelge sapması erken uyarı sağlayarak yöneticinin düzeltici eylem almasını sağlar."
        )
        sample_en = (
            "Agile methodology emphasizes iterative development and rapid feedback cycles. "
            "The scrum master coordinates daily standup meetings and removes blockers for the cross functional team. "
            "Sprint backlog items are estimated in story points and delivered as working increments."
        )

        for lang, text in [("tr", sample_tr), ("en", sample_en)]:
            print(f"\n--- Testing Language: {lang.upper()} ---")
            for target in [0.05, 0.10, 0.20, 0.30]:
                hyp, act, details = calibrate_noise(text, target, lang=lang, seed=args.seed)
                print(f"Target WER: {target*100:4.1f}% | Actual WER: {act*100:4.1f}% | S={details['S']} D={details['D']} I={details['I']}")
        print("\nSelf-test completed successfully.")
        return

    text = ""
    if args.text:
        text = args.text
    elif args.file and args.file.exists():
        text = args.file.read_text(encoding="utf-8").strip()
    else:
        text = "Proje yönetimi dersi kapsamında kazanılmış değer ve maliyet sapması analizi yapılacaktır."

    if args.ladder:
        ladder = generate_synthetic_noise_ladder(text, lang=args.lang, seed=args.seed)
        print(f"{'Target WER (%)':<15} {'Actual WER (%)':<15} {'Preview (first 60 chars)'}")
        print("-" * 75)
        for row in ladder:
            print(f"{row['target_wer']*100:<15.1f} {row['actual_wer']*100:<15.1f} {row['hypothesis'][:60]}...")
    else:
        hyp, act, details = calibrate_noise(text, args.wer, lang=args.lang, seed=args.seed)
        print(f"Target WER: {args.wer*100:.1f}%")
        print(f"Actual WER: {act*100:.1f}%")
        print(f"S={details['S']} D={details['D']} I={details['I']} N={details['N']}")
        print(f"\nCorrupted Text:\n{hyp}")


if __name__ == "__main__":
    main()
