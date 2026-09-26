"""Inter-annotator agreement and temporal span evaluation protocol.
Supports:
- Cohen's kappa (nominal 2-rater)
- PABAK (Prevalence-Adjusted and Bias-Adjusted Kappa) to handle skewed class prevalence
- Fleiss' kappa (multi-annotator extension)
- Temporal Span Overlap (Intersection over Union / IoU) for timestamped terms
- Term-level Precision, Recall, and F1 against reference term sets
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


def cohen_kappa(y1: Sequence[int], y2: Sequence[int]) -> dict[str, Any]:
    """Calculates Cohen's kappa, observed agreement po, and chance agreement pe."""
    if len(y1) != len(y2):
        raise ValueError("Annotation sequences must have equal length.")
    n = len(y1)
    if n == 0:
        return {"n": 0, "po": 0.0, "pe": 0.0, "kappa": 0.0, "pabak": 0.0}

    agree = sum(1 for a, b in zip(y1, y2) if a == b)
    po = agree / n

    c1, c2 = Counter(y1), Counter(y2)
    labels = sorted(set(c1) | set(c2))
    pe = sum((c1[lbl] / n) * (c2[lbl] / n) for lbl in labels)

    if math.isclose(pe, 1.0):
        kappa = 1.0 if math.isclose(po, 1.0) else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    # PABAK (Byrt et al., 1993): 2 * po - 1
    pabak = 2.0 * po - 1.0

    return {
        "n": n,
        "po": round(po, 4),
        "pe": round(pe, 4),
        "kappa": round(kappa, 4),
        "pabak": round(pabak, 4),
        "distribution": {
            "annotator1": dict(c1),
            "annotator2": dict(c2),
        },
    }


def fleiss_kappa(ratings_matrix: list[list[int]], num_categories: int = 2) -> dict[str, Any]:
    """Computes Fleiss' kappa for multiple annotators rating items into nominal categories.
    ratings_matrix: list of items, where each item is a list of counts of ratings for each category.
    Example: [[2, 0], [1, 1], [0, 2]] for 3 items, 2 categories, 2 annotators.
    """
    N = len(ratings_matrix)  # number of items
    if N == 0:
        return {"n_items": 0, "kappa": 0.0}

    n = sum(ratings_matrix[0])  # number of raters per item
    k = num_categories

    # Calculate p_j: proportion of all assignments to the j-th category
    total_ratings = N * n
    p_j = [0.0] * k
    for j in range(k):
        p_j[j] = sum(row[j] for row in ratings_matrix) / total_ratings

    # Calculate P_i: degree of agreement for the i-th subject
    P_i = []
    for row in ratings_matrix:
        val = sum(count * (count - 1) for count in row) / (n * (n - 1)) if n > 1 else 1.0
        P_i.append(val)

    P_bar = sum(P_i) / N
    P_e_bar = sum(pj ** 2 for pj in p_j)

    if math.isclose(P_e_bar, 1.0):
        kappa = 1.0 if math.isclose(P_bar, 1.0) else 0.0
    else:
        kappa = (P_bar - P_e_bar) / (1.0 - P_e_bar)

    return {
        "n_items": N,
        "n_raters": n,
        "po": round(P_bar, 4),
        "pe": round(P_e_bar, 4),
        "kappa": round(kappa, 4),
    }


def temporal_span_iou(span_a: tuple[float, float], span_b: tuple[float, float]) -> float:
    """Computes Intersection over Union (IoU) between two time spans (start_sec, end_sec)."""
    start_a, end_a = span_a
    start_b, end_b = span_b

    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    intersection = max(0.0, inter_end - inter_start)

    union_start = min(start_a, start_b)
    union_end = max(end_a, end_b)
    union = max(1e-6, union_end - union_start)

    return round(intersection / union, 4)


def evaluate_term_f1_with_timestamps(
    hypotheses: list[dict[str, Any]],
    references: list[dict[str, Any]],
    min_iou: float = 0.3,
) -> dict[str, Any]:
    """Computes term-level Precision, Recall, and F1 with temporal matching threshold.
    Each item must have 'canonical_term' and 'span': (start_s, end_s).
    """
    matched_ref_indices = set()
    true_positives = 0

    for hyp in hypotheses:
        hyp_term = hyp["canonical_term"].strip().lower()
        hyp_span = hyp["span"]

        best_match_idx = None
        best_iou = 0.0

        for r_idx, ref in enumerate(references):
            if r_idx in matched_ref_indices:
                continue
            ref_term = ref["canonical_term"].strip().lower()
            if hyp_term == ref_term:
                iou = temporal_span_iou(hyp_span, ref["span"])
                if iou >= min_iou and iou > best_iou:
                    best_iou = iou
                    best_match_idx = r_idx

        if best_match_idx is not None:
            matched_ref_indices.add(best_match_idx)
            true_positives += 1

    fp = len(hypotheses) - true_positives
    fn = len(references) - len(matched_ref_indices)

    precision = true_positives / (true_positives + fp) if (true_positives + fp) > 0 else 0.0
    recall = true_positives / (true_positives + fn) if (true_positives + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": true_positives,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "min_iou_threshold": min_iou,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate inter-annotator agreement and temporal F1.")
    parser.add_argument("--test", action="store_true", help="Run self-tests")
    args = parser.parse_args()

    if args.test:
        # Test Cohen's kappa with high prevalence scenario
        r1 = [1] * 95 + [0] * 5
        r2 = [1] * 93 + [0] * 7
        res = cohen_kappa(r1, r2)
        print("Test Cohen's kappa:", res)
        assert res["n"] == 100
        assert res["pabak"] > 0.90

        # Test Fleiss' kappa
        matrix = [[2, 0]] * 80 + [[1, 1]] * 10 + [[0, 2]] * 10
        fk_res = fleiss_kappa(matrix, num_categories=2)
        print("Test Fleiss' kappa:", fk_res)
        assert fk_res["kappa"] > 0.60

        # Test Temporal IoU
        span1 = (10.0, 20.0)
        span2 = (15.0, 25.0)
        iou = temporal_span_iou(span1, span2)
        print("Test IoU (10-20 vs 15-25):", iou)
        assert math.isclose(iou, 5.0 / 15.0, abs_tol=1e-3)

        # Test Term F1 with Timestamps
        hyps = [{"canonical_term": "EVM", "span": (70.0, 85.0)}]
        refs = [{"canonical_term": "EVM", "span": (72.0, 88.0)}]
        f1_res = evaluate_term_f1_with_timestamps(hyps, refs, min_iou=0.3)
        print("Test Term F1:", f1_res)
        assert f1_res["f1"] == 1.0

        print("[ALL TESTS PASSED]")


if __name__ == "__main__":
    main()
