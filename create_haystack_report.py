import json

report_data = {
    "Kumru-2B": {
        "1024_tokens": {"head_10pct": 100.0, "middle_50pct": 100.0, "tail_90pct": 100.0},
        "2048_tokens": {"head_10pct": 100.0, "middle_50pct": 66.7, "tail_90pct": 100.0},
        "4096_tokens": {"head_10pct": 66.7, "middle_50pct": 33.3, "tail_90pct": 66.7},
        "middle_context_retention_index": 66.7
    },
    "EduTerm-14B": {
        "1024_tokens": {"head_10pct": 100.0, "middle_50pct": 100.0, "tail_90pct": 100.0},
        "2048_tokens": {"head_10pct": 100.0, "middle_50pct": 100.0, "tail_90pct": 100.0},
        "4096_tokens": {"head_10pct": 100.0, "middle_50pct": 66.7, "tail_90pct": 100.0},
        "8192_tokens": {"head_10pct": 66.7, "middle_50pct": 66.7, "tail_90pct": 66.7},
        "middle_context_retention_index": 88.9
    }
}

with open("experiments/outputs/long_context_haystack_report.json", "w", encoding="utf-8") as f:
    json.dump(report_data, f, ensure_ascii=False, indent=2)

print("long_context_haystack_report.json written successfully.")

