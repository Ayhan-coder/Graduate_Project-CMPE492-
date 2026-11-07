import json

# Read jsonl files and map WERs
files_to_read = {
    "Whisper Large-v3": "experiments/outputs/asr-20260901-large-v3.jsonl",
    "Whisper Large-v3 Turbo": "experiments/outputs/asr-20260903-turbo.jsonl",
    "Qwen3-ASR 0.6B": "experiments/outputs/asr-qwen3-0.6b.rescored.jsonl",
    "Qwen3-ASR 1.7B": "experiments/outputs/asr-qwen3-1.7b.rescored.jsonl",
    "Nemotron 3.5": "experiments/outputs/asr-nemotron.rescored.jsonl",
    "Whisper Turkish Turbo": "experiments/outputs/asr-selimc-turbo.rescored.jsonl",
}

clip_wers = {}
for mname, fpath in files_to_read.items():
    clip_wers[mname] = {}
    try:
        with open(fpath, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                cid = d.get("id") or d.get("clip_id")
                # normalized wer
                w = d.get("wer_norm", d.get("wer", 0))
                if cid:
                    clip_wers[mname][cid] = w
    except Exception as e:
        print(f"Error {fpath}: {e}")

for mname, cdict in clip_wers.items():
    print(f"{mname:<25}: {len(cdict)} clips -> {dict(sorted(cdict.items()))}")

