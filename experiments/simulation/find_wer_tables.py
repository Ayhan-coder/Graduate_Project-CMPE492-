import glob
import json

for f in glob.glob('experiments/outputs/*.json'):
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and ('wer' in data[0] or 'wer_norm' in data[0]):
            print(f"Found list with WER in {f}: {len(data)} items")
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and ('wer' in v[0] or 'wer_norm' in v[0]):
                    print(f"Found key '{k}' with WER in {f}: {len(v)} items")
    except Exception as e:
        pass

