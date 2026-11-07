import json
import glob
from scipy import stats

with open('experiments/outputs/asr_by_lang.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

models = d['models']

rows = []
for mname, mdata in models.items():
    tr = mdata.get('langs', {}).get('tr', {})
    en = mdata.get('langs', {}).get('en', {})
    tr_wer = tr.get('wer_norm', 0)
    tr_ci = tr.get('wer_ci', [0, 0])
    en_wer = en.get('wer_norm', 0)
    en_ci = en.get('wer_ci', [0, 0])
    rtf = mdata.get('pooled', {}).get('rtf', 0)
    rows.append({
        'name': mname,
        'tr_wer': tr_wer,
        'tr_ci': tr_ci,
        'en_wer': en_wer,
        'en_ci': en_ci,
        'rtf': rtf
    })

print(f"{'Model':<40} | {'TR WER [95% CI]':<25} | {'EN WER [95% CI]':<25} | {'RTF'}")
print("-" * 105)
for r in rows:
    tr_str = f"{r['tr_wer']:.3f} [{r['tr_ci'][0]:.3f}, {r['tr_ci'][1]:.3f}]"
    en_str = f"{r['en_wer']:.3f} [{r['en_ci'][0]:.3f}, {r['en_ci'][1]:.3f}]" if r['en_wer'] > 0 else "-"
    print(f"{r['name']:<40} | {tr_str:<25} | {en_str:<25} | {r['rtf']:.3f}")

