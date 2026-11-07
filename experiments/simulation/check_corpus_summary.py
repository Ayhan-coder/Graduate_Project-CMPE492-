import json

with open('experiments/data/data_inventory.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

records = d.get('records', [])
print(f"Total records in data_inventory.json: {len(records)}")

by_lang = {}
by_audio = {}
for r in records:
    l = r.get('lang')
    by_lang[l] = by_lang.get(l, 0) + 1
    aud = r.get('audio', {}).get('status')
    by_audio[aud] = by_audio.get(aud, 0) + 1

print("By language:", by_lang)
print("By audio status:", by_audio)
print("-" * 80)
for r in records:
    print(f"{r['id']:<6} | {r['lang']} | {r['duration']:<8} | {r.get('audio',{}).get('status'):<20} | {r['title'][:40]}")

