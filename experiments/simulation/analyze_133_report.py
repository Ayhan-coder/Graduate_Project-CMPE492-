import json

with open('experiments/outputs/full_corpus_133_report.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

lectures = d.get('lectures', [])
print(f"Total lectures in report: {len(lectures)}")

with_terms = [l for l in lectures if l.get('terms_extracted_count', 0) > 0]
without_terms = [l for l in lectures if l.get('terms_extracted_count', 0) == 0]

print(f"Lectures with extracted terms: {len(with_terms)}")
print(f"Lectures with 0 terms: {len(without_terms)}")

print("\nSample lectures WITH terms:")
for l in with_terms[:5]:
    term_names = [t.get('term', '') for t in l.get('terms', [])]
    print(f"  {l['id']:<6} | {l['language']} | Terms: {len(term_names):<2} | {term_names}")

print("\nInspecting first 3 lectures WITHOUT terms raw snippet:")
for l in without_terms[:3]:
    snip = repr(l.get('raw_output_snippet', ''))[:120]
    print(f"  {l['id']:<6} | snippet: {snip}")

