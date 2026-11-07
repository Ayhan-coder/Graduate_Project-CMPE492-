import io
import re
import collections

t = io.open("Boğaziçi_CMPE492_Final_Report.tex", encoding="utf-8").read()
spans = re.findall(r"\\texttt\{([^{}]*)\}", t)
long = [s for s in spans if len(s) > 40]
print("texttt total", len(spans), "long>40", len(long))
esc = collections.Counter()
for s in long:
    for m in re.finditer(r"\\(.)", s):
        esc[m.group(1)] += 1
print("escapes in long texttt:", dict(esc))
for s in long:
    print("  ", s[:120])
