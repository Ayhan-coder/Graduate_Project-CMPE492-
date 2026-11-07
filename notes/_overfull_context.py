import io
import re

log = io.open("Boğaziçi_CMPE492_Final_Report.log", encoding="utf-8", errors="replace").read().splitlines()
seen = set()
for i, l in enumerate(log):
    if not l.startswith("Overfull"):
        continue
    w = float(re.search(r"\(([0-9.]+)pt", l).group(1))
    if w < 30:
        continue
    key = l.split(" at lines ")[-1]
    if key in seen:
        continue
    seen.add(key)
    print("-----", l)
    print("\n".join(log[i + 1:i + 6])[:450])
