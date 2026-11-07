import io
import re

t = io.open("Boğaziçi_CMPE492_Final_Report.tex", encoding="utf-8").read()
b = io.open("references.bib", encoding="utf-8").read()
cited = set()
for m in re.finditer(r"\\cite\w*\{([^}]*)\}", t):
    for k in m.group(1).split(","):
        cited.add(k.strip())
bibkeys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", b))
print("cited", len(cited), "bib entries", len(bibkeys))
print("MISSING IN BIB:", sorted(cited - bibkeys))
labels = set(re.findall(r"\\label\{([^}]*)\}", t))
refs = set(re.findall(r"\\(?:ref|eqref|pageref|autoref)\{([^}]*)\}", t))
print("UNDEFINED LABELS:", sorted(refs - labels))
print("UNUSED LABELS:", len(labels - refs))
print("sections:")
for m in re.finditer(r"\\(?:sub)*section\{([^}]*)\}", t):
    line = t[:m.start()].count("\n") + 1
    print(line, m.group(1)[:80])
