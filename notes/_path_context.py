import io
import re

t = io.open("Boğaziçi_CMPE492_Final_Report.tex", encoding="utf-8").read()
for m in re.finditer(r"\\texttt\{([^{}]*)\}", t):
    b = m.group(1)
    if len(b) > 40 and "/" in b:
        pre = t[max(0, m.start() - 30):m.start()].replace("\r\n", " ")
        print(repr(pre), "->", b[:60])
