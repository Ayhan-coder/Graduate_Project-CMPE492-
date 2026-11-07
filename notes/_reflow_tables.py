import io
import re

PATH = "Boğaziçi_CMPE492_Final_Report.tex"
X = r">{\raggedright\arraybackslash}X"

with io.open(PATH, encoding="utf-8", newline="") as fh:
    lines = fh.read().split("\r\n")

out = []
i = 0
n_fixed = 0
while i < len(lines):
    line = lines[i]
    if line.strip() == r"\resizebox{\textwidth}{!}{%":
        begin = lines[i + 1]
        m = re.match(r"\\begin\{tabularx\}\{\\textwidth\}\{(.*)\}\s*$", begin.strip())
        assert m, begin
        spec = m.group(1)
        assert spec.startswith("l"), spec
        newspec = X + spec[1:]
        out.append(r"\footnotesize\setlength{\tabcolsep}{3pt}")
        out.append("\\begin{tabularx}{\\textwidth}{" + newspec + "}")
        i += 2
        # copy body until the closing \end{tabularx}% + }
        while True:
            cur = lines[i]
            if cur.strip() in (r"\end{tabularx}%", r"\end{tabularx}"):
                out.append(r"\end{tabularx}")
                assert lines[i + 1].strip() == "}", lines[i + 1]
                i += 2
                break
            out.append(cur)
            i += 1
        n_fixed += 1
        continue
    out.append(line)
    i += 1

text = "\r\n".join(out)
with io.open(PATH, "w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("tables reflowed:", n_fixed, "bare LF:", text.count("\n") - text.count("\r\n"))
