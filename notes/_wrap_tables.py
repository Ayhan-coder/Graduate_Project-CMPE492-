import io

PATH = "Boğaziçi_CMPE492_Final_Report.tex"
PAIRS = [(748, 758), (801, 821), (860, 868), (896, 905), (933, 948),
         (980, 993), (1022, 1032), (1060, 1068), (1097, 1110), (1140, 1147)]

with io.open(PATH, encoding="utf-8", newline="") as fh:
    text = fh.read()
crlf = text.count("\r\n")
lines = text.split("\r\n")
assert len(lines) == crlf + 1 or True

for begin, end in sorted(PAIRS, reverse=True):
    b = lines[begin - 1]
    e = lines[end - 1]
    assert b.strip().startswith(r"\begin{tabularx}"), (begin, b)
    assert e.strip() == r"\end{tabularx}", (end, e)
    lines[end - 1] = e + "%"
    lines.insert(end, "}")
    lines.insert(begin - 1, r"\resizebox{\textwidth}{!}{%")

out = "\r\n".join(lines)
with io.open(PATH, "w", encoding="utf-8", newline="") as fh:
    fh.write(out)
print("wrapped", len(PAIRS), "tables; bare LF now", out.count("\n") - out.count("\r\n"))
