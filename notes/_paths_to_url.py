import io
import re

PATH = "Boğaziçi_CMPE492_Final_Report.tex"
with io.open(PATH, encoding="utf-8", newline="") as fh:
    text = fh.read()

count = 0


def repl(m):
    global count
    body = m.group(1)
    if len(body) > 40 and "/" in body:
        count += 1
        return "\\path{" + body.replace("\\_", "_") + "}"
    return m.group(0)


text = re.sub(r"\\texttt\{([^{}]*)\}", repl, text)

anchor = "\\usepackage{url}\n"
assert anchor in text.replace("\r\n", "\n")
text = text.replace(
    "\\usepackage{url}\r\n",
    "\\usepackage{url}\r\n"
    "\\expandafter\\def\\expandafter\\UrlBreaks\\expandafter{\\UrlBreaks\\do\\/\\do\\_}\r\n",
    1,
)

with io.open(PATH, "w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("paths converted:", count, "bare LF:", text.count("\n") - text.count("\r\n"))
