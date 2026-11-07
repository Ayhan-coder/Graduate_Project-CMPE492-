import io
import re

PATH = "Boğaziçi_CMPE492_Final_Report.tex"
with io.open(PATH, encoding="utf-8", newline="") as fh:
    text = fh.read()

text = text.replace("\\usepackage{tabularx}\r\n", "\\usepackage{tabularx}\r\n\\usepackage{ragged2e}\r\n", 1)
text = text.replace(r"\raggedright\arraybackslash", r"\RaggedRight\arraybackslash")

# moving arguments: \path is not safe inside captions / section titles
skip = []
for m in re.finditer(r"\\(?:caption|(?:sub)*section)\{", text):
    depth = 1
    j = m.end()
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    skip.append((m.start(), j))

count = 0
skipped = 0


def in_skip(pos):
    return any(a <= pos < b for a, b in skip)


def repl(m):
    global count, skipped
    body = m.group(1)
    is_identifier = ("/" in body) or ("\\_" in body and len(body) > 20)
    if not is_identifier:
        return m.group(0)
    if in_skip(m.start()):
        skipped += 1
        return m.group(0)
    count += 1
    return "\\path{" + body.replace("\\_", "_") + "}"


text = re.sub(r"\\texttt\{([^{}]*)\}", repl, text)

with io.open(PATH, "w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("identifiers converted:", count, "skipped in moving args:", skipped)
print("RaggedRight cells:", text.count(r"\RaggedRight\arraybackslash"))
print("bare LF:", text.count("\n") - text.count("\r\n"))
