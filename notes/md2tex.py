"""md2tex.py — compile notes/Bitirme_Projesi_Raporu.md into a LaTeX thesis report.

Single source of truth: the Markdown file. Re-run after every Markdown edit:

    python md2tex.py            # write .tex
    latexmk -pdf Bitirme_Projesi_Raporu.tex

Box-drawing characters in the ASCII architecture diagram are folded to ASCII so
the document builds with plain pdflatex + T1 encoding.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MD_SRC = ROOT / "Bitirme_Projesi_Raporu.md"
TEX_OUT = ROOT / "Bitirme_Projesi_Raporu.tex"

WARNINGS: list[str] = []

# ---------------------------------------------------------------- placeholders

MATH_TOKEN = "\x00M{}\x00"
CODE_TOKEN = "\x00C{}\x00"

BOX_DRAWING = {
    "│": "|", "┌": "+", "┐": "+", "└": "+", "┘": "+", "├": "+", "┤": "+",
    "┬": "+", "┴": "+", "┼": "+", "─": "-", "═": "=", "▼": "v", "▲": "^",
    "►": ">", "◄": "<", "→": "->", "←": "<-", "↓": "v", "↑": "^", "␣": " ",
    "−": "-", "≈": "~", "×": "x", "Δ": "D", "≥": ">=", "≤": "<=", "±": "+/-",
}

TEXT_FOLD = {
    "Δ": "$\\Delta$", "−": "$-$", "≈": "$\\approx$", "→": "$\\to$",
    "≥": "$\\ge$", "≤": "$\\le$", "×": "$\\times$", "±": "$\\pm$",
    "∞": "$\\infty$", "·": "$\\cdot$",
}

LATEX_SPECIALS = {
    "\\": "\\textbackslash{}",
    "&": "\\&",
    "%": "\\%",
    "$": "\\$",
    "#": "\\#",
    "_": "\\_",
    "{": "\\{",
    "}": "\\}",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
}


def escape_text(s: str, fold_math: bool = True) -> str:
    # TEXT_FOLD runs after math spans are stashed, so the inserted $..$ never re-enters
    # the math pass; code spans pass fold_math=False and get ASCII instead.
    table = dict(LATEX_SPECIALS)
    if fold_math:
        table.update(TEXT_FOLD)
    out = []
    for ch in s:
        out.append(table.get(ch, ch))
    return "".join(out)


def math_to_plain(s: str) -> str:
    """Rough PDF-bookmark text for a math span."""
    t = s.strip("$")
    t = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = t.replace("_", "").replace("^", "").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", t).strip()


def fold_unicode(s: str) -> str:
    for bad, good in BOX_DRAWING.items():
        s = s.replace(bad, good)
    return s


# Emoji / pictograph blocks have no glyph in T1 Type 1 fonts, so pdflatex aborts
# on them. They are dropped rather than substituted, and every drop is reported.
PICTOGRAPH = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # tiles, symbols, pictographs, extended-A
    "\U0001F1E6-\U0001F1FF"  # regional indicators
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F900-\U0001F9FF"
    " -‏⁠﻿"  # zero-width / variation selectors
    "]+"
)


def strip_pictographs(s: str, where: str = "") -> str:
    def drop(m: re.Match) -> str:
        for ch in set(m.group(0)):
            WARNINGS.append(
                "dropped U+%04X (%s) %s— no pdflatex glyph; edit the Markdown if it must stay"
                % (ord(ch), ch, ("in %s" % where if where else ""))
            )
        return ""

    return PICTOGRAPH.sub(drop, s)


def smart_quotes(s: str) -> str:
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    return s


# ---------------------------------------------------------------- inline pass

def convert_inline(text: str) -> str:
    maths: list[str] = []
    codes: list[str] = []
    text = strip_pictographs(text)

    def stash_math(m: re.Match) -> str:
        maths.append(m.group(0))
        return MATH_TOKEN.format(len(maths) - 1)

    # display math first so its $$ is not eaten by the inline pattern
    text = re.sub(r"\$\$[^$]*\$\$", stash_math, text)
    text = re.sub(r"\$[^$\n]+\$", stash_math, text)

    def stash_code(m: re.Match) -> str:
        codes.append(m.group(1))
        return CODE_TOKEN.format(len(codes) - 1)

    text = re.sub(r"`([^`\n]+)`", stash_code, text)

    text = escape_text(text)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\\textbf{\\emph{\1}}", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\\emph{\1}", text)
    text = smart_quotes(text)

    for i, code in enumerate(codes):
        body = escape_text(fold_unicode(code))
        # long identifiers and paths cannot hyphenate; permit breaks at _ and /
        body = body.replace("\\_", "\\_\\allowbreak{}").replace("/", "/\\allowbreak{}")
        text = text.replace(CODE_TOKEN.format(i), "\\texttt{" + body + "}")
    for i, math in enumerate(maths):
        text = text.replace(MATH_TOKEN.format(i), math)
    return text


# ---------------------------------------------------------------- block parse

def split_row(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [c.strip() for c in inner.split("|")]


def build_table(rows: list[str]) -> list[str]:
    header = split_row(rows[0])
    aligns = split_row(rows[1])
    body = [split_row(r) for r in rows[2:]]
    ncols = len(header)

    def align_of(i: int) -> str:
        if i >= len(aligns):
            return "l"
        a = aligns[i]
        left, right = a.startswith(":"), a.endswith(":")
        return "c" if left and right else ("r" if right else "l")

    widths = [max([len(header[i])] + [len(r[i]) if i < len(r) else 0 for r in body]) for i in range(ncols)]
    # a column wider than its share of the page must be able to wrap, otherwise
    # long headers ("CER (Norm)") push the whole tabular past \textwidth
    wide = [i for i, w in enumerate(widths) if w > 9]

    spec, kinds = [], []
    for i in range(ncols):
        al = align_of(i)
        if i in wide:
            spec.append("C" if al == "c" else "L")
            kinds.append("X")
        else:
            spec.append({"c": "c", "r": "r", "l": "l"}[al])
            kinds.append(al)

    def cell(txt: str, bold: bool) -> str:
        t = convert_inline(txt)
        return "\\textbf{%s}" % t if bold and not t.startswith("\\textbf") else t

    out = []
    env = "tabularx" if wide else "tabular"
    colspec = "".join(spec)
    head = ("\\begin{tabularx}{\\textwidth}{%s}" % colspec) if wide else ("\\begin{tabular}{%s}" % colspec)
    out.append("\\begin{table}[htbp]")
    out.append("\\centering")
    out.append("\\small")
    if ncols >= 5:
        out.append("\\setlength{\\tabcolsep}{%dpt}" % (5 if ncols < 7 else 4))
    out.append(head)
    out.append("\\toprule")
    out.append(" & ".join(cell(c, True) for c in header) + " \\\\")
    out.append("\\midrule")
    for r in body:
        r = r + [""] * (ncols - len(r))
        out.append(" & ".join(cell(r[i], False) for i in range(ncols)) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{%s}" % env)
    out.append("\\end{table}")
    if len(header) != ncols:
        WARNINGS.append("table: inconsistent column count")
    return out


def build_verbatim(block: list[str]) -> list[str]:
    lines = [fold_unicode(strip_pictographs(l, "code block")) for l in block]
    return (
        ["\\begin{center}", "\\begin{small}", "\\begin{verbatim}"]
        + lines
        + ["\\end{verbatim}", "\\end{small}", "\\end{center}"]
    )


def heading_level(n: int) -> str:
    return {1: "part", 2: "section", 3: "subsection", 4: "subsubsection"}.get(n, "paragraph")


UNNUMBERED = {"özet", "abstract", "kaynakça", "references", "içindekiler"}


def build_heading(level: int, raw: str) -> list[str]:
    cmd = heading_level(level)
    text = raw.strip()
    manual = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$", text)
    auto = True
    if manual:
        text = manual.group(2)
    elif re.match(r"^[A-Za-zÇĞİÖŞÜçğıöşü0-9]", text):
        auto = False

    if text.lower().strip() in UNNUMBERED:
        auto = False

    body = convert_inline(text)
    if "$" in body:
        plain = re.sub(r"\$([^$]*)\$", lambda m: math_to_plain(m.group(0)) or " ", body)
        body = "\\texorpdfstring{%s}{%s}" % (body, plain)
    if not auto:
        cmd += "*"
    return ["", "\\%s{%s}" % (cmd, body)]


# ---------------------------------------------------------------- doc emit

PREAMBLE = r"""%% AUTO-GENERATED by md2tex.py from Bitirme_Projesi_Raporu.md — do not edit by hand.
%% Edit the Markdown, then run:  python md2tex.py && latexmk -pdf Bitirme_Projesi_Raporu.tex
\documentclass[11pt,a4paper]{report}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{textcomp}
\usepackage[margin=2.5cm,top=3cm,bottom=3cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{enumitem}
\usepackage[table]{xcolor}
\usepackage{titlesec}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage{setspace}

\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{C}{>{\centering\arraybackslash}X}

\renewcommand{\contentsname}{İçindekiler}
\renewcommand{\bibname}{Kaynakça}

%% The report class numbers \section as <chapter>.<section>; this thesis has no
%% chapters, so the default would render "0.1 Giriş" and break the Markdown's own
%% "Bölüm 6.3" cross-references. Drop the leading chapter component.
\renewcommand{\thesection}{\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
\renewcommand{\thesubsubsection}{\thesubsection.\arabic{subsubsection}}

\titleformat{\section}{\Large\bfseries\color{blue!70!black}}{\thesection}{0.8em}{}
\titleformat{\subsection}{\large\bfseries}{\thesubsection}{0.8em}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{\thesubsubsection}{0.8em}{}
\setlist{itemsep=2pt,topsep=4pt}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Boğaziçi Üniversitesi \textperiodcentered\ Bilgisayar Mühendisliği Bitirme Tezi}
\fancyhead[R]{\small Ali Ayhan Günder}
\fancyfoot[C]{\thepage}
\setstretch{1.08}
\emergencystretch=3em

\begin{document}
"""


def front_matter(title: str, meta: list[str]) -> list[str]:
    out = [
        "\\begin{titlepage}",
        "\\centering",
        "{\\huge\\bfseries BOĞAZİÇİ ÜNİVERSİTESİ}\\\\[4pt]",
        "{\\LARGE\\bfseries Bilgisayar Mühendisliği Bölümü}\\\\[28pt]",
        "{\\Large\\bfseries BİLGİSAYAR MÜHENDİSLİĞİ LİSANS BİTİRME TEZİ}\\\\[24pt]",
        "{\\huge\\bfseries " + convert_inline(title) + "}\\\\[24pt]",
    ]
    for line in meta:
        m = re.match(r"^\*\*(.+?):\*\*\s*(.*)$", line.strip())
        if m:
            out.append("{\\large \\textbf{%s:} %s}\\\\[6pt]" % (m.group(1), convert_inline(m.group(2))))
        elif line.strip():
            out.append("{\\large %s}\\\\[6pt]" % convert_inline(line))
    out += ["\\vfill", "\\end{titlepage}", "", "\\tableofcontents", "\\newpage", ""]
    return out


def bibliography(items: list[str]) -> list[str]:
    out = ["\\begin{thebibliography}{99}", ""]
    for idx, raw in enumerate(items, 1):
        body = re.sub(r"^\d+\.\s+", "", raw.strip())
        key = "ref%02d" % idx
        out.append("\\bibitem{%s} %s" % (key, convert_inline(body)))
    out += ["", "\\end{thebibliography}"]
    return out


def convert(md: str) -> list[str]:
    lines = md.split("\n")
    tex: list[str] = []
    title, meta = "Bitirme Projesi Raporu", []
    i, n = 0, len(lines)
    in_fence, fence_buf = False, []
    list_stack: list[list] = []
    blank_seen = False
    bib_items, in_bib = [], False
    para: list[str] = []

    def flush_para():
        nonlocal para
        if para:
            tex.append(convert_inline(" ".join(para).strip()))
            tex.append("")
            para = []

    def close_list(depth: int = 0):
        """Close open list levels down to `depth` (0 = close all)."""
        while len(list_stack) > depth:
            tex.append("\\end{%s}" % list_stack[-1][1])
            list_stack.pop()

    while i < n:
        raw = lines[i]
        s = raw.strip()

        if in_fence:
            if s.startswith("```"):
                tex += build_verbatim(fence_buf)
                tex.append("")
                in_fence = False
                fence_buf = []
            else:
                fence_buf.append(raw)
            i += 1
            continue
        if s.startswith("```"):
            flush_para()
            close_list()
            in_fence = True
            fence_buf = []
            i += 1
            continue

        # tables
        if s.startswith("|") and i + 1 < n and re.match(r"^\|[\s:|-]+\|?\s*$", lines[i + 1].strip()):
            flush_para()
            close_list()
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            tex += build_table(rows)
            tex.append("")
            continue

        # headings
        h = re.match(r"^(#{1,6})\s+(.*)$", s)
        if h:
            flush_para()
            close_list()
            level = len(h.group(1))
            text = h.group(2).strip()
            if level == 2 and text.lower() in ("kaynakça", "references"):
                in_bib = True
                i += 1
                continue
            if in_bib:
                tex += bibliography(bib_items)
                bib_items, in_bib = [], False
            if level == 1:
                title = text
                i += 1
                continue
            tex += build_heading(level, text)
            i += 1
            continue

        if re.match(r"^---+\s*$", s):
            flush_para()
            close_list()
            i += 1
            continue

        if in_bib:
            if re.match(r"^\d+\.\s+", s):
                bib_items.append(s)
            elif bib_items and s:
                bib_items[-1] += " " + s
            i += 1
            continue

        # front matter (before the first section heading)
        if not any(t.startswith("\\section") for t in tex):
            if s:
                meta.append(s)
            i += 1
            continue

        # blockquotes
        if s.startswith(">"):
            flush_para()
            close_list()
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            tex.append("\\begin{quote}")
            tex.append(convert_inline(" ".join(b for b in buf if b)))
            tex.append("\\end{quote}")
            tex.append("")
            continue

        # list items, indentation-aware so nested bullets do not restart numbering
        lm = re.match(r"^(\s*)([-*]|\d+[.)])\s+(.*)$", raw)
        if lm:
            flush_para()
            blank_seen = False
            indent = len(lm.group(1).expandtabs(4))
            kind = "itemize" if lm.group(2) in ("-", "*") else "enumerate"
            while list_stack and list_stack[-1][0] > indent:
                close_list(len(list_stack) - 1)
            if list_stack and list_stack[-1][0] == indent and list_stack[-1][1] != kind:
                close_list(len(list_stack) - 1)
            if not list_stack or list_stack[-1][0] < indent:
                tex.append("\\begin{%s}" % kind)
                list_stack.append([indent, kind])
            tex.append("  " * len(list_stack) + "\\item " + convert_inline(lm.group(3).strip()))
            i += 1
            continue

        if not s:
            flush_para()
            blank_seen = True
            i += 1
            continue

        if list_stack and blank_seen:
            close_list()
        if list_stack and not blank_seen:
            # lazy continuation: append to the open item
            for j in range(len(tex) - 1, -1, -1):
                if tex[j].lstrip().startswith("\\item"):
                    tex[j] += " " + convert_inline(s)
                    break
            i += 1
            continue

        para.append(s)
        i += 1

    flush_para()
    close_list()
    if in_bib and bib_items:
        tex += bibliography(bib_items)
    tex.append("\n\\end{document}\n")
    return front_matter(title, meta), tex


def main() -> int:
    # Turkish glyphs and emoji in the Markdown source crash a cp1254 console.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    md = MD_SRC.read_text(encoding="utf-8")
    head, body = convert(md)
    TEX_OUT.write_text("\n".join([PREAMBLE] + head + body), encoding="utf-8")
    print("wrote", TEX_OUT, TEX_OUT.stat().st_size, "bytes")
    for w in WARNINGS:
        print("warning:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
