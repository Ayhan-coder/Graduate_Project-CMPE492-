import json
import subprocess
from pathlib import Path

# 1. Update LaTeX Report
tex_path = Path("notes/Boğaziçi_CMPE492_Final_Report.tex")
tex = tex_path.read_text(encoding="utf-8")

old_table = r"""\begin{tabularx}{\textwidth}{lccccc}
\toprule
\textbf{Model Checkpoint} & \textbf{VRAM (GB)} & \textbf{Training Time} & \textbf{JSON Schema} & \textbf{Macro F1 (\%)} & \textbf{Precision (\%)} \\
\midrule
Kumru-2B (Zero-Shot Base) & 9.43~GB & --- & 87.5\% & 0.0\% & 0.0\% \\
\textbf{Kumru-2B (EduTerm SFT)} & \textbf{9.43~GB} & \textbf{54.5 min} & \textbf{100.0\%} & \textbf{43.8\%} & \textbf{70.8\%} \\
Qwen2.5-7B (EduTerm-7B SFT) & 16.80~GB & 2.1 hr & 100.0\% & 46.5\% & 75.0\% \\
\textbf{EduTerm-14B (CEFR SFT)} & \textbf{24.95~GB} & \textbf{6.6 hr} & \textbf{100.0\%} & \textbf{47.2\%} & \textbf{87.5\%} \\
\bottomrule
\end{tabularx}"""

new_table = r"""\begin{tabularx}{\textwidth}{lcccccc}
\toprule
\textbf{Model Checkpoint} & \textbf{Infer VRAM} & \textbf{JSON} & \textbf{Precision} & \textbf{Recall} & \textbf{Macro F1} & \textbf{Bloom Acc} \\
\midrule
Kumru-2B (Zero-Shot Base) & 1.71~GB & 87.5\% & 0.0\% & 0.0\% & 0.0\% & 0.0\% \\
\textbf{Kumru-2B (EduTerm-2B SFT)} & \textbf{3.20~GB} & \textbf{100.0\%} & \textbf{70.8\%} & \textbf{33.3\%} & \textbf{43.8\%} & \textbf{37.5\%} \\
Qwen2.5-14B (Zero-Shot Base) & 9.58~GB & 37.5\% & 8.3\% & 12.5\% & 10.0\% & 0.0\% \\
\textbf{EduTerm-14B-TR (Full CEFR SFT)} & \textbf{11.34~GB} & \textbf{100.0\%} & \textbf{87.5\%} & \textbf{32.3\%} & \textbf{46.7\%} & \textbf{50.0\%} \\
\bottomrule
\end{tabularx}"""

if old_table in tex:
    tex = tex.replace(old_table, new_table)
    tex_path.write_text(tex, encoding="utf-8")
    print("Updated LaTeX Table tab:kumru_vs_qwen_comparison!")
else:
    print("Warning: old_table not found in tex!")

# Recompile PDF
cmd = ["pdflatex", "-interaction=nonstopmode", "Boğaziçi_CMPE492_Final_Report.tex"]
res = subprocess.run(cmd, cwd="notes", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print("pdflatex code:", res.returncode)

import shutil
shutil.copyfile("notes/Boğaziçi_CMPE492_Final_Report.pdf", "GUNCEL_TESLIM_PDFLERI/Boğaziçi_CMPE492_Final_Report.pdf")
print("PDF copied to GUNCEL_TESLIM_PDFLERI successfully!")

# 2. Update Markdown Report
md_path = Path("notes/Bitirme_Projesi_Raporu.md")
md = md_path.read_text(encoding="utf-8")

old_md_table = """| Model Mimarisi | Parametre / Dil | Peak VRAM | Eğitim Süresi | JSON Uyum | Macro Precision | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Kumru-2B (Zero-Shot) | 2.39B / Türkçe | 9.43 GB | — | %87.5 | %0.0 | %0.0 |
| **Kumru-2B (EduTerm SFT)** | **2.39B / Türkçe** | **9.43 GB** | **54.5 dk** | **%100.0** | **%70.8** | **%43.8** |
| Qwen2.5-7B (EduTerm SFT) | 7.61B / Çok Dilli | 16.80 GB | 2.1 saat | %100.0 | %75.0 | %46.5 |
| **EduTerm-14B (CEFR SFT)** | **14.7B / Çok Dilli** | **24.95 GB** | **6.6 saat** | **%100.0** | **%87.5** | **%47.2** |"""

new_md_table = """| Model Mimarisi | Çıkarım VRAM | JSON Uyum | Macro Precision | Macro Recall | Macro F1 | Bloom Doğruluğu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Kumru-2B (Zero-Shot Base) | 1.71 GB | %87.5 | %0.0 | %0.0 | %0.0 | %0.0 |
| **Kumru-2B (EduTerm-2B SFT)** | **3.20 GB** | **%100.0** | **%70.8** | **%33.3** | **%43.8** | **%37.5** |
| Qwen2.5-14B (Zero-Shot Base) | 9.58 GB | %37.5 | %8.3 | %12.5 | %10.0 | %0.0 |
| **EduTerm-14B-TR (Full CEFR SFT)** | **11.34 GB** | **%100.0** | **%87.5** | **%32.3** | **%46.7** | **%50.0** |"""

if old_md_table in md:
    md = md.replace(old_md_table, new_md_table)
    md_path.write_text(md, encoding="utf-8")
    print("Updated Bitirme_Projesi_Raporu.md table!")
else:
    print("Warning: old_md_table not found in md!")

