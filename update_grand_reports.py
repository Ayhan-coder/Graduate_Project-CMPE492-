import json
import subprocess
from pathlib import Path

# 1. Update LaTeX Report
tex_path = Path("notes/Boğaziçi_CMPE492_Final_Report.tex")
tex = tex_path.read_text(encoding="utf-8")

old_table = r"""\begin{tabularx}{\textwidth}{lcccccc}
\toprule
\textbf{Model Checkpoint} & \textbf{Infer VRAM} & \textbf{JSON} & \textbf{Precision} & \textbf{Recall} & \textbf{Macro F1} & \textbf{Bloom Acc} \\
\midrule
Kumru-2B (Zero-Shot Base) & 1.71~GB & 87.5\% & 0.0\% & 0.0\% & 0.0\% & 0.0\% \\
\textbf{Kumru-2B (EduTerm-2B SFT)} & \textbf{3.20~GB} & \textbf{100.0\%} & \textbf{70.8\%} & \textbf{33.3\%} & \textbf{43.8\%} & \textbf{37.5\%} \\
Qwen2.5-14B (Zero-Shot Base) & 9.58~GB & 37.5\% & 8.3\% & 12.5\% & 10.0\% & 0.0\% \\
\textbf{EduTerm-14B-TR (Full CEFR SFT)} & \textbf{11.34~GB} & \textbf{100.0\%} & \textbf{87.5\%} & \textbf{32.3\%} & \textbf{46.7\%} & \textbf{50.0\%} \\
\bottomrule
\end{tabularx}"""

new_table = r"""\begin{tabularx}{\textwidth}{lcccccc}
\toprule
\textbf{Model Checkpoint} & \textbf{Train VRAM} & \textbf{Train Time} & \textbf{JSON} & \textbf{Precision} & \textbf{Recall} & \textbf{Macro F1} \\
\midrule
Kumru-2B (Zero-Shot Base) & --- & --- & 87.5\% & 0.0\% & 0.0\% & 0.0\% \\
\textbf{Kumru-2B (EduTerm-2B SFT)} & \textbf{9.43~GB} & \textbf{54.5 min} & \textbf{100.0\%} & \textbf{70.8\%} & \textbf{33.3\%} & \textbf{43.8\%} \\
Trendyol-LLM-8B-T1 (Zero-Shot) & --- & --- & 0.0\% & 0.0\% & 0.0\% & 0.0\% \\
\textbf{Trendyol-LLM-8B (EduTerm SFT)} & \textbf{19.01~GB} & \textbf{3.14 hr} & \textbf{62.5\%} & \textbf{50.0\%} & \textbf{18.8\%} & \textbf{27.1\%} \\
Qwen2.5-14B (Zero-Shot Base) & --- & --- & 37.5\% & 8.3\% & 12.5\% & 10.0\% \\
\textbf{EduTerm-14B-TR (Full CEFR SFT)} & \textbf{24.95~GB} & \textbf{6.60 hr} & \textbf{100.0\%} & \textbf{87.5\%} & \textbf{32.3\%} & \textbf{46.7\%} \\
\bottomrule
\end{tabularx}"""

if old_table in tex:
    tex = tex.replace(old_table, new_table)
    
    # Add textual analysis of Trendyol-8B
    marker = r"\subsection{Key Pedagogical and Linguistic Findings}"
    trendyol_analysis = r"""\subsection{Analysis of Heavyweight Turkish Reasoning Model: Trendyol-LLM-8B-T1}
To investigate whether increasing the parameter budget of open-source Turkish models yields superior pedagogical extraction, we evaluated \texttt{Trendyol/Trendyol-LLM-8B-T1} (8.23-billion parameters, built on Qwen3 with native Turkish chain-of-thought) trained under the identical 3-epoch QLoRA protocol on NVIDIA A40 (Slurm Job ID \texttt{34091}, 11,286.4s / 3.14 hours).

While the un-finetuned base model achieved $0.0\%$ extraction F1 due to strong conversational chattiness, the SFT adaptation successfully lowered training loss from $7.7398 \to 0.8283$, achieving \textbf{50.0\% Precision} and \textbf{27.1\% Macro F1}. However, as shown in Table~\ref{tab:kumru_vs_qwen_comparison}, Kumru-2B outscored Trendyol-8B in both JSON compliance ($100.0\%$ vs. $62.5\%$) and extraction F1 ($43.8\%$ vs. $27.1\%$). Qualitative inspection reveals that Trendyol-8B's pre-training on free-form e-commerce reasoning tokens creates strong resistance to strict JSON schema constraints, occasionally emitting recursive internal definitions or formatting leaks under strict greedy decoding. Consequently, \textbf{Kumru-2B emerges as the superior edge model} (9.43~GB VRAM, 54.5 min training), while \textbf{EduTerm-14B-TR dominates high-capacity pedagogical steering} (87.5\% Precision, 100\% JSON adherence).

"""
    tex = tex.replace(marker, trendyol_analysis + marker)
    tex_path.write_text(tex, encoding="utf-8")
    print("Updated LaTeX report with Trendyol-8B results!")
else:
    print("Warning: old_table not found in tex!")

# Recompile PDF
cmd = ["pdflatex", "-interaction=nonstopmode", "Boğaziçi_CMPE492_Final_Report.tex"]
res = subprocess.run(cmd, cwd="notes", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print("pdflatex exit code:", res.returncode)

import shutil
shutil.copyfile("notes/Boğaziçi_CMPE492_Final_Report.pdf", "GUNCEL_TESLIM_PDFLERI/Boğaziçi_CMPE492_Final_Report.pdf")
print("Copied PDF to GUNCEL_TESLIM_PDFLERI successfully!")

# 2. Update Markdown Report
md_path = Path("notes/Bitirme_Projesi_Raporu.md")
md = md_path.read_text(encoding="utf-8")

old_md_table = """| Model Mimarisi | Çıkarım VRAM | JSON Uyum | Macro Precision | Macro Recall | Macro F1 | Bloom Doğruluğu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Kumru-2B (Zero-Shot Base) | 1.71 GB | %87.5 | %0.0 | %0.0 | %0.0 | %0.0 |
| **Kumru-2B (EduTerm-2B SFT)** | **3.20 GB** | **%100.0** | **%70.8** | **%33.3** | **%43.8** | **%37.5** |
| Qwen2.5-14B (Zero-Shot Base) | 9.58 GB | %37.5 | %8.3 | %12.5 | %10.0 | %0.0 |
| **EduTerm-14B-TR (Full CEFR SFT)** | **11.34 GB** | **%100.0** | **%87.5** | **%32.3** | **%46.7** | **%50.0** |"""

new_md_table = """| Model Mimarisi | Eğitim VRAM | Eğitim Süresi | JSON Uyum | Macro Precision | Macro Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Kumru-2B (Zero-Shot Base) | --- | --- | %87.5 | %0.0 | %0.0 | %0.0 |
| **Kumru-2B (EduTerm-2B SFT)** | **9.43 GB** | **54.5 dk** | **%100.0** | **%70.8** | **%33.3** | **%43.8** |
| Trendyol-LLM-8B-T1 (Zero-Shot) | --- | --- | %0.0 | %0.0 | %0.0 | %0.0 |
| **Trendyol-LLM-8B (EduTerm SFT)** | **19.01 GB** | **3.14 saat** | **%62.5** | **%50.0** | **%18.8** | **%27.1** |
| Qwen2.5-14B (Zero-Shot Base) | --- | --- | %37.5 | %8.3 | %12.5 | %10.0 |
| **EduTerm-14B-TR (Full CEFR SFT)** | **24.95 GB** | **6.60 saat** | **%100.0** | **%87.5** | **%32.3** | **%46.7** |"""

if old_md_table in md:
    md = md.replace(old_md_table, new_md_table)
    
    trendyol_md_analysis = """
### 8.16.3 Ağır Siklet Türkçe Akıl Yürütme Modeli: Trendyol-LLM-8B-T1 Deneyi
Açık kaynak Türkçe ekosisteminin en gelişmiş modellerinden biri olan `Trendyol/Trendyol-LLM-8B-T1` (8.23 milyar parametre, Qwen3 mimarisi) A40 GPU üzerinde 3 tam devir boyunca 4.666 örnekli CEFR setimizle QLoRA SFT eğitimine alınmıştır (Slurm Job ID `34091`, 3.14 saat).

Eğitim sürecinde kayıp değeri kararlı şekilde $7.7398$'den $0.8283$'e düşmüş, hassasiyet sıfırdan **%50.0'ye** ve Macro F1 **%27.1'e** yükselmiştir. Ancak yapılan niteliksel incelemelerde:
1. **JSON Şema Direnci**: Trendyol-8B, yoğun e-ticaret ve serbest düşünce zinciri (chain-of-thought) ile önceden eğitildiği için katı yapılandırılmış JSON şablonlarında zaman zaman içsel yinelemeler yapmış ve JSON geçerliliği %62.5'te kalmıştır.
2. **Mimari Verimlilik Karşılaştırması**: 
   - **Kumru-2B**, yalnızca 9.43 GB VRAM ve 54.5 dakikalık eğitimle **%100 JSON uyumu ve %43.8 F1** sunarak **Uç Donanım (Edge/Local)** için en dengeli ve pratik Türkçe model olduğunu kanıtlamıştır.
   - **EduTerm-14B-TR**, çok katmanlı parametre derinliğiyle **%87.5 Precision, %50 Bloom doğruluğu ve %100 JSON uyumu** sağlayarak kurumsal/sunucu seviyesinde en güçlü pedagojik model unvanını korumuştur.
"""
    pos = md.find("## 8.17 Otomatik Eşleştirici Denetimi")
    if pos != -1:
        md = md[:pos] + trendyol_md_analysis + "\n\n" + md[pos:]
        md_path.write_text(md, encoding="utf-8")
        print("Updated Bitirme_Projesi_Raporu.md successfully!")
    else:
        print("Marker 8.17 not found in md!")
else:
    print("Warning: old_md_table not found in md!")

