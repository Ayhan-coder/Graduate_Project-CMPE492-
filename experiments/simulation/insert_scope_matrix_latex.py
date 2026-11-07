import io

with open('notes/Boğaziçi_CMPE492_Final_Report.tex', 'r', encoding='utf-8') as fh:
    text = fh.read()

scope_matrix_latex = r"""\section{Empirical Scope Matrix: Hypotheses, Datasets, and Evaluation Protocols}
\label{sec:scope_matrix}
To maintain complete methodological transparency across the heterogeneous experimental evaluations conducted in this project, Table~\ref{tab:scope_matrix} provides a unified \textbf{One-Page Scope Matrix}. This matrix links every empirical claim to its exact evaluation partition, sample size ($n$), ground truth establishment protocol (\texttt{gold\_how}), and verification mechanism:

\begin{table}[htbp]
\centering
\caption{One-Page Scope Matrix: Empirical Claims, Datasets, Ground Truth, and Evaluators.}
\label{tab:scope_matrix}
\small
\begin{tabularx}{\textwidth}{lXcXl}
\toprule
\textbf{Empirical Claim} & \textbf{Evaluation Subset} & \textbf{$n$} & \textbf{Ground Truth (\texttt{gold\_how})} & \textbf{Evaluation Mechanism} \\
\midrule
Acoustic Fidelity \& Speed & Gold Pilot Clips & $7$ clips & Dual Human Phonetic Transcripts & WER, CER, RTF, Bootstrap CI, Wilcoxon \\
Cross-Layer Error Drift & Acoustic Degradation Ladder & $14$ steps & Verified Closed Terminology List & Precision, Recall, F1, Sensitivity Slopes \\
Terminology Extraction Parity & Double-Annotated Sample & $454$ items & Double-Blind Human Verification & Cohen's $\kappa=0.531$, PABAK$=0.669$ \\
Cross-Family Judge Autonomy & Cross-Family Panel & $454$ pairs & Verified Human Ground Truth & Qwen2.5-7B vs. Mistral-7B-Instruct \\
Model Parameter Scaling & Grand 7-Tier Scaling Suite & $7$ models & Structured JSON Target Schema & Schema Adherence, Ate\c{s}man B1, VRAM \\
Multi-Domain SFT (7B / 14B) & Domain-Dense Corpus & $2{,}296$ / $4{,}666$ & LeakageGuard Synthetic ChatML & Validation Loss, Bloom Shift ($0\%\to 62.5\%$) \\
Controllable CEFR B1/B2/C1 & Multi-Tier Triplet Dataset & $4{,}666$ train / $586$ val & Level-Conditioned Prompts & Syllable-Parsed Ate\c{s}man ($91.7\%$ Accuracy) \\
Multi-Turn Socratic Tutoring & Synthetic Student Personas & $250$ turns & Pedagogical Misconceptions & Socratic Questioning, Zero Spoon-Feeding \\
Curriculum Concept DAG & Full Curriculum Corpus & $33\to 133$ & Machine Transcripts (\texttt{machine\_only}) & NetworkX Directed Acyclic Topological Sort \\
\bottomrule
\end{tabularx}
\end{table}

"""

target = r"\section{Full-Corpus Batch Pipeline and Scaled Pedagogical DAG Analysis (33 Verified Lectures)}"
if target in text:
    text = text.replace(target, scope_matrix_latex + target)
    with open('notes/Boğaziçi_CMPE492_Final_Report.tex', 'w', encoding='utf-8') as fh:
        fh.write(text)
    print("SUCCESS: Added Scope Matrix to Boğaziçi_CMPE492_Final_Report.tex!")
else:
    print("ERROR: Target not found in LaTeX report!")

