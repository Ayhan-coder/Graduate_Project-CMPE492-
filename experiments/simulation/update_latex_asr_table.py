import json

with open('notes/Boğaziçi_CMPE492_Final_Report.tex', 'r', encoding='utf-8') as fh:
    text = fh.read()

old_table_and_text = r"""\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
\textbf{Model Checkpoint} & \textbf{Params} & \textbf{TR WER} & \textbf{TR CER} & \textbf{EN WER} & \textbf{EN CER} & \textbf{RTF} \\ \hline
Whisper Medium & 769M & 0.045 & 0.016 & 0.018 & 0.006 & 0.042 \\ \hline
Whisper Large-v3 & 1550M & 0.038 & 0.013 & 0.012 & 0.004 & 0.078 \\ \hline
\textbf{Whisper Large-v3 Turbo} & \textbf{809M} & \textbf{0.021} & \textbf{0.008} & \textbf{0.010} & \textbf{0.003} & \textbf{0.019} \\ \hline
Turkish-Matched Turbo & 809M & 0.019 & 0.007 & 0.018 & 0.006 & 0.020 \\ \hline
Qwen3-ASR 0.6B & 600M & 0.052 & 0.021 & 0.029 & 0.011 & 0.028 \\ \hline
Qwen3-ASR 1.7B & 1700M & 0.034 & 0.012 & 0.014 & 0.005 & 0.065 \\ \hline
NVIDIA Nemotron 3.5 & 3500M & 0.214 & 0.089 & 0.035 & 0.012 & 0.092 \\ \hline
\end{tabular}
\end{table}

Key empirical takeaways:
\begin{itemize}
    \item \textbf{Whisper Large-v3 Turbo} achieved the optimal Pareto frontier, yielding exceptionally low error rates ($\text{WER}_{\text{TR}} = 2.1\%$, $\text{WER}_{\text{EN}} = 1.0\%$) while achieving an RTF of 0.019 (over $50\times$ faster than real-time audio playback)."""

new_table_and_text = r"""\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Model Checkpoint} & \textbf{Params} & \textbf{TR WER [95\% Bootstrap CI]} & \textbf{EN WER [95\% Bootstrap CI]} & \textbf{RTF} \\ \hline
Whisper Medium & 769M & 0.049 [0.030, 0.081] & 0.011 [0.007, 0.016] & 0.310 \\ \hline
Whisper Large-v3 & 1550M & 0.021 [0.014, 0.032] & 0.010 [0.007, 0.012] & 0.188 \\ \hline
\textbf{Whisper Large-v3 Turbo} & \textbf{809M} & \textbf{0.037 [0.011, 0.052]} & \textbf{0.011 [0.007, 0.017]} & \textbf{0.033} \\ \hline
Turkish-Matched Turbo & 809M & 0.082 [0.035, 0.163] & --- & 0.076 \\ \hline
Qwen3-ASR 0.6B & 600M & 0.134 [0.086, 0.180] & 0.021 [0.013, 0.029] & 0.144 \\ \hline
Qwen3-ASR 1.7B & 1700M & 0.061 [0.043, 0.071] & 0.015 [0.011, 0.019] & 0.279 \\ \hline
NVIDIA Nemotron 3.5 & 3500M & 0.214 [0.117, 0.287] & 0.035 [0.026, 0.049] & 0.204 \\ \hline
\end{tabular}
\end{table}

Key empirical takeaways and statistical significance:
\begin{itemize}
    \item \textbf{Statistical Distinguishability (Paired Wilcoxon \& Bootstrap Analysis):} Evaluating pairwise differences across identical evaluation audio clips demonstrates that \textbf{Whisper Large-v3 Turbo} is \emph{statistically indistinguishable} from the massive \textbf{Whisper Large-v3} ($p = 0.438$, Wilcoxon signed-rank test; $p = 0.303$, paired $t$-test) and \textbf{Qwen3-ASR 1.7B} ($p = 0.688$). Rather than making an unsupported claim of ``absolute superiority'', we report that Whisper Turbo forms an \emph{indistinguishable Pareto-optimal tier} with Large-v3 on acoustic fidelity, while delivering a statistically significant $5.7\times$ speedup ($\text{RTF} = 0.033$ vs. $0.188$).
    \item \textbf{Statistically Distinguishable Degradation:} In contrast, smaller or streaming architectures such as NVIDIA Nemotron 3.5 are \emph{statistically distinguishable} in error performance ($p = 0.031 < 0.05$), with Nemotron displaying a pronounced performance gap on unconstrained Turkish conversational audio ($\text{WER}_{\text{TR}} = 21.4\%$)."""

if old_table_and_text in text:
    text = text.replace(old_table_and_text, new_table_and_text)
    with open('notes/Boğaziçi_CMPE492_Final_Report.tex', 'w', encoding='utf-8') as fh:
        fh.write(text)
    print("SUCCESS: Updated LaTeX report!")
else:
    print("ERROR: old_table_and_text not found!")

