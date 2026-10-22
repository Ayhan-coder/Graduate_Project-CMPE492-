@echo off
cd /d "C:\Users\Slayer\Desktop\Bitirme Projesi\experiments"
set PYTHONIOENCODING=utf-8
set PY=C:\Users\Slayer\AppData\Local\Programs\Python\Python312\python.exe
if not exist "%PY%" set PY=python

echo === VAD RESUME %DATE% %TIME% ===>> logs\vad_large_v3.log
"%PY%" run_asr.py --model openai/whisper-large-v3 --use-vad --ids TR-1 TR-2 TR-3 EN-1 EN-2 EN-3 EN-4 --output outputs/asr-vad-large-v3.jsonl --resume >> logs\vad_large_v3.log 2>&1
echo === VAD EXIT %%ERRORLEVEL%% %DATE% %TIME% ===>> logs\vad_large_v3.log

echo === VAD+BIAS START %DATE% %TIME% ===>> logs\vad_bias_large_v3.log
"%PY%" run_asr.py --model openai/whisper-large-v3 --use-vad --prompt-glossary --ids TR-1 TR-2 TR-3 EN-1 EN-2 EN-3 EN-4 --output outputs/asr-vad-bias-large-v3.jsonl --resume >> logs\vad_bias_large_v3.log 2>&1
echo === VAD+BIAS DONE %DATE% %TIME% ===>> logs\vad_bias_large_v3.log
