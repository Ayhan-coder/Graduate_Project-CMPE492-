#!/bin/bash
# Cumulative acoustic-intervention ladder: (2) +energy VAD completion, (3) +VAD+glossary biasing.
cd "C:/Users/Slayer/Desktop/Bitirme Projesi/experiments" || exit 1
export PYTHONIOENCODING=utf-8
IDS="TR-1 TR-2 TR-3 EN-1 EN-2 EN-3 EN-4"

echo "=== VAD RESUME $(date) ===" >> logs/vad_large_v3.log
python run_asr.py --model openai/whisper-large-v3 --use-vad \
  --ids $IDS --output outputs/asr-vad-large-v3.jsonl --resume \
  >> logs/vad_large_v3.log 2>&1
echo "=== VAD EXIT $? $(date) ===" >> logs/vad_large_v3.log

echo "=== VAD+BIAS START $(date) ===" >> logs/vad_bias_large_v3.log
python run_asr.py --model openai/whisper-large-v3 --use-vad --prompt-glossary \
  --ids $IDS --output outputs/asr-vad-bias-large-v3.jsonl --resume \
  >> logs/vad_bias_large_v3.log 2>&1
echo "=== VAD+BIAS EXIT $? $(date) ===" >> logs/vad_bias_large_v3.log
