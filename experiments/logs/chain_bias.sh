#!/bin/sh
cd "C:/Users/Slayer/Desktop/Bitirme Projesi/experiments"
while kill -0 1877 2>/dev/null; do sleep 20; done
export PYTHONIOENCODING=utf-8
python run_asr.py --model openai/whisper-large-v3 --use-vad --prompt-glossary --output outputs/asr-vad-bias-large-v3.jsonl
