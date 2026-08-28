---
library_name: transformers
language:
- tr
license: mit
base_model: openai/whisper-large-v3-turbo
tags:
- generated_from_trainer
datasets:
- mozilla-foundation/common_voice_17_0
metrics:
- wer
model-index:
- name: "Whisper Large v3 Turbo TR - Selim \xC7ava\u015F"
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Common Voice 17.0
      type: mozilla-foundation/common_voice_17_0
      config: tr
      split: test
      args: 'config: tr, split: test'
    metrics:
    - name: Wer
      type: wer
      value: 18.92291759135967
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Whisper Large v3 Turbo TR - Selim Çavaş

This model is a fine-tuned version of [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) on the Common Voice 17.0 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3123
- Wer: 18.9229

## Intended uses & limitations

This model can be used in various application areas, including

  - Transcription of Turkish language
  - Voice commands
  - Automatic subtitling for Turkish videos
    
## How To Use

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "selimc/whisper-large-v3-turbo-turkish"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

result = pipe("test.mp3")
print(result["text"])
```

## Training

Due to colab GPU constraints I was able to train using only the 25% of the Turkish data available in the Common Voice 17.0 dataset. 😔

Got a GPU to spare? Let's collaborate and take this model to the next level! 🚀

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 4000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer     |
|:-------------:|:-----:|:----:|:---------------:|:-------:|
| 0.1223        | 1.6   | 1000 | 0.3187          | 24.4415 |
| 0.0501        | 3.2   | 2000 | 0.3123          | 20.9720 |
| 0.0226        | 4.8   | 3000 | 0.3010          | 19.6183 |
| 0.001         | 6.4   | 4000 | 0.3123          | 18.9229 |


### Framework versions

- Transformers 4.45.2
- Pytorch 2.4.1+cu121
- Datasets 3.0.1
- Tokenizers 0.20.1
