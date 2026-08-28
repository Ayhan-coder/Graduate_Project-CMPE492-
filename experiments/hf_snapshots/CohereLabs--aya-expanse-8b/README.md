---
inference: false
library_name: transformers
language:
- en
- fr
- de
- es
- it
- pt
- ja
- ko
- zh
- ar
- el
- fa
- pl
- id
- cs
- he
- hi
- nl
- ro
- ru
- tr
- uk
- vi
license: cc-by-nc-4.0
extra_gated_prompt: "By submitting this form, you agree to the [License Agreement](https://cohere.com/c4ai-cc-by-nc-license)  and acknowledge that the information you provide will be collected, used, and shared in accordance with Cohere’s [Privacy Policy]( https://cohere.com/privacy). You’ll receive email updates about Cohere Labs and Cohere research, events, products and services. You can unsubscribe at any time." 
extra_gated_fields:
 Name: text
 Affiliation: text
 Country: country
 I agree to use this model for non-commercial use ONLY: checkbox
---

# Model Card for Aya Expanse 8B

<img src="./assets/aya-expanse-8B.png" width="650" style="margin-left:'auto' margin-right:'auto' display:'block'"/>

**Aya Expanse 8B** is an open-weight research release of a model with highly advanced multilingual capabilities. It focuses on pairing a highly performant pre-trained [Command family](https://huggingface.co/CohereLabs/c4ai-command-r-plus) of models with the result of a year’s dedicated research from [Cohere Labs](https://cohere.for.ai/), including [data arbitrage](https://arxiv.org/abs/2408.14960), [multilingual preference training](https://arxiv.org/abs/2407.02552), [safety tuning](https://arxiv.org/abs/2406.18682), and [model merging](https://arxiv.org/abs/2410.10801). The result is a powerful multilingual large language model.

This model card corresponds to the 8-billion version of the Aya Expanse model. We also released an 32-billion version which you can find [here](https://huggingface.co/CohereLabs/aya-expanse-32B).

- Developed by: [Cohere Labs](https://cohere.for.ai/) 
- Point of Contact: [Cohere Labs](https://cohere.for.ai/)
- License: [CC-BY-NC](https://cohere.com/cohere-labs-cc-by-nc-license), requires also adhering to [Cohere Lab's Acceptable Use Policy](https://docs.cohere.com/docs/cohere-labs-acceptable-use-policy)
- Model: Aya Expanse 8B
- Model Size: 8 billion parameters

### Supported Languages

We cover 23 languages: Arabic, Chinese (simplified & traditional), Czech, Dutch, English, French, German, Greek, Hebrew, Hebrew, Hindi, Indonesian, Italian, Japanese, Korean, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Turkish, Ukrainian, and Vietnamese.

### Try it: Aya Expanse in Action

Use the [Cohere playground](https://dashboard.cohere.com/playground/chat) or our [Hugging Face Space](https://huggingface.co/spaces/CohereLabs/aya_expanse) for interactive exploration.

### How to Use Aya Expanse

Install the transformers library and load Aya Expanse 8B as follows:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "CohereLabs/aya-expanse-8b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Format the message with the chat template
messages = [{"role": "user", "content": "Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

gen_tokens = model.generate(
    input_ids, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.3,
    )

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
```

### Example Notebooks

**Fine-Tuning:**
- [Detailed Fine-Tuning Notebook](https://colab.research.google.com/drive/1ryPYXzqb7oIn2fchMLdCNSIH5KfyEtv4).

**Community-Contributed Use Cases:**:

The following notebooks contributed by *Cohere Labs Community* members show how Aya Expanse can be used for different use cases:
- [Mulitlingual Writing Assistant](https://colab.research.google.com/drive/1SRLWQ0HdYN_NbRMVVUHTDXb-LSMZWF60)
- [AyaMCooking](https://colab.research.google.com/drive/1-cnn4LXYoZ4ARBpnsjQM3sU7egOL_fLB?usp=sharing)
- [Multilingual Question-Answering System](https://colab.research.google.com/drive/1bbB8hzyzCJbfMVjsZPeh4yNEALJFGNQy?usp=sharing)


## Model Details

**Input**: Models input text only.

**Output**: Models generate text only.

**Model Architecture**: Aya Expanse 8B is an auto-regressive language model that uses an optimized transformer architecture. Post-training includes supervised finetuning, preference training, and model merging.

**Languages covered**: The model is particularly optimized for multilinguality and supports the following languages: Arabic, Chinese (simplified & traditional), Czech, Dutch, English, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Korean, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Turkish, Ukrainian, and Vietnamese

**Context length**: 8K

For more details about how the model was trained, check out [our blogpost](https://huggingface.co/blog/aya-expanse).

### Evaluation

We evaluated Aya Expanse 8B against Gemma 2 9B, Llama 3.1 8B, Ministral 8B, and Qwen 2.5 7B using the `dolly_human_edited` subset from the [Aya Evaluation Suite dataset](https://huggingface.co/datasets/CohereLabs/aya_evaluation_suite) and m-ArenaHard, a dataset based on the [Arena-Hard-Auto dataset](https://huggingface.co/datasets/lmarena-ai/arena-hard-auto-v0.1) and translated to the 23 languages we support in Aya Expanse 8B. Win-rates were determined using gpt-4o-2024-08-06 as a judge. For a conservative benchmark, we report results from gpt-4o-2024-08-06, though gpt-4o-mini scores showed even stronger performance.

The m-ArenaHard dataset, used to evaluate Aya Expanse’s capabilities, is publicly available [here](https://huggingface.co/datasets/CohereLabs/m-ArenaHard).

<img src="./assets/winrates_marenahard_complete.png" width="650" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
<img src="./assets/winrates_dolly.png" width="650" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
<img src="./assets/winrates_by_lang.png"  width="650" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
<img src="./assets/winrates_step_by_step.png"  width="650" style="margin-left:'auto' margin-right:'auto' display:'block'"/>


### Model Card Contact

For errors or additional questions about details in this model card, contact labs@cohere.com

### Terms of Use

We hope that the release of this model will make community-based research efforts more accessible, by releasing the weights of a highly performant multilingual model to researchers all over the world. This model is governed by a [CC-BY-NC](https://cohere.com/cohere-labs-cc-by-nc-license), requires also adhering to [Cohere Lab's Acceptable Use Policy](https://docs.cohere.com/docs/cohere-labs-acceptable-use-policy)

### Cite
You can cite Aya Expanse using:

```
@misc{dang2024ayaexpansecombiningresearch,
      title={Aya Expanse: Combining Research Breakthroughs for a New Multilingual Frontier}, 
      author={John Dang and Shivalika Singh and Daniel D'souza and Arash Ahmadian and Alejandro Salamanca and Madeline Smith and Aidan Peppin and Sungjin Hong and Manoj Govindassamy and Terrence Zhao and Sandra Kublik and Meor Amer and Viraat Aryabumi and Jon Ander Campos and Yi-Chern Tan and Tom Kocmi and Florian Strub and Nathan Grinsztajn and Yannis Flet-Berliac and Acyr Locatelli and Hangyu Lin and Dwarak Talupuru and Bharat Venkitesh and David Cairuz and Bowen Yang and Tim Chung and Wei-Yin Ko and Sylvie Shang Shi and Amir Shukayev and Sammie Bae and Aleksandra Piktus and Roman Castagné and Felipe Cruz-Salinas and Eddie Kim and Lucas Crawhall-Stein and Adrien Morisot and Sudip Roy and Phil Blunsom and Ivan Zhang and Aidan Gomez and Nick Frosst and Marzieh Fadaee and Beyza Ermis and Ahmet Üstün and Sara Hooker},
      year={2024},
      eprint={2412.04261},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.04261}, 
}
```
