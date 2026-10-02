# finetune-eval

**Fine-tuning tutorials stop at the loss curve. `finetune-eval` tells you whether fine-tuning actually made your model better — per example, not just on average — and catches regressions the aggregate score hides.**

Two models can score identically on aggregate accuracy while one is quietly worse at things it used to get right and better at new things — the average cancels out. `finetune-eval` runs your base and fine-tuned models on the same held-out set and shows a per-example diff, flagging regressions specifically.

## Why

- Aggregate accuracy (`"went from 80% to 85%"`) hides exactly what changed underneath.
- Fine-tuning on narrow domain data can silently degrade general capabilities — a classic and hard-to-spot failure mode.
- Most fine-tuning toolkits (Axolotl, Unsloth, etc.) train well but leave evaluation as an afterthought.

## Install

```bash
pip install finetune-eval
# to also handle the actual LoRA fine-tuning (via Unsloth):
pip install "finetune-eval[training]"
```

## Quick start

```python
from finetune_eval import EvalHarness, EvalCase, generate_html_report

cases = [
    EvalCase(input="What is your refund policy?", expected_substring="30 days"),
    EvalCase(input="What is 2 plus 2?", expected_substring="4"),
    # ... your held-out examples
]

harness = EvalHarness(
    base_model_fn=lambda prompt: base_model.generate(prompt),
    finetuned_model_fn=lambda prompt: finetuned_model.generate(prompt),
)
result = harness.compare(cases)

print(result.summary())
# Base accuracy: 33%  ->  Fine-tuned accuracy: 33%
# Improved: 1  |  Regressed: 1  |  Total examples: 3

for r in result.regressions:
    print(r.input, "\n  base:", r.base_output, "\n  finetuned:", r.finetuned_output)

with open("report.html", "w") as f:
    f.write(generate_html_report(result))
```

Open `report.html` for a color-coded side-by-side diff, regressions listed first.

## Wiring up real fine-tuning (Unsloth)

This package deliberately does not reimplement LoRA training — it wraps whatever training backend you use. A typical flow with [Unsloth](https://github.com/unslothai/unsloth):

```python
from unsloth import FastLanguageModel

base_model, tokenizer = FastLanguageModel.from_pretrained("unsloth/llama-3-8b-bnb-4bit")
# ... run your LoRA fine-tuning with Unsloth/peft as normal ...
finetuned_model = ...  # the trained model

def base_fn(prompt):
    return generate(base_model, tokenizer, prompt)

def finetuned_fn(prompt):
    return generate(finetuned_model, tokenizer, prompt)

harness = EvalHarness(base_fn, finetuned_fn)
result = harness.compare(your_held_out_cases)
```

**Important:** hold out your eval cases from the training data before you start training — this package assumes that split is already done; it doesn't enforce it for you (yet, see Roadmap).

## Development

```bash
python -m unittest discover -s tests -v
python3 example.py    # runs against fake models, no GPU needed, generates finetune_eval_report.html
```

## Roadmap

- [ ] Enforce/verify the train/eval split isn't leaking
- [ ] Judge-model-based scoring (not just substring match) for open-ended answers
- [ ] Per-tag accuracy breakdown (e.g. "domain" tasks improved, "general" tasks regressed)
- [ ] Direct Unsloth training wrapper (`finetune_eval train --config config.yaml`)

## License

MIT
