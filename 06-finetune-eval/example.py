"""
Example: comparing a "base" model against a "fine-tuned" model on a held-out
eval set, catching a regression the aggregate score alone would hide.

This uses fake model functions so it runs with zero GPU/network access.
In production, base_model_fn/finetuned_model_fn wrap real inference calls
(see the README for wiring this up with Unsloth).

Run: python3 example.py
"""
from finetune_eval import EvalHarness, EvalCase, generate_html_report

# Simulate: fine-tuning made the model much better at company-specific
# refund questions, but slightly worse at a generic math question it used
# to get right (a classic fine-tuning regression -- overfitting to the
# new domain data degrades an unrelated general capability).

def base_model(prompt: str) -> str:
    if "refund" in prompt.lower():
        return "I don't have information about that."
    if "2 plus 2" in prompt.lower():
        return "4"
    return "I don't know."

def finetuned_model(prompt: str) -> str:
    if "refund" in prompt.lower():
        return "Our refund policy allows returns within 30 days."
    if "2 plus 2" in prompt.lower():
        return "I believe it's approximately four, though I'd verify."  # got worse!
    return "I don't know."

cases = [
    EvalCase(input="What is your refund policy?", expected_substring="30 days", tags=["domain"]),
    EvalCase(input="What is your return window?", expected_substring="30 days", tags=["domain"]),
    EvalCase(input="What is 2 plus 2?", expected_substring="4", tags=["general_math"]),
]

harness = EvalHarness(base_model_fn=base_model, finetuned_model_fn=finetuned_model)
result = harness.compare(cases)

print(result.summary())
print()
for c in result.regressions:
    print(f"REGRESSION on: {c.input}")
    print(f"  base:       {c.base_output}")
    print(f"  fine-tuned: {c.finetuned_output}")

with open("finetune_eval_report.html", "w") as f:
    f.write(generate_html_report(result))
print("\nWrote finetune_eval_report.html")
