"""
Example: route 5 mixed queries through cheap/expensive mock models and
print the savings summary.

Run: python3 example.py
"""
from modelrouter import Router, MockProvider

def cheap_behavior(prompt: str) -> str:
    # Simulate a small model that answers simple factual/arithmetic
    # questions confidently, but hedges on genuinely complex ones --
    # in production this is a real LLM call, not a lookup table.
    simple_answers = {
        "6 times 7": "42",
        "capital of France": "Paris",
        "2 plus 2": "4",
    }
    for key, answer in simple_answers.items():
        if key in prompt:
            return answer
    return "I'm not sure, it depends on several nuanced factors here."


cheap = MockProvider(
    name="gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
    behavior=cheap_behavior,
)
expensive = MockProvider(
    name="gpt-4o", cost_per_1k_input=0.0025, cost_per_1k_output=0.01,
    behavior=lambda p: "Here is a detailed, well-reasoned answer to a hard question.",
)

router = Router(cheap, expensive, confidence_threshold=0.6)

queries = [
    "What is 6 times 7?",
    "What is the capital of France?",
    "Explain the tax implications of a cross-border merger with nuanced regulatory considerations.",
    "What is 2 plus 2?",
    "Analyze the geopolitical ramifications of this trade policy in detail.",
]

for q in queries:
    decision = router.route(q)
    tag = "ESCALATED" if decision.escalated else "cheap    "
    print(f"[{tag}] ({decision.provider_used:>12}) conf={decision.confidence:.2f}  \"{q[:50]}...\"")

print()
print(router.tracker.summary())
