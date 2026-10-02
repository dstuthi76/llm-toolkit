# modelrouter

**Send easy queries to a cheap model, escalate to an expensive one only when needed. Drop-in OpenAI-compatible proxy — change one URL, keep all your existing code.**

Most apps send every request to one expensive model. A large fraction of real queries don't need it. `modelrouter` tries the cheap model first, runs a fast confidence check on the answer, and only calls the expensive model when the cheap one looks unreliable — then reports exactly how much you saved vs. an always-expensive baseline.

## Why

- GPT-4-class models can cost 10-50x more per token than smaller models.
- Most of that spend is on queries a cheap model could have handled fine.
- `modelrouter` runs as a proxy server that mimics the OpenAI API, so adopting it is a one-line `base_url` change — not a rewrite.

## Install

```bash
pip install modelrouter
# to hit real OpenAI-compatible endpoints instead of mocks:
pip install "modelrouter[openai]"
```

## Quick start (as a library)

```python
from modelrouter import Router, MockProvider

cheap = MockProvider("gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
                      behavior=lambda p: call_cheap_model(p))
expensive = MockProvider("gpt-4o", cost_per_1k_input=0.0025, cost_per_1k_output=0.01,
                          behavior=lambda p: call_expensive_model(p))

router = Router(cheap, expensive, confidence_threshold=0.6)
decision = router.route("What's 15% of 80?")

print(decision.result.text, decision.provider_used, decision.escalated)
print(router.tracker.summary())
# 5 requests | escalation rate: 40% | spent: $0.0004 | would have spent: $0.0007 | saved: 38%
```

Swap `MockProvider` for `OpenAICompatibleProvider(name, model, api_key, cost_per_1k_input, cost_per_1k_output)` to hit real models.

## Quick start (as a drop-in proxy)

```bash
export OPENAI_API_KEY=sk-...
python -m modelrouter        # starts a server on :8000
```

Then in your existing app, change only this line:

```python
client = OpenAI(base_url="http://localhost:8000/v1")  # was: api.openai.com
```

Everything else in your code — `client.chat.completions.create(...)` — stays identical. The proxy handles routing/escalation transparently and returns a standard chat-completion response with routing metadata attached under `x_router_meta`.

## How confidence is judged

The default `word_count_confidence` function is a cheap, offline heuristic (checks for hedge phrases like "I'm not sure", penalizes long/complex prompts). It's intentionally simple so the router works out of the box with zero extra API calls. For better accuracy in production, pass your own `confidence_fn`:

```python
def llm_judge_confidence(prompt: str, cheap_result) -> float:
    score = judge_model.complete(f"Rate confidence 0-1 that this fully answers: {prompt}\nAnswer: {cheap_result.text}")
    return float(score)

router = Router(cheap, expensive, confidence_fn=llm_judge_confidence)
```

## Cost tracking

Every routed request is logged with its actual cost and what it *would* have cost on the expensive model alone:

```python
print(router.tracker.summary())
print(router.tracker.total_saved)
print(router.tracker.escalation_rate)
```

## Development

```bash
python -m unittest discover -s tests -v
python3 example.py
```

## Roadmap

- [ ] `litellm` integration for zero-boilerplate multi-provider support
- [ ] Learned (not heuristic) query-difficulty classifier
- [ ] Redis-backed response caching for repeated queries
- [ ] Streaming response support in the proxy

## License

MIT
