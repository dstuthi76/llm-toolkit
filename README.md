# llm-toolkit

**Eight small, focused, production-grade tools for problems that show up once you actually run LLM applications in the real world — agent debugging, context management, multi-agent QA, cost control, output reliability, fine-tuning eval, privacy-preserving RAG, and prompt regression testing.**

Each tool lives in its own directory as an independently installable Python package: its own `pyproject.toml`, its own test suite (all offline, zero API keys required to run), its own README, and a runnable `example.py` that demonstrates the core value in under 5 seconds.

## The packages

| # | Package | Problem it solves | Tests |
|---|---|---|---|
| 1 | [`tracewright`](./01-trace-debugger) | Agent fails after 15 steps — which one caused it? | 8 ✅ |
| 2 | [`contextprune`](./02-context-pruning) | More RAG context isn't always better context | 6 ✅ |
| 3 | [`agentlint`](./03-agent-lint) | Multi-agent conversations loop, drift, or give up silently | 5 ✅ |
| 4 | [`modelrouter`](./04-model-router) | Every query hits the expensive model, even trivial ones | 4 ✅ |
| 5 | [`reliably`](./05-reliable-extract) | Valid JSON ≠ correct JSON; blind retries repeat mistakes | 7 ✅ |
| 6 | [`finetune_eval`](./06-finetune-eval) | Aggregate accuracy hides per-example regressions | 4 ✅ |
| 7 | [`privacyrag`](./07-privacy-rag) | RAG over sensitive docs needs redaction + an audit trail | 13 ✅ |
| 8 | [`promptcheck`](./08-prompt-regression) | Prompt edits silently break things nobody tests for | 11 ✅ |

**58/58 tests pass**, offline, no API keys required. Run any package's suite yourself:

```bash
cd 01-trace-debugger && python -m unittest discover -s tests -v
```

## A shared design decision, explained once

Every package ships with a **mock/offline implementation** of anything that would normally require a network call or paid API (an LLM judge, an embedding model, a cloud provider) — alongside a **real implementation** you swap in for production. This means:

- You can clone this repo and run every test suite and every example with zero setup, zero API keys, and no internet connection.
- The "swap point" between mock and real is always one constructor argument (e.g. `Judge(client=OpenAICompatibleClient(...))` instead of `Judge()`), documented in each package's README.
- The core logic — the actual algorithm each tool exists for — is real, tested, and runnable today, not a stub waiting for you to fill in.

## Try one right now

```bash
git clone <this-repo-url>
cd llm-toolkit/01-trace-debugger
python3 example.py
open tracewright_report.html   # or just cat it -- it's plain HTML
```

Or the one with a real live HTTP server:

```bash
cd llm-toolkit/04-model-router
python3 example.py             # see routing decisions + cost savings printed
```

## Which one should I actually use first?

- Building agents and things silently fail? → `tracewright`
- RAG answers get worse as you add more documents? → `contextprune`
- Multiple agents talking to each other and something's off? → `agentlint`
- LLM API bill too high? → `modelrouter`
- Structured extraction keeps returning plausible-looking garbage? → `reliably`
- Just fine-tuned a model and not sure it actually helped? → `finetune_eval`
- Building RAG over medical/legal/financial documents? → `privacyrag`
- Prompt changes keep breaking things without anyone noticing? → `promptcheck`

## Contributing

Each package is intentionally small and single-purpose. PRs that keep that focus — a better default heuristic, a real-provider adapter, a bug fix — are especially welcome. See each package's own README for its specific roadmap.

## License

MIT — see individual package directories.
