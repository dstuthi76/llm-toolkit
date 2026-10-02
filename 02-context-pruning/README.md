# contextprune

**More context isn't always better context. Score, prune, and reorder chunks before they hit your prompt.**

Big context windows tempt you to dump everything in — but irrelevant chunks measurably hurt answer quality ("context rot"), and models pay less attention to information buried in the middle of a long prompt ("lost in the middle"). `contextprune` scores each candidate chunk's relevance to the query, drops the low-value ones under a token budget, and reorders the survivors so the most relevant ones sit at the start and end of the prompt.

## Why

- Dumping every retrieved chunk into the prompt costs tokens *and* can lower accuracy.
- Nobody measures whether their RAG pipeline's context is actually helping vs. hurting.
- `contextprune` gives you a `prune()` call plus a benchmark harness to prove the difference on your own data.

## Install

```bash
pip install contextprune
# for real semantic embeddings instead of the offline TF-IDF default:
pip install "contextprune[semantic]"
```

## Quick start

```python
from contextprune import ContextPruner, Chunk

chunks = [
    Chunk(text="Refund policy: returns within 30 days for a full refund."),
    Chunk(text="Our company was founded in 2019 in Austin."),
    Chunk(text="Damaged items are refundable regardless of the 30 day window."),
]

pruner = ContextPruner()
result = pruner.prune(query="What is the refund policy?", chunks=chunks, token_budget=100)

print(result.ordered_text)   # ready to paste straight into your prompt
print(f"{len(result.kept)}/{len(chunks)} chunks kept, {result.total_tokens} tokens")
```

## Embedders: offline by default, semantic when you want it

The default `TfidfEmbedder` is intentionally dependency-free and offline — pure word-overlap scoring via scikit-learn, good enough to build and test the pruning logic without downloading a model or hitting a network. It is **not** semantic: it won't know "car" and "automobile" are related. For real semantic relevance:

```python
from contextprune.embeddings import SentenceTransformerEmbedder
pruner = ContextPruner(embedder=SentenceTransformerEmbedder())
```

## Proving it actually helps: the benchmark harness

Don't take pruning on faith — measure it against your own answer function:

```python
from contextprune.benchmark import run_benchmark, BenchmarkCase
from contextprune import Chunk

cases = [
    BenchmarkCase(
        query="What is the refund window?",
        chunks=[...],
        expected_substring="30 days",
    ),
    # ... more cases
]

result = run_benchmark(cases, answer_fn=my_llm_call, token_budget=500)
print(f"Full context accuracy:   {result.full_context_accuracy:.0%}")
print(f"Pruned context accuracy: {result.pruned_context_accuracy:.0%}")
print(f"Avg tokens: {result.avg_tokens_full:.0f} -> {result.avg_tokens_pruned:.0f}")
```

## How ordering works

After scoring, survivors are placed alternately at the front and back of the output (highest score first, then second-highest last, third-highest second, etc.) rather than in original document order. This directly targets the empirically observed "lost in the middle" effect where LLMs under-attend to information in the center of a long context.

## Development

```bash
python -m unittest discover -s tests -v
```

## Roadmap

- [ ] Cross-encoder reranking option (more accurate than embedding cosine similarity, more expensive)
- [ ] Near-duplicate chunk detection and de-duplication
- [ ] Streaming/incremental pruning for very large candidate sets

## License

MIT
