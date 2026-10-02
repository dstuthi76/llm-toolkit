"""
Example: pruning a pile of mixed-relevance chunks down to a token budget,
and reordering survivors to fight the "lost in the middle" effect.

Run: python3 example.py
"""
from contextprune import ContextPruner, Chunk

chunks = [
    Chunk(text="Refund policy: customers may return items within 30 days of purchase for a full refund."),
    Chunk(text="Our company was founded in 2019 by two engineers in a garage in Austin."),
    Chunk(text="Refunds are processed to the original payment method within 5-7 business days."),
    Chunk(text="The office has a rooftop garden and a ping pong table on the third floor."),
    Chunk(text="Damaged or defective items are eligible for refund regardless of the 30 day window."),
    Chunk(text="Our support team is available 24/7 via chat, email, and phone."),
]

pruner = ContextPruner()
result = pruner.prune(
    query="What is the refund policy?",
    chunks=chunks,
    token_budget=60,
)

print(f"Kept {len(result.kept)} of {len(chunks)} chunks ({result.total_tokens} tokens)\n")
print("--- Final prompt context ---")
print(result.ordered_text)
print("\n--- All scores (for debugging) ---")
for s in sorted(result.scores, key=lambda s: s.relevance, reverse=True):
    marker = "KEPT" if s.chunk in result.kept else "cut "
    print(f"[{marker}] {s.relevance:.3f}  {s.chunk.text[:60]}")
