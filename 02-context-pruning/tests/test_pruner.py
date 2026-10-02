import unittest
from contextprune.pruner import ContextPruner, Chunk


class TestContextPruner(unittest.TestCase):
    def test_relevant_chunk_scores_higher_than_irrelevant(self):
        pruner = ContextPruner()
        chunks = [
            Chunk(text="The refund policy allows returns within 30 days of purchase."),
            Chunk(text="Our office is located in downtown Seattle near the pier."),
            Chunk(text="Refunds are processed within 5 business days after approval."),
        ]
        scored = pruner.score("What is the refund policy?", chunks)
        by_text = {s.chunk.text: s.relevance for s in scored}
        self.assertGreater(
            by_text["The refund policy allows returns within 30 days of purchase."],
            by_text["Our office is located in downtown Seattle near the pier."],
        )

    def test_prune_respects_token_budget(self):
        pruner = ContextPruner()
        chunks = [Chunk(text="refund policy " * 50, token_count=200) for _ in range(5)]
        result = pruner.prune("refund policy", chunks, token_budget=450)
        self.assertLessEqual(result.total_tokens, 450)
        self.assertLessEqual(len(result.kept), 3)  # 200*3=600 > 450, so at most 2 fit

    def test_low_relevance_chunks_are_dropped_even_with_budget_left(self):
        pruner = ContextPruner(relevance_floor=0.3)
        chunks = [
            Chunk(text="refund policy return window thirty days", token_count=10),
            Chunk(text="completely unrelated text about gardening tomatoes", token_count=10),
        ]
        result = pruner.prune("refund policy", chunks, token_budget=1000)
        kept_texts = [c.text for c in result.kept]
        self.assertIn("refund policy return window thirty days", kept_texts)
        self.assertNotIn("completely unrelated text about gardening tomatoes", kept_texts)

    def test_empty_chunks_returns_empty_result(self):
        pruner = ContextPruner()
        result = pruner.prune("anything", [], token_budget=500)
        self.assertEqual(result.kept, [])
        self.assertEqual(result.ordered_text, "")

    def test_ordering_places_most_relevant_at_edges(self):
        pruner = ContextPruner()
        chunks = [
            Chunk(text="apple banana fruit market price", token_count=10),   # will be most relevant
            Chunk(text="irrelevant filler about weather clouds", token_count=10),
            Chunk(text="apple fruit market analysis report", token_count=10),  # second most relevant
        ]
        result = pruner.prune("apple fruit market", chunks, token_budget=1000)
        ordered_chunks_text = result.ordered_text.split("\n\n")
        # the least relevant chunk should not be first or last
        self.assertNotEqual(ordered_chunks_text[0], "irrelevant filler about weather clouds")
        self.assertNotEqual(ordered_chunks_text[-1], "irrelevant filler about weather clouds")


if __name__ == "__main__":
    unittest.main()
