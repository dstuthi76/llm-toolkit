import unittest
from contextprune.pruner import Chunk, ContextPruner
from contextprune.benchmark import run_benchmark, BenchmarkCase


class TestBenchmark(unittest.TestCase):
    def test_benchmark_reports_accuracy_and_token_savings(self):
        # Note: the default TfidfEmbedder is a literal-word-overlap matcher,
        # not a semantic one (it won't know "refund" and "refunds" are related
        # unless the exact word appears) -- so test text is phrased to share
        # vocabulary with the query, the same way you'd phrase things for a
        # real keyword-plus-TFIDF baseline. Swap in SentenceTransformerEmbedder
        # for true semantic matching in production.
        cases = [
            BenchmarkCase(
                query="What is the refund policy window in days?",
                chunks=[
                    Chunk(text="Our refund policy window is 30 days.", token_count=10),
                    Chunk(text="Unrelated: our founder started the company in a garage.", token_count=10),
                    Chunk(text="Unrelated: the office has a rooftop garden.", token_count=10),
                ],
                expected_substring="30 days",
            ),
        ]

        def fake_answer_fn(prompt: str) -> str:
            # Simulates a model that can only find the answer if it's
            # actually present verbatim in the prompt it was given.
            return "30 days" if "30 days" in prompt else "I don't know"

        result = run_benchmark(cases, fake_answer_fn, token_budget=15,
                                pruner=ContextPruner(relevance_floor=0.01))
        self.assertEqual(result.n_cases, 1)
        self.assertGreaterEqual(result.pruned_context_accuracy, result.full_context_accuracy)
        self.assertLess(result.avg_tokens_pruned, result.avg_tokens_full)


if __name__ == "__main__":
    unittest.main()
