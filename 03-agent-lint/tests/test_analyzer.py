import unittest
from agentlint.analyzer import ConversationAnalyzer, Message


class TestConversationAnalyzer(unittest.TestCase):
    def test_detects_repetition_loop(self):
        messages = [
            Message(0, "researcher", "I think we should check the sales data for Q3 trends."),
            Message(1, "critic", "Have you considered checking the sales data for Q3 trends?"),
            Message(2, "researcher", "Yes, I think we should check the sales data for Q3 trends."),
        ]
        analyzer = ConversationAnalyzer(loop_similarity_threshold=0.5)
        findings = analyzer.analyze("Analyze Q3 sales performance", messages)
        loop_findings = [f for f in findings if f.kind == "loop"]
        self.assertGreater(len(loop_findings), 0)

    def test_detects_drift_from_original_task(self):
        messages = [
            Message(0, "researcher", "Let's analyze Q3 sales performance and revenue trends."),
            Message(1, "critic", "Q3 sales revenue trends look strong across all regions."),
            Message(2, "researcher", "By the way, what's a good recipe for chocolate cake?"),
            Message(3, "critic", "Chocolate cake needs flour, sugar, eggs, and cocoa powder."),
            Message(4, "researcher", "I love baking cakes on weekends with my family."),
            Message(5, "critic", "Cakes are best served with vanilla ice cream on top."),
        ]
        analyzer = ConversationAnalyzer(drift_drop_threshold=0.1)
        findings = analyzer.analyze("Analyze Q3 sales performance and revenue trends", messages)
        drift_findings = [f for f in findings if f.kind == "drift"]
        self.assertGreater(len(drift_findings), 0)

    def test_detects_premature_termination(self):
        messages = [
            Message(0, "researcher", "Let's figure out the root cause of the outage."),
            Message(1, "critic", "I'm not sure what caused it, I don't know how to proceed further."),
        ]
        analyzer = ConversationAnalyzer()
        findings = analyzer.analyze("Find the root cause of the outage", messages)
        term_findings = [f for f in findings if f.kind == "premature_termination"]
        self.assertEqual(len(term_findings), 1)
        self.assertEqual(term_findings[0].message_indices, [1])

    def test_clean_conversation_has_no_loop_or_termination_findings(self):
        messages = [
            Message(0, "researcher", "Revenue grew 12% in Q3 driven by the new product line."),
            Message(1, "critic", "Good, let's confirm that with the finance team's numbers."),
            Message(2, "researcher", "Confirmed: finance reports 12.3% growth, consistent with our finding."),
            Message(3, "critic", "Great, final answer: Q3 revenue grew approximately 12%."),
        ]
        analyzer = ConversationAnalyzer()
        findings = analyzer.analyze("Determine Q3 revenue growth", messages)
        self.assertEqual([f for f in findings if f.kind == "loop"], [])
        self.assertEqual([f for f in findings if f.kind == "premature_termination"], [])

    def test_empty_messages_returns_no_findings(self):
        analyzer = ConversationAnalyzer()
        self.assertEqual(analyzer.analyze("some task", []), [])


if __name__ == "__main__":
    unittest.main()
