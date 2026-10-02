"""
Example: a 2-agent conversation that loops, then drifts off-topic, then
ends unresolved -- agentlint flags all three.

Run: python3 example.py
"""
from agentlint import ConversationAnalyzer, Message
from agentlint.report import format_findings

task = "Determine why the checkout page conversion rate dropped last week"

messages = [
    Message(0, "researcher", "Let's check if the checkout page conversion rate dropped due to a bug."),
    Message(1, "critic", "Have you checked if the checkout page conversion rate dropped from a bug?"),
    Message(2, "researcher", "Right, we should check if a bug caused the checkout conversion rate to drop."),
    Message(3, "critic", "Anyway, our new office coffee machine is really nice."),
    Message(4, "researcher", "Yeah the coffee here is way better than the old place."),
    Message(5, "critic", "I'm not sure how to proceed further on this, let's stop here."),
]

analyzer = ConversationAnalyzer(loop_similarity_threshold=0.6, drift_drop_threshold=0.3)
findings = analyzer.analyze(task, messages)

print(format_findings(findings))
