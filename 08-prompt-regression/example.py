"""
Example: catch a "silent prompt regression" -- someone tweaks a system
prompt and it starts giving worse answers, without anyone noticing until
this suite runs.

Run: python3 example.py
"""
import tempfile
import os
from promptcheck import PromptCase, run_prompt_suite, SnapshotStore, MockJudge

cases = [
    PromptCase(name="refund_policy", prompt="What's your refund policy?",
               expected_behavior="Refunds within 30 days, no questions asked"),
    PromptCase(name="shipping_time", prompt="How long does shipping take?",
               expected_behavior="Shipping takes 3 to 5 business days"),
]

with tempfile.TemporaryDirectory() as d:
    snapshot_path = os.path.join(d, "snapshots.json")
    store = SnapshotStore(snapshot_path)

    # --- "before" run: the prompt is working well ---
    def good_llm(prompt: str) -> str:
        if "refund" in prompt.lower():
            return "We offer refunds within 30 days, no questions asked."
        return "Shipping typically takes 3 to 5 business days."

    print("=== Run 1: baseline ===")
    result1 = run_prompt_suite(cases, llm_fn=good_llm, snapshot_store=store)
    print(result1.summary())

    # --- someone edits the system prompt, breaks the refund answer ---
    def regressed_llm(prompt: str) -> str:
        if "refund" in prompt.lower():
            return "Please contact support for assistance with your request."  # regressed!
        return "Shipping typically takes 3 to 5 business days."

    print("\n=== Run 2: after a prompt change ===")
    result2 = run_prompt_suite(cases, llm_fn=regressed_llm, snapshot_store=store,
                                judge=MockJudge(overlap_threshold=0.4))
    print(result2.summary())

    for r in result2.failed:
        print(f"\nFAILED: {r.case.name}")
        print(f"  Expected: {r.case.expected_behavior}")
        print(f"  Got:      {r.output}")
        print(f"  Reason:   {r.verdict.reason}")

    for r in result2.changed:
        print(f"\nCHANGED since last snapshot: {r.case.name}")
        print(f"  Was: {r.previous_snapshot_output}")
        print(f"  Now: {r.output}")
