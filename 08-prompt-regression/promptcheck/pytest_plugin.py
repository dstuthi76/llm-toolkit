"""
A pytest fixture that makes prompt tests read like ordinary pytest tests:

    def test_refund_answer(llm_case):
        output = llm_case("What's your refund policy?")
        assert_semantically_equivalent(output, "Refunds within 30 days, no questions asked")

Register it via conftest.py:
    pytest_plugins = ["promptcheck.pytest_plugin"]

Or via pyproject.toml entry point (see this package's own pyproject.toml).

Set your real LLM call via the PROMPTCHECK_LLM_FN environment convention,
or simply monkeypatch/override the `llm_case` fixture in your own conftest
if you need per-project wiring -- pytest fixtures are designed to be
overridden this way.
"""
from __future__ import annotations
import pytest


def _default_llm_fn(prompt: str) -> str:
    raise RuntimeError(
        "No real LLM call is wired up. Override the `llm_case` fixture in your "
        "own conftest.py to call your actual model, e.g.:\n\n"
        "    @pytest.fixture\n"
        "    def llm_case():\n"
        "        return lambda prompt: my_real_llm_call(prompt)\n"
    )


@pytest.fixture
def llm_case():
    """Returns a callable: prompt -> output. Override this in your conftest.py."""
    return _default_llm_fn
