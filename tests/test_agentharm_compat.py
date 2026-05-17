"""AgentHarm n_irrelevant_tools shape compatibility.

Mirrors inspect_evals/agentharm/utils.py:225-247 (`n_irrelevant_tools`): fixed-count
injection of irrelevant tools alongside the eval's own tools. context_exhaustion
reproduces this shape via `n_filler=N` with `target_tokens=None`.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from inspect_tools._fixtures import FIXTURE_SCHEMAS
from inspect_tools._solver import context_exhaustion


@pytest.fixture(autouse=True)
def _patch_corpus(monkeypatch):
    monkeypatch.setattr("inspect_tools._library._CORPUS_CACHE", FIXTURE_SCHEMAS)


async def test_k_eval_tools_plus_n_filler(make_state, dummy_tool_factory, noop_generate):
    """K eval tools + n_filler=N → state.tools has exactly K+N entries."""
    K = 3
    N = 4
    eval_tools = [dummy_tool_factory(f"eval_{i}") for i in range(K)]
    solver = context_exhaustion(n_filler=N)
    state = make_state(tools=eval_tools)
    await solver(state, noop_generate)
    assert len(state.tools) == K + N


async def test_deterministic_across_solver_instances(make_state, dummy_tool, noop_generate):
    solver_a = context_exhaustion(n_filler=3)
    solver_b = context_exhaustion(n_filler=3)
    state_a = make_state(tools=[dummy_tool], sample_id="sample-x")
    state_b = make_state(tools=[dummy_tool], sample_id="sample-x")
    await solver_a(state_a, noop_generate)
    await solver_b(state_b, noop_generate)
    names_a = state_a.metadata["inspect_tools"]["context_exhaustion"]["injected_tool_names"]
    names_b = state_b.metadata["inspect_tools"]["context_exhaustion"]["injected_tool_names"]
    assert names_a == names_b


async def test_no_model_api_calls(make_state, dummy_tool, noop_generate, mock_model):
    """AgentHarm-compat mode never invokes count_tool_tokens."""
    solver = context_exhaustion(n_filler=3)
    state = make_state(tools=[dummy_tool])
    with patch("inspect_tools._solver.get_model", return_value=mock_model):
        await solver(state, noop_generate)
    mock_model.count_tool_tokens.assert_not_called()
