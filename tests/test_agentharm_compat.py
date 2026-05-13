"""Reproduces AgentHarm's ``n_irrelevant_tools`` arithmetic shape as a special case.

We do NOT import AgentHarm. The contract is local arithmetic + additive merge:
- pool_size = N (literal count, depth_schedule=None)
- eval-state with K real tools → final state.tools has K + N entries
- Same seed → same selection across runs
"""

from inspect_ai.tool import ToolDef

from inspect_tools import context_exhaustion




async def test_n_irrelevant_tools_arithmetic_equivalence(make_state, dummy_tool_factory, noop_generate):
    eval_tool = dummy_tool_factory("real_eval_tool")
    state = make_state(sample_id="agentharm-task-1", tools=[eval_tool])
    solver = context_exhaustion(depth_schedule=None, pool_size=8)
    await solver(state, noop_generate)
    # min(pool_size=8, fixture Cat-A pool of 8) -> 8 fillers
    assert len(state.tools) == 1 + 8
    assert ToolDef(state.tools[0]).name == "real_eval_tool"


async def test_n_irrelevant_tools_deterministic(make_state, dummy_tool_factory, noop_generate):
    """Same sample_id + pool_size → identical filler selection across runs (mirrors
    AgentHarm-with-fixed-seed pattern)."""
    s1 = make_state(sample_id="agentharm-task-1", tools=[dummy_tool_factory("real")])
    s2 = make_state(sample_id="agentharm-task-1", tools=[dummy_tool_factory("real")])
    solver = context_exhaustion(depth_schedule=None, pool_size=5)
    await solver(s1, noop_generate)
    await solver(s2, noop_generate)
    n1 = [ToolDef(t).name for t in s1.tools[1:]]
    n2 = [ToolDef(t).name for t in s2.tools[1:]]
    assert n1 == n2
    assert len(set(n1)) == 5  # all distinct


async def test_n_irrelevant_tools_clamps_to_pool_size(make_state, dummy_tool_factory, noop_generate):
    """pool_size > library size → injects all available, no error (AgentHarm uses
    min(n_irrelevant_tools, len(irrelevant_tool_names)))."""
    state = make_state(sample_id="x", tools=[dummy_tool_factory("real")])
    solver = context_exhaustion(depth_schedule=None, pool_size=999)
    await solver(state, noop_generate)
    # Cat-A pool has 8 schemas in our fixture → injects all 8.
    assert len(state.tools) == 1 + 8
