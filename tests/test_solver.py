import tiktoken

from inspect_context_pressure import context_pressure
from inspect_context_pressure._fixtures import FIXTURE_SCHEMAS
from inspect_context_pressure._inject import count_tools_tokens




def _enc():
    return tiktoken.get_encoding("cl100k_base")


async def test_pool_size_literal_when_depth_schedule_none(make_state, dummy_tool, noop_generate):
    """pool_size override path: depth_schedule=None → exact pool_size schemas added."""
    state = make_state(tools=[dummy_tool])
    solver = context_pressure(depth_schedule=None, pool_size=3)
    await solver(state, noop_generate)
    assert len(state.tools) == 1 + 3


async def test_per_trial_seed_deterministic(make_state, dummy_tool, noop_generate):
    """Same (sample_id, depth, shape_seed) → identical filler set across runs."""
    s1 = make_state(sample_id="x", tools=[dummy_tool], metadata={"target_tokens": 0})
    s2 = make_state(sample_id="x", tools=[dummy_tool], metadata={"target_tokens": 0})
    solver = context_pressure(depth_schedule=None, pool_size=4)
    await solver(s1, noop_generate)
    await solver(s2, noop_generate)
    names_1 = [type(t).__name__ + ":" + getattr(t, "__name__", "") for t in s1.tools[1:]]
    # Compare using underlying ToolDef name registry: easier to introspect via the wrapped callable
    from inspect_ai.tool import ToolDef

    n1 = [ToolDef(t).name for t in s1.tools[1:]]
    n2 = [ToolDef(t).name for t in s2.tools[1:]]
    assert n1 == n2
    assert len(n1) == 4


async def test_per_trial_seed_varies_with_depth(make_state, dummy_tool, noop_generate):
    """Different target_tokens → different sampling order (with high probability for our fixture)."""
    from inspect_ai.tool import ToolDef

    s1 = make_state(sample_id="x", tools=[dummy_tool], metadata={"target_tokens": 4_000})
    s2 = make_state(sample_id="x", tools=[dummy_tool], metadata={"target_tokens": 64_000})
    solver = context_pressure(pool_size=99)  # depth-schedule path; large fallback irrelevant
    await solver(s1, noop_generate)
    await solver(s2, noop_generate)
    n1 = [ToolDef(t).name for t in s1.tools[1:]]
    n2 = [ToolDef(t).name for t in s2.tools[1:]]
    # Either the count differs or the order differs.
    assert n1 != n2 or len(n1) != len(n2)


async def test_n_scales_to_target_tokens(make_state, dummy_tool, noop_generate):
    """When target_tokens is set, the post-injection token count approaches target.

    Uses a target the 8-schema Cat-A fixture pool can plausibly satisfy. With the
    real ICP-4 corpus (~thousands of schemas), 4K/16K/64K/256K targets all fit.
    Greedy fill stops the first time cumulative tokens reach the target, so the
    final total is at most one schema's overshoot above target.
    """
    enc = _enc()
    target = 600
    state = make_state(sample_id="x", tools=[dummy_tool], metadata={"target_tokens": target})
    solver = context_pressure()
    await solver(state, noop_generate)
    total = count_tools_tokens(state.tools, enc)
    cat_a_count = sum(1 for s in FIXTURE_SCHEMAS if s["content_category"] == "A_general_popular")
    n_added = len(state.tools) - 1
    if n_added < cat_a_count:
        # Saturated: target met (within one schema's overshoot).
        assert total >= target
    else:
        # Pool exhausted before target — accept under-saturation.
        assert n_added == cat_a_count


async def test_depth_path_falls_through_when_no_target_tokens(make_state, dummy_tool, noop_generate):
    """depth_schedule set but sample lacks target_tokens → uses pool_size fallback."""
    state = make_state(tools=[dummy_tool])
    solver = context_pressure(pool_size=2)  # default depth_schedule but no target_tokens
    await solver(state, noop_generate)
    assert len(state.tools) == 1 + 2


async def test_domain_filter_restricts_pool(make_state, dummy_tool, noop_generate):
    from inspect_ai.tool import ToolDef

    state = make_state(tools=[dummy_tool])
    solver = context_pressure(depth_schedule=None, pool_size=10, domain_filter=["cloud-ops"])
    await solver(state, noop_generate)
    cloud_names = {s["name"] for s in FIXTURE_SCHEMAS if s["domain"] == "cloud-ops"}
    injected = [ToolDef(t).name for t in state.tools[1:]]
    assert set(injected).issubset(cloud_names)
    assert len(injected) == len(cloud_names)  # min(pool_size=10, 2 cloud-ops schemas) == 2


async def test_content_category_B_isolates_vacuous_controls(make_state, dummy_tool, noop_generate):
    """ICP-7 Gamage-vs-Levy isolation arm: only Category B schemas get sampled."""
    from inspect_ai.tool import ToolDef

    state = make_state(tools=[dummy_tool])
    solver = context_pressure(
        depth_schedule=None, pool_size=10, content_category="B_vacuous_controls"
    )
    await solver(state, noop_generate)
    cat_b_names = {s["name"] for s in FIXTURE_SCHEMAS if s["content_category"] == "B_vacuous_controls"}
    injected = [ToolDef(t).name for t in state.tools[1:]]
    assert set(injected) == cat_b_names


async def test_exclude_names_blocks_collisions(make_state, dummy_tool, noop_generate):
    from inspect_ai.tool import ToolDef

    blocked = "github_create_pull_request"
    state = make_state(tools=[dummy_tool])
    solver = context_pressure(
        depth_schedule=None,
        pool_size=10,
        exclude_names=[blocked],
    )
    await solver(state, noop_generate)
    injected = [ToolDef(t).name for t in state.tools[1:]]
    assert blocked not in injected


async def test_extend_with_appends_user_schemas(make_state, dummy_tool, noop_generate):
    """User-supplied schemas appear in the candidate pool."""
    from inspect_ai.tool import ToolDef

    user_schema = {
        "name": "user_provided_widget",
        "description": "A user-supplied filler schema for testing extend_with.",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        "domain": "misc",
        "content_category": "A_general_popular",
        "mcp_server": "test/user",
    }
    state = make_state(tools=[dummy_tool])
    solver = context_pressure(
        depth_schedule=None,
        pool_size=20,  # larger than pool to guarantee inclusion
        extend_with=[user_schema],
    )
    await solver(state, noop_generate)
    injected = [ToolDef(t).name for t in state.tools[1:]]
    assert "user_provided_widget" in injected


async def test_composition_spec_exclude_keywords(make_state, dummy_tool, noop_generate):
    """exclude_keywords in composition_spec filters by name/description substring."""
    from inspect_ai.tool import ToolDef

    state = make_state(tools=[dummy_tool])
    solver = context_pressure(
        depth_schedule=None,
        pool_size=10,
        composition_spec={"exclude_keywords": ["snowflake", "stripe"]},
    )
    await solver(state, noop_generate)
    injected = {ToolDef(t).name for t in state.tools[1:]}
    assert "snowflake_execute_query" not in injected
    assert "stripe_create_charge" not in injected


async def test_e2e_with_simple_task_preserves_eval_tool(make_state, dummy_tool, noop_generate):
    """End-to-end integration: solver runs, eval's own tool preserved, scorer not touched."""
    from inspect_ai.tool import ToolDef

    state = make_state(tools=[dummy_tool], metadata={"target_tokens": 2_000})
    initial_id = id(state.tools[0])
    initial_messages = list(state.messages)
    solver = context_pressure()
    out_state = await solver(state, noop_generate)
    # Eval-owned tool preserved.
    assert id(out_state.tools[0]) == initial_id
    assert ToolDef(out_state.tools[0]).name == "eval_owned_tool"
    # Filler tools were added.
    assert len(out_state.tools) > 1
    # Messages untouched (we never invoked generate).
    assert list(out_state.messages) == initial_messages
