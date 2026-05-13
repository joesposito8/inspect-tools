from inspect_ai.tool import ToolDef

from inspect_tools import context_exhaustion
from inspect_tools._fixtures import FIXTURE_SCHEMAS
from inspect_tools._inject import count_tools_tokens


async def test_pool_size_literal_when_depth_schedule_none(make_state, dummy_tool, noop_generate):
    """pool_size override path: depth_schedule=None → exact pool_size schemas added."""
    state = make_state(tools=[dummy_tool])
    solver = context_exhaustion(depth_schedule=None, pool_size=3)
    await solver(state, noop_generate)
    assert len(state.tools) == 1 + 3


async def test_per_trial_seed_deterministic(make_state, dummy_tool, noop_generate):
    """Same (sample_id, depth, shape_seed) → identical filler set across runs."""
    s1 = make_state(sample_id="x", tools=[dummy_tool], metadata={"target_tokens": 0})
    s2 = make_state(sample_id="x", tools=[dummy_tool], metadata={"target_tokens": 0})
    solver = context_exhaustion(depth_schedule=None, pool_size=4)
    await solver(s1, noop_generate)
    await solver(s2, noop_generate)
    n1 = [ToolDef(t).name for t in s1.tools[1:]]
    n2 = [ToolDef(t).name for t in s2.tools[1:]]
    assert n1 == n2
    assert len(n1) == 4


async def test_per_trial_seed_varies_with_depth(make_state, dummy_tool, noop_generate):
    """Different target_tokens → different sampling order or count."""
    s1 = make_state(sample_id="x", tools=[dummy_tool], metadata={"target_tokens": 4_000})
    s2 = make_state(sample_id="x", tools=[dummy_tool], metadata={"target_tokens": 64_000})
    solver = context_exhaustion(pool_size=99)
    await solver(s1, noop_generate)
    await solver(s2, noop_generate)
    n1 = [ToolDef(t).name for t in s1.tools[1:]]
    n2 = [ToolDef(t).name for t in s2.tools[1:]]
    assert n1 != n2 or len(n1) != len(n2)


async def test_n_scales_to_target_tokens(make_state, dummy_tool, noop_generate, encoding):
    """When target_tokens is set, the post-injection token count approaches target.

    Greedy fill stops the first time cumulative tokens reach the target. With the
    real ICP-4 corpus (~thousands of schemas), the deeper targets all fit; the
    8-schema Cat-A fixture only satisfies smaller targets, so we accept either
    saturation or pool exhaustion.
    """
    target = 600
    state = make_state(sample_id="x", tools=[dummy_tool], metadata={"target_tokens": target})
    solver = context_exhaustion()
    await solver(state, noop_generate)
    total = count_tools_tokens(state.tools, encoding)
    cat_a_count = sum(1 for s in FIXTURE_SCHEMAS if s["content_category"] == "A_general_popular")
    n_added = len(state.tools) - 1
    if n_added < cat_a_count:
        assert total >= target
    else:
        assert n_added == cat_a_count


async def test_depth_path_falls_through_when_no_target_tokens(make_state, dummy_tool, noop_generate):
    """depth_schedule set but sample lacks target_tokens → uses pool_size fallback."""
    state = make_state(tools=[dummy_tool])
    solver = context_exhaustion(pool_size=2)
    await solver(state, noop_generate)
    assert len(state.tools) == 1 + 2


async def test_domain_filter_restricts_pool(make_state, dummy_tool, noop_generate):
    state = make_state(tools=[dummy_tool])
    solver = context_exhaustion(depth_schedule=None, pool_size=10, domain_filter=["cloud-ops"])
    await solver(state, noop_generate)
    cloud_names = {s["name"] for s in FIXTURE_SCHEMAS if s["domain"] == "cloud-ops"}
    injected = [ToolDef(t).name for t in state.tools[1:]]
    assert set(injected).issubset(cloud_names)
    assert len(injected) == len(cloud_names)


async def test_content_category_B_isolates_vacuous_controls(make_state, dummy_tool, noop_generate):
    """ICP-7 Gamage-vs-Levy isolation arm: only Category B schemas get sampled."""
    state = make_state(tools=[dummy_tool])
    solver = context_exhaustion(
        depth_schedule=None, pool_size=10, content_category="B_vacuous_controls"
    )
    await solver(state, noop_generate)
    cat_b_names = {s["name"] for s in FIXTURE_SCHEMAS if s["content_category"] == "B_vacuous_controls"}
    injected = [ToolDef(t).name for t in state.tools[1:]]
    assert set(injected) == cat_b_names


async def test_exclude_names_blocks_collisions(make_state, dummy_tool, noop_generate):
    blocked = "github_create_pull_request"
    state = make_state(tools=[dummy_tool])
    solver = context_exhaustion(
        depth_schedule=None,
        pool_size=10,
        exclude_names=[blocked],
    )
    await solver(state, noop_generate)
    injected = [ToolDef(t).name for t in state.tools[1:]]
    assert blocked not in injected


async def test_extend_with_appends_user_schemas(make_state, dummy_tool, noop_generate):
    user_schema = {
        "name": "user_provided_widget",
        "description": "A user-supplied filler schema for testing extend_with.",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "string", "description": "x"}},
            "required": ["x"],
        },
        "domain": "misc",
        "content_category": "A_general_popular",
        "mcp_server": "test/user",
    }
    state = make_state(tools=[dummy_tool])
    solver = context_exhaustion(
        depth_schedule=None,
        pool_size=20,
        extend_with=[user_schema],
    )
    await solver(state, noop_generate)
    injected = [ToolDef(t).name for t in state.tools[1:]]
    assert "user_provided_widget" in injected


async def test_composition_spec_exclude_keywords(make_state, dummy_tool, noop_generate):
    state = make_state(tools=[dummy_tool])
    solver = context_exhaustion(
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
    state = make_state(tools=[dummy_tool], metadata={"target_tokens": 2_000})
    initial_id = id(state.tools[0])
    initial_messages = list(state.messages)
    solver = context_exhaustion()
    out_state = await solver(state, noop_generate)
    assert id(out_state.tools[0]) == initial_id
    assert ToolDef(out_state.tools[0]).name == "eval_owned_tool"
    assert len(out_state.tools) > 1
    assert list(out_state.messages) == initial_messages
