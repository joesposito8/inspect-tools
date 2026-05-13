from inspect_context_pressure._fixtures import FIXTURE_SCHEMAS
from inspect_context_pressure._inject import (
    FILLER_INVOCATION_KEY,
    count_schema_tokens,
    count_tools_tokens,
    inject_filler_tools,
    to_inspect_tool_def,
)


def test_to_inspect_tool_def_roundtrip():
    schema = FIXTURE_SCHEMAS[0]
    td = to_inspect_tool_def(schema)
    assert td.name == schema["name"]
    assert td.description == schema["description"]
    assert set(td.parameters.required) == set(schema["parameters"]["required"])
    assert set(td.parameters.properties.keys()) == set(schema["parameters"]["properties"].keys())


def test_to_inspect_tool_def_distinct_descriptions():
    """All filler tools share a callable shape but keep their distinct descriptions."""
    descs = {to_inspect_tool_def(s).description for s in FIXTURE_SCHEMAS[:5]}
    assert len(descs) == 5


def test_count_tools_tokens_monotone(make_state, dummy_tool_factory, encoding):
    state = make_state(tools=[dummy_tool_factory("alpha")])
    base = count_tools_tokens(state.tools, encoding)
    state.tools = list(state.tools) + [dummy_tool_factory("beta")]
    after = count_tools_tokens(state.tools, encoding)
    assert after > base


def test_count_schema_tokens_positive(encoding):
    for schema in FIXTURE_SCHEMAS:
        assert count_schema_tokens(schema, encoding) > 0


async def test_filler_noop_increments_counter(make_state):
    state = make_state()
    n_added = inject_filler_tools(state, FIXTURE_SCHEMAS[:3])
    assert n_added == 3
    assert state.metadata[FILLER_INVOCATION_KEY] == 0
    for tool in state.tools[-3:]:
        await tool()
    assert state.metadata[FILLER_INVOCATION_KEY] == 3


def test_inject_is_additive(make_state, dummy_tool):
    state = make_state(tools=[dummy_tool])
    initial_id = id(state.tools[0])
    inject_filler_tools(state, FIXTURE_SCHEMAS[:2])
    assert len(state.tools) == 3
    assert id(state.tools[0]) == initial_id


async def test_filler_returns_benign_string(make_state):
    state = make_state()
    inject_filler_tools(state, FIXTURE_SCHEMAS[:1])
    result = await state.tools[0]()
    assert "no-op" in result.lower()
