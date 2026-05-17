"""Tests for inspect_tools._solver.context_exhaustion."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from inspect_tools._fixtures import FIXTURE_SCHEMAS
from inspect_tools._solver import context_exhaustion


@pytest.fixture(autouse=True)
def _patch_corpus(monkeypatch):
    """Force load_corpus() to return our fixture corpus."""
    monkeypatch.setattr("inspect_tools._library._CORPUS_CACHE", FIXTURE_SCHEMAS)


# === Literal mode (target_tokens=None) ===


async def test_literal_mode_injects_n_filler(make_state, dummy_tool, noop_generate):
    solver = context_exhaustion(n_filler=3)
    state = make_state(tools=[dummy_tool])
    await solver(state, noop_generate)
    # 1 eval tool + 3 filler = 4
    assert len(state.tools) == 4


async def test_literal_mode_additive_preserves_eval_tool(make_state, dummy_tool, noop_generate):
    solver = context_exhaustion(n_filler=3)
    state = make_state(tools=[dummy_tool])
    await solver(state, noop_generate)
    tool_def_names = [t.__name__ if hasattr(t, "__name__") else None for t in state.tools]
    # The eval-owned tool's name should still be present somewhere in state.tools
    # (we identify it via the manifest's injected_tool_names)
    manifest = state.metadata["inspect_tools"]["context_exhaustion"]
    eval_tool_count = len(state.tools) - len(manifest["injected_tool_names"])
    assert eval_tool_count == 1


async def test_literal_mode_clamps_to_pool_size(make_state, noop_generate):
    solver = context_exhaustion(n_filler=999)  # more than 7 fixtures
    state = make_state()
    await solver(state, noop_generate)
    assert len(state.tools) == len(FIXTURE_SCHEMAS)


async def test_literal_mode_no_model_calls(make_state, noop_generate, mock_model):
    """target_tokens=None should never invoke the model."""
    solver = context_exhaustion(n_filler=3)
    state = make_state()
    with patch("inspect_tools._solver.get_model", return_value=mock_model):
        await solver(state, noop_generate)
    mock_model.count_tool_tokens.assert_not_called()


async def test_literal_mode_deterministic(make_state, noop_generate):
    solver1 = context_exhaustion(n_filler=3)
    solver2 = context_exhaustion(n_filler=3)
    state1 = make_state(sample_id="sample-a", epoch=1)
    state2 = make_state(sample_id="sample-a", epoch=1)
    await solver1(state1, noop_generate)
    await solver2(state2, noop_generate)
    names1 = state1.metadata["inspect_tools"]["context_exhaustion"]["injected_tool_names"]
    names2 = state2.metadata["inspect_tools"]["context_exhaustion"]["injected_tool_names"]
    assert names1 == names2


async def test_epoch_driven_variance(make_state, noop_generate):
    """Same sample, different epoch → at least one injected name differs."""
    solver = context_exhaustion(n_filler=3)
    state_e1 = make_state(sample_id="sample-a", epoch=1)
    state_e2 = make_state(sample_id="sample-a", epoch=2)
    await solver(state_e1, noop_generate)
    await solver(state_e2, noop_generate)
    names1 = set(state_e1.metadata["inspect_tools"]["context_exhaustion"]["injected_tool_names"])
    names2 = set(state_e2.metadata["inspect_tools"]["context_exhaustion"]["injected_tool_names"])
    assert names1 != names2


async def test_different_sample_id_gives_different_names(make_state, noop_generate):
    """Probabilistic: across 5 samples, at least one set should differ from sample-0."""
    solver = context_exhaustion(n_filler=3)
    states = [make_state(sample_id=f"sample-{i}") for i in range(5)]
    for s in states:
        await solver(s, noop_generate)
    name_sets = [
        tuple(s.metadata["inspect_tools"]["context_exhaustion"]["injected_tool_names"])
        for s in states
    ]
    assert len(set(name_sets)) >= 2


# === Depth-derived mode (target_tokens set) ===


async def test_bisect_standard(make_state, noop_generate, mock_model):
    """mock_model returns 100*len(tools); target=300 with no eval tools → n=3."""
    solver = context_exhaustion(target_tokens=300)
    state = make_state()
    with patch("inspect_tools._solver.get_model", return_value=mock_model):
        await solver(state, noop_generate)
    assert len(state.tools) == 3


async def test_bisect_overshoot(make_state, noop_generate, mock_model):
    """target=301 → smallest n where 100n >= 301 is n=4."""
    solver = context_exhaustion(target_tokens=301)
    state = make_state()
    with patch("inspect_tools._solver.get_model", return_value=mock_model):
        await solver(state, noop_generate)
    assert len(state.tools) == 4


async def test_bisect_pool_exhaustion(make_state, noop_generate, mock_model):
    """target larger than full-pool count (100 * len(FIXTURE_SCHEMAS) = 700) → all 7 injected."""
    solver = context_exhaustion(target_tokens=10_000)
    state = make_state()
    with patch("inspect_tools._solver.get_model", return_value=mock_model):
        await solver(state, noop_generate)
    assert len(state.tools) == len(FIXTURE_SCHEMAS)


async def test_bisect_already_over_baseline(make_state, dummy_tool_factory, noop_generate, mock_model):
    """When state.tools already exceeds target, no fillers injected."""
    # 10 eval tools = 1000 tokens; target = 500
    eval_tools = [dummy_tool_factory(f"eval_tool_{i}") for i in range(10)]
    solver = context_exhaustion(target_tokens=500)
    state = make_state(tools=eval_tools)
    with patch("inspect_tools._solver.get_model", return_value=mock_model):
        await solver(state, noop_generate)
    # Eval tools preserved; nothing injected
    assert len(state.tools) == 10
    manifest = state.metadata["inspect_tools"]["context_exhaustion"]
    assert manifest["injected_tool_names"] == []


# === Filter kwargs ===


async def test_domain_filter(make_state, noop_generate, mock_model):
    solver = context_exhaustion(target_tokens=10_000, domain_filter=["cloud-ops"])
    state = make_state()
    with patch("inspect_tools._solver.get_model", return_value=mock_model):
        await solver(state, noop_generate)
    injected = state.metadata["inspect_tools"]["context_exhaustion"]["injected_tool_names"]
    cloud_ops_names = {s.name for s in FIXTURE_SCHEMAS if s.domain == "cloud-ops"}
    assert set(injected).issubset(cloud_ops_names)


async def test_exclude_names(make_state, noop_generate):
    target_name = FIXTURE_SCHEMAS[0].name
    solver = context_exhaustion(n_filler=99, exclude_names=[target_name])
    state = make_state()
    await solver(state, noop_generate)
    injected = state.metadata["inspect_tools"]["context_exhaustion"]["injected_tool_names"]
    assert target_name not in injected


async def test_extend_with_user_schemas_appear(make_state, noop_generate):
    """Add 50 unique-named user schemas; with n_filler=99 and the 7 fixtures, expect ≥1 user schema."""
    from inspect_tools.schema import ToolSchema

    extras = [
        ToolSchema(
            name=f"user_extra_{i}",
            description=f"User-supplied tool {i}.",
            inputSchema={"type": "object", "properties": {}, "required": []},
            domain="misc",
            content_category="general_popular",
            source_url=f"https://test.fixture/user_{i}",
        )
        for i in range(50)
    ]
    solver = context_exhaustion(n_filler=99, extend_with=extras)
    state = make_state()
    await solver(state, noop_generate)
    injected = state.metadata["inspect_tools"]["context_exhaustion"]["injected_tool_names"]
    user_names = {f"user_extra_{i}" for i in range(50)}
    assert any(n in user_names for n in injected)


# === Manifest ===


async def test_manifest_all_keys_present(make_state, noop_generate):
    solver = context_exhaustion(n_filler=2)
    state = make_state()
    await solver(state, noop_generate)
    manifest = state.metadata["inspect_tools"]["context_exhaustion"]
    assert "injected_tool_names" in manifest
    assert "pool_filter" in manifest
    assert "library_seed_per_sample" in manifest
    assert "target_tokens" in manifest
    assert "actual_tokens" in manifest
    assert "corpus_sha" in manifest
    assert "invocations" in manifest


async def test_manifest_corpus_sha_matches_library(make_state, noop_generate):
    """Manifest records the corpus sha the trial was built against, so EvalLogs
    are tied to a specific corpus version and stay comparable across re-runs."""
    from inspect_tools._library import corpus_sha
    solver = context_exhaustion(n_filler=2)
    state = make_state()
    await solver(state, noop_generate)
    manifest = state.metadata["inspect_tools"]["context_exhaustion"]
    assert manifest["corpus_sha"] == corpus_sha()
    assert isinstance(manifest["corpus_sha"], str)
    assert len(manifest["corpus_sha"]) == 16


async def test_manifest_actual_tokens_none_in_literal_mode(make_state, noop_generate):
    """Literal mode does no token counting → actual_tokens stays None."""
    solver = context_exhaustion(n_filler=3)
    state = make_state()
    await solver(state, noop_generate)
    assert state.metadata["inspect_tools"]["context_exhaustion"]["actual_tokens"] is None


async def test_manifest_actual_tokens_reaches_target(make_state, noop_generate, mock_model):
    """Depth mode: actual_tokens should be >= target (overshoot allowed) when pool reachable."""
    solver = context_exhaustion(target_tokens=300)
    state = make_state()
    with patch("inspect_tools._solver.get_model", return_value=mock_model):
        await solver(state, noop_generate)
    actual = state.metadata["inspect_tools"]["context_exhaustion"]["actual_tokens"]
    assert actual == 300  # mock_model returns 100 * len(tools); n=3 → 300


async def test_manifest_actual_tokens_pool_exhausted(make_state, noop_generate, mock_model):
    """Pool exhausted before target → actual_tokens is the full-pool count."""
    solver = context_exhaustion(target_tokens=10_000)
    state = make_state()
    with patch("inspect_tools._solver.get_model", return_value=mock_model):
        await solver(state, noop_generate)
    actual = state.metadata["inspect_tools"]["context_exhaustion"]["actual_tokens"]
    # 7 fixtures × 100 = 700
    assert actual == 100 * len(state.tools)
    assert actual < 10_000  # confirms unreachable


async def test_manifest_pool_filter_reflects_kwargs(make_state, noop_generate):
    solver = context_exhaustion(
        n_filler=2,
        domain_filter=["cloud-ops"],
        exclude_names=["foo"],
    )
    state = make_state()
    await solver(state, noop_generate)
    pf = state.metadata["inspect_tools"]["context_exhaustion"]["pool_filter"]
    assert pf["n_filler"] == 2
    assert pf["content_category"] == ["general_popular"]
    assert pf["domain_filter"] == ["cloud-ops"]
    assert pf["exclude_names"] == ["foo"]
    assert pf["target_tokens"] is None
    assert pf["extend_with_names"] == []


async def test_manifest_target_tokens_captured(make_state, noop_generate, mock_model):
    solver = context_exhaustion(target_tokens=4_000)
    state = make_state()
    with patch("inspect_tools._solver.get_model", return_value=mock_model):
        await solver(state, noop_generate)
    assert state.metadata["inspect_tools"]["context_exhaustion"]["target_tokens"] == 4_000


# === Depth-axis prefix-nesting (V8) ===


async def test_depth_prefix_nesting(make_state, noop_generate, mock_model):
    """Same (sample_id, epoch), different target_tokens → smaller depth's injected
    set is a strict prefix of larger depth's. Enables clean attribution of score
    movement to 'tools added' rather than 'completely different tools'."""
    small = context_exhaustion(target_tokens=300)
    large = context_exhaustion(target_tokens=600)
    s_small = make_state(sample_id="X", epoch=1)
    s_large = make_state(sample_id="X", epoch=1)
    with patch("inspect_tools._solver.get_model", return_value=mock_model):
        await small(s_small, noop_generate)
        await large(s_large, noop_generate)
    names_small = s_small.metadata["inspect_tools"]["context_exhaustion"]["injected_tool_names"]
    names_large = s_large.metadata["inspect_tools"]["context_exhaustion"]["injected_tool_names"]
    assert len(names_small) < len(names_large)
    assert names_large[: len(names_small)] == names_small


async def test_depth_seed_independent_of_target_tokens(make_state, noop_generate, mock_model):
    """library_seed_per_sample is the SAME across depths for a fixed (sample_id, epoch)."""
    s_a = make_state(sample_id="X", epoch=1)
    s_b = make_state(sample_id="X", epoch=1)
    with patch("inspect_tools._solver.get_model", return_value=mock_model):
        await context_exhaustion(target_tokens=300)(s_a, noop_generate)
        await context_exhaustion(target_tokens=600)(s_b, noop_generate)
    assert (
        s_a.metadata["inspect_tools"]["context_exhaustion"]["library_seed_per_sample"]
        == s_b.metadata["inspect_tools"]["context_exhaustion"]["library_seed_per_sample"]
    )
