"""Tests for inspect_tools._manifest.RunManifestHook.on_task_start."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from inspect_ai.hooks import TaskStart

from inspect_tools import _solver
from inspect_tools._library import corpus_sha
from inspect_tools._manifest import RunManifestHook


def _make_spec(
    task: str = "stub_task",
    task_version: int | str = 0,
    model: str = "anthropic/claude-sonnet-4-6",
    task_args: dict | None = None,
    metadata: dict | None = None,
) -> SimpleNamespace:
    """Minimal EvalSpec stand-in. The hook only reads .task, .task_version,
    .model, .task_args, .metadata; SimpleNamespace satisfies those reads."""
    return SimpleNamespace(
        task=task,
        task_version=task_version,
        model=model,
        task_args=task_args if task_args is not None else {},
        metadata=metadata,
    )


def _make_event(spec) -> TaskStart:
    return TaskStart(
        eval_set_id=None,
        run_id="run-x",
        eval_id="eval-x",
        spec=spec,
    )


@pytest.fixture(autouse=True)
def _reset_flag(monkeypatch):
    """Snapshot and restore the module-level gate flag around each test."""
    original = _solver._CONTEXT_EXHAUSTION_ACTIVE
    yield
    _solver._CONTEXT_EXHAUSTION_ACTIVE = original


async def test_hook_writes_run_level_fields_when_active():
    _solver._CONTEXT_EXHAUSTION_ACTIVE = True
    spec = _make_spec(
        task="my_eval",
        task_version=3,
        model="anthropic/claude-haiku-4-5",
        task_args={"target_tokens": 64_000},
    )
    await RunManifestHook().on_task_start(_make_event(spec))
    m = spec.metadata["inspect_tools"]["context_exhaustion"]
    assert m["wrapped_task"] == "my_eval"
    assert m["wrapped_task_version"] == 3
    assert m["models_evaluated"] == ["anthropic/claude-haiku-4-5"]
    assert m["tokenizer_id"] == "anthropic/claude-haiku-4-5:count_tool_tokens"
    assert m["depth_schedule"] == [64_000]
    assert m["tool_corpus_version"] == corpus_sha()
    # package_version is best-effort; allow either an installed version or the fallback.
    assert isinstance(m["package_version"], str) and m["package_version"]


async def test_hook_skips_when_flag_unset():
    _solver._CONTEXT_EXHAUSTION_ACTIVE = False
    spec = _make_spec(metadata={"existing": "keep"})
    await RunManifestHook().on_task_start(_make_event(spec))
    # Untouched
    assert spec.metadata == {"existing": "keep"}


async def test_hook_depth_schedule_empty_without_target_tokens():
    _solver._CONTEXT_EXHAUSTION_ACTIVE = True
    spec = _make_spec(task_args={"n_filler": 5})
    await RunManifestHook().on_task_start(_make_event(spec))
    m = spec.metadata["inspect_tools"]["context_exhaustion"]
    assert m["depth_schedule"] == []


async def test_hook_initializes_none_metadata():
    _solver._CONTEXT_EXHAUSTION_ACTIVE = True
    spec = _make_spec(metadata=None)
    await RunManifestHook().on_task_start(_make_event(spec))
    assert spec.metadata is not None
    assert "inspect_tools" in spec.metadata


async def test_hook_preserves_existing_user_metadata():
    _solver._CONTEXT_EXHAUSTION_ACTIVE = True
    spec = _make_spec(metadata={"full_task_version": "2-A", "other": [1, 2]})
    await RunManifestHook().on_task_start(_make_event(spec))
    assert spec.metadata["full_task_version"] == "2-A"
    assert spec.metadata["other"] == [1, 2]
    assert "inspect_tools" in spec.metadata


async def test_hook_does_not_clobber_existing_inspect_tools_block():
    """If a sibling Solver (e.g. inject_description) wrote first, preserve it."""
    _solver._CONTEXT_EXHAUSTION_ACTIVE = True
    spec = _make_spec(
        metadata={"inspect_tools": {"inject_description": {"foo": "bar"}}}
    )
    await RunManifestHook().on_task_start(_make_event(spec))
    assert spec.metadata["inspect_tools"]["inject_description"] == {"foo": "bar"}
    assert "context_exhaustion" in spec.metadata["inspect_tools"]
