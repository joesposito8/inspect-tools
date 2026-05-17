"""Tests for inspect_tools._library."""
from __future__ import annotations

import pytest

from inspect_tools._fixtures import FIXTURE_SCHEMAS
from inspect_tools._library import corpus_sha, filter_pool, load_corpus
from inspect_tools.schema import ToolSchema


def test_load_corpus_returns_non_empty_list():
    corpus = load_corpus()
    assert isinstance(corpus, list)
    assert len(corpus) > 0
    assert all(isinstance(s, ToolSchema) for s in corpus)


def test_load_corpus_cached():
    a = load_corpus()
    b = load_corpus()
    assert a is b  # same object on second call


def test_corpus_sha_format():
    sha = corpus_sha()
    assert isinstance(sha, str)
    assert len(sha) == 16
    assert all(c in "0123456789abcdef" for c in sha)


def test_corpus_sha_stable():
    assert corpus_sha() == corpus_sha()


def test_filter_pool_content_category_filters_correctly():
    filtered = filter_pool(FIXTURE_SCHEMAS, content_category=["general_popular"])
    assert all(s.content_category == "general_popular" for s in filtered)
    assert len(filtered) == len(FIXTURE_SCHEMAS)  # all fixtures are general_popular


def test_filter_pool_content_category_none_returns_all():
    filtered = filter_pool(FIXTURE_SCHEMAS, content_category=None)
    assert len(filtered) == len(FIXTURE_SCHEMAS)


def test_filter_pool_non_existent_category_returns_empty():
    filtered = filter_pool(FIXTURE_SCHEMAS, content_category=["injection"])
    assert filtered == []


def test_filter_pool_domain_filter():
    filtered = filter_pool(FIXTURE_SCHEMAS, domain_filter=["cloud-ops"])
    assert all(s.domain == "cloud-ops" for s in filtered)
    assert len(filtered) >= 1


def test_filter_pool_multiple_domains():
    filtered = filter_pool(FIXTURE_SCHEMAS, domain_filter=["cloud-ops", "search"])
    domains = {s.domain for s in filtered}
    assert domains.issubset({"cloud-ops", "search"})


def test_filter_pool_exclude_names():
    target = FIXTURE_SCHEMAS[0].name
    filtered = filter_pool(FIXTURE_SCHEMAS, exclude_names=[target])
    assert all(s.name != target for s in filtered)
    assert len(filtered) == len(FIXTURE_SCHEMAS) - 1


def test_filter_pool_extend_with_appends():
    extra = ToolSchema(
        name="extra_tool",
        description="A user-supplied tool.",
        inputSchema={"type": "object", "properties": {}, "required": []},
        domain="misc",
        content_category="general_popular",
        source_url="https://test.fixture/extra",
    )
    filtered = filter_pool(FIXTURE_SCHEMAS, extend_with=[extra])
    names = [s.name for s in filtered]
    assert "extra_tool" in names
    assert len(filtered) == len(FIXTURE_SCHEMAS) + 1


def test_filter_pool_extend_with_respects_content_category():
    extra = ToolSchema(
        name="extra_injection_tool",
        description="A user-supplied tool in a different category.",
        inputSchema={"type": "object", "properties": {}, "required": []},
        domain="misc",
        content_category="general_popular",  # Literal only allows this in v1.0
        source_url="https://test.fixture/extra",
    )
    # When filtering to a non-matching category, extra is dropped along with library
    filtered = filter_pool(
        FIXTURE_SCHEMAS, content_category=["nonexistent_category"], extend_with=[extra]
    )
    assert filtered == []


def test_filter_pool_extend_with_respects_exclude_names():
    extra = ToolSchema(
        name="excluded_extra",
        description="A user-supplied tool we then exclude.",
        inputSchema={"type": "object", "properties": {}, "required": []},
        domain="misc",
        content_category="general_popular",
        source_url="https://test.fixture/extra",
    )
    filtered = filter_pool(
        FIXTURE_SCHEMAS, exclude_names=["excluded_extra"], extend_with=[extra]
    )
    assert all(s.name != "excluded_extra" for s in filtered)


def test_filter_pool_does_not_mutate_input():
    original_len = len(FIXTURE_SCHEMAS)
    filter_pool(FIXTURE_SCHEMAS, domain_filter=["cloud-ops"])
    assert len(FIXTURE_SCHEMAS) == original_len
