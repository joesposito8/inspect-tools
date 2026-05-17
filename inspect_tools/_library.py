"""Corpus loader + pool filtering."""
from __future__ import annotations

import hashlib
import json
from importlib.resources import files

from inspect_tools.schema import ToolSchema

CORPUS_VERSION = "v1"
_CORPUS_RESOURCE = files("inspect_tools.data") / "tool_schemas_v1.json"
_CORPUS_CACHE: list[ToolSchema] | None = None
_CORPUS_SHA_CACHE: str | None = None


def load_corpus() -> list[ToolSchema]:
    """Load + Pydantic-validate the shipped corpus snapshot. Cached after first call."""
    global _CORPUS_CACHE
    if _CORPUS_CACHE is None:
        records = json.loads(_CORPUS_RESOURCE.read_text())
        _CORPUS_CACHE = [ToolSchema.model_validate(r) for r in records]
    return _CORPUS_CACHE


def corpus_sha() -> str:
    """16-char prefix of SHA-256 of the corpus file. For ICP-6's manifest."""
    global _CORPUS_SHA_CACHE
    if _CORPUS_SHA_CACHE is None:
        _CORPUS_SHA_CACHE = hashlib.sha256(_CORPUS_RESOURCE.read_bytes()).hexdigest()[:16]
    return _CORPUS_SHA_CACHE


def filter_pool(
    library: list[ToolSchema],
    *,
    content_category: list[str] | None = None,
    domain_filter: list[str] | None = None,
    exclude_names: list[str] | None = None,
    extend_with: list[ToolSchema] | None = None,
) -> list[ToolSchema]:
    """Filter order: content_category → domain → exclude_names → append extend_with.

    extend_with items are also subject to content_category and exclude_names.
    content_category=None skips the category filter. context_exhaustion passes
    ["general_popular"]; v1.x sibling solvers pass ["injection"] or ["tool_shadowing"].
    """
    cat_set = set(content_category) if content_category is not None else None
    excl_set = set(exclude_names) if exclude_names else set()
    pool = list(library)
    if cat_set is not None:
        pool = [s for s in pool if s.content_category in cat_set]
    if domain_filter is not None:
        domain_set = set(domain_filter)
        pool = [s for s in pool if s.domain in domain_set]
    if excl_set:
        pool = [s for s in pool if s.name not in excl_set]
    if extend_with:
        for s in extend_with:
            if cat_set is not None and s.content_category not in cat_set:
                continue
            if s.name in excl_set:
                continue
            pool.append(s)
    return pool
