from inspect_tools._fixtures import FIXTURE_SCHEMAS
from inspect_tools._types import ToolSchema


def load_fixture_library() -> list[ToolSchema]:
    """Return the inline ICP-3 fixture corpus.

    ICP-4 ships the scraped library that replaces this. Callers should not
    mutate the returned list.
    """
    return list(FIXTURE_SCHEMAS)


def filter_pool(
    library: list[ToolSchema],
    *,
    domain_filter: list[str] | None = None,
    content_category: str = "A_general_popular",
    exclude_names: list[str] | None = None,
    extend_with: list[ToolSchema] | None = None,
    composition_spec: dict | None = None,
) -> list[ToolSchema]:
    """Filter the schema pool by the ICP-3 kwargs and composition_spec.

    Filter order: content_category, domain, exclude_names, composition_spec
    (exclude_keywords + tool_categories), then append extend_with. Order matters
    only for performance; the result is set-equivalent regardless.
    """
    pool = [s for s in library if s["content_category"] == content_category]

    if domain_filter is not None:
        allowed = set(domain_filter)
        pool = [s for s in pool if s["domain"] in allowed]

    if exclude_names:
        blocked = set(exclude_names)
        pool = [s for s in pool if s["name"] not in blocked]

    if composition_spec:
        spec_categories = composition_spec.get("tool_categories")
        if spec_categories:
            allowed = set(spec_categories)
            pool = [s for s in pool if s["domain"] in allowed]

        exclude_keywords = composition_spec.get("exclude_keywords")
        if exclude_keywords:
            lowered = [kw.lower() for kw in exclude_keywords]
            pool = [
                s
                for s in pool
                if not any(kw in s["name"].lower() or kw in s["description"].lower() for kw in lowered)
            ]

    if extend_with:
        # Respect the active content_category filter for user-provided schemas too.
        for schema in extend_with:
            if schema["content_category"] == content_category:
                if exclude_names and schema["name"] in exclude_names:
                    continue
                pool.append(schema)

    return pool
