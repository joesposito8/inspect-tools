"""outputSchema package selector + brace-aware placeholder substitution.

Each schema's `outputSchema.examples` carries 1-3 hand-authored response packages.
Strings inside packages contain `{kwarg | default}` placeholders.
`synthesize_response` picks one package via the supplied RNG and runs
substitution against agent kwargs.
"""
from __future__ import annotations

import json
import random
from typing import Any, Iterator

from inspect_tools.schema import ToolSchema


def _parse_default(s: str) -> Any:
    """Parse default as JSON literal; fall back to raw string if not JSON-valid."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


def _scan_placeholders(s: str) -> Iterator[tuple[int, int, str, str]]:
    """Yield (start, end, key, default) for each balanced `{key | default}` placeholder.

    Depth-counting handles braces inside defaults (`{tags | []}`, `{cfg | {"a": 1}}`).
    Skips bare `{key}` (no pipe) and malformed unclosed `{`.
    """
    i = 0
    while i < len(s):
        if s[i] == "{":
            depth, j = 1, i + 1
            while j < len(s) and depth > 0:
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                content = s[i + 1 : j - 1]
                pipe = content.find("|")
                if pipe > 0:
                    key = content[:pipe].strip()
                    default = content[pipe + 1 :].strip()
                    if key.isidentifier():
                        yield i, j, key, default
                        i = j
                        continue
        i += 1


def substitute(node: Any, kwargs: dict) -> Any:
    """Recursively apply `{key | default}` substitution.

    - Whole-value: string is exactly one full-coverage placeholder → type preserved
      (int, bool, list, dict, null).
    - Embedded: placeholders inside longer strings (or multiple per string) →
      str-coerced interpolation woven into the original text.
    """
    if isinstance(node, str):
        spans = list(_scan_placeholders(node))
        if len(spans) == 1 and spans[0][0] == 0 and spans[0][1] == len(node):
            _, _, key, default = spans[0]
            return kwargs[key] if key in kwargs else _parse_default(default)
        if not spans:
            return node
        parts: list[str] = []
        last = 0
        for start, end, key, default in spans:
            parts.append(node[last:start])
            parts.append(str(kwargs[key]) if key in kwargs else default)
            last = end
        parts.append(node[last:])
        return "".join(parts)
    if isinstance(node, dict):
        return {k: substitute(v, kwargs) for k, v in node.items()}
    if isinstance(node, list):
        return [substitute(x, kwargs) for x in node]
    return node


def synthesize_response(
    schema: ToolSchema, kwargs: dict, rng: random.Random
) -> dict | str:
    """Pick a response package from schema.outputSchema.examples; substitute kwargs.

    Fallback: outputSchema missing or examples empty → return {"ok": True}.
    """
    pkgs = (schema.outputSchema or {}).get("examples") or []
    if not pkgs:
        return {"ok": True}
    package = rng.choice(pkgs)
    return substitute(package, kwargs)
