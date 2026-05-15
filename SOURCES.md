# SOURCES — `inspect_tools` corpus attribution

## What this corpus is

`inspect_tools/data/tool_schemas_v1.json` is a curated corpus of MCP-style tool
schemas used as stimulus filler by the `context_exhaustion` Solver. The corpus
exists to measure how Inspect-eval performance degrades when a model's `tools`
parameter is saturated with realistic-looking tool schemas at varied context
depths.

## Source

All schemas were scraped from the **Smithery Registry API**
(`https://registry.smithery.ai`) on **2026-05-13** via
`scripts/scrape_tool_schemas.py`. Smithery is a public discovery registry for
MCP servers.

- **Scrape endpoint**: `GET /servers?q={query}&pageSize=50` (listing) and
  `GET /servers/{qualifiedName}` (detail).
- **Auth**: free bearer token from `smithery.ai/account/api-keys`.
- **Selection**: per-domain vendor allowlist (~280 names across 7 domains)
  plus top-N popular by `useCount` per domain, filtered to `verified=true`
  servers. 248 vendor MCPs resolved → 1639 tool records pre-curation.
- **Curation**: aggressive automated quality filters (Pydantic schema
  validation, JSON Schema check, placeholder/spam/dead-marker detection, token
  bounds) reduced to 1066 candidates; agentic + human chat curation across 7
  per-domain batches, followed by a 5-axis rubric audit (vendor authenticity /
  description naturalness / schema realism / deployment realism / corpus
  contribution) and per-MCP REVIEW pass for borderline cases reduced the
  corpus to **1,239 final tool schemas across 173 unique MCP vendors** with
  cross-domain reclassifications applied.

## Per-record attribution

Each record retains `source_url` pointing to the originating Smithery server
page (e.g., `https://smithery.ai/server/github`). This preserves attribution
to the original MCP publisher.

## License and use

Tool schemas (name + description + JSON Schema parameter spec) are functional
API metadata, not creative works (cf. *Oracle v. Google* fair-use weighting on
API declarations). Schemas published to Smithery's public discovery registry
imply redistribution permission via publication. Smithery's own SDK is
Apache-2.0-licensed (`github.com/smithery-ai/typescript-api`), signaling
intent for broad-ecosystem use of the registry.

This corpus is used as **transformative non-commercial research stimuli** for
context-exhaustion measurement — schemas are passed to the model in the
`tools` parameter but are never invoked. No per-record license tracking; the
attribution model is per-MCP via `source_url`.

## Composition

| Domain | Tool records |
|---|---|
| cloud-ops | 50 |
| dev-tools | 105 |
| data-analytics | 89 |
| communication | 79 |
| search | 44 |
| productivity | 216 |
| misc | 656 |
| **total** | **1,239** |

173 unique MCP vendors, average ~7 tools per MCP (per-MCP cap of 10 enforced
at scrape time).

All records carry `content_category: "general_popular"`. Future v1.x corpora
may widen `content_category` to include `injection` and `tool_shadowing`
variants.

## Refresh policy

The corpus is **version-pinned** to `tool_schemas_v1`. The scrape script is
idempotent (cache-respecting) but the upstream registry changes over time; a
v1.1+ refresh will re-run the script and produce a new pinned snapshot
(`tool_schemas_v1_1.json`).

## Reproducibility

The shipped JSON snapshot in git is the reproducibility artifact. The scrape
script is a maintainer tool, not a runtime dependency. Re-running the script
on a different date will produce different candidates because upstream
registry state changes; that's expected.
