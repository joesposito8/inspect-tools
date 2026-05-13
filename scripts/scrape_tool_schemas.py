#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcp>=1.27",
#   "pydantic>=2.0",
#   "httpx>=0.27",
#   "jsonschema>=4.0",
#   "langdetect>=1.0.9",
#   "tiktoken>=0.5",
# ]
# ///
"""Scrape MCP tool schemas from Smithery's Registry API.

Pipeline (per plan): list -> filter listings -> sample per domain ->
fetch details -> normalize to ToolSchema -> aggressive filter ->
dedupe -> categorize -> emit candidates.

Requires SMITHERY_API_KEY env var. Outputs candidates JSON for chat curation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import langdetect
import tiktoken
from langdetect import DetectorFactory

# Make ToolSchema importable from the package alongside this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspect_tools.schema import Domain, ToolSchema  # noqa: E402

# langdetect determinism
DetectorFactory.seed = 42

REGISTRY = "https://registry.smithery.ai"
CACHE_ROOT = Path.home() / ".cache" / "inspect_tools" / "scrape" / "smithery"
OUT_PATH = Path(__file__).resolve().parents[1] / "inspect_tools" / "data" / "tool_schemas_v1.candidates.json"
POLITE_DELAY_S = 1.0
LIST_PAGE_SIZE = 50
PER_DOMAIN_DETAIL_CAP = 5  # top N canonical servers per domain to fetch detail for

# Per-domain search queries. The registry caps default-listing results at 500
# (sorted by popularity), so domain-niche MCPs like AWS/K8s/Terraform fall
# outside that window. We query per-domain to surface them.
DOMAIN_QUERIES: dict[str, list[str]] = {
    "cloud-ops": ["aws", "kubernetes", "terraform", "docker", "azure", "gcp"],
    "dev-tools": ["github", "git", "gitlab", "npm", "vscode"],
    "data-analytics": ["postgres", "bigquery", "snowflake", "clickhouse", "sql"],
    "communication": ["slack", "discord", "email", "gmail", "twilio"],
    "search": ["search", "browser", "scrape", "puppeteer"],
    "productivity": ["notion", "jira", "calendar", "linear", "trello"],
}

# --- Domain keyword rules ---------------------------------------------------

DOMAIN_KEYWORDS: dict[Domain, list[str]] = {
    "cloud-ops": ["aws", "azure", "gcp", "kubernetes", "k8s", "docker", "terraform",
                  "deploy", "provision", "ec2", "rds", "container", "pod", "helm",
                  "cloudwatch", "lambda", "s3", "iam"],
    "dev-tools": ["git", "github", "gitlab", "npm", "pypi", "lint", "test", "build",
                  "ci", "codespaces", "vscode", "linter", "compile"],
    "data-analytics": ["sql", "postgres", "mysql", "bigquery", "snowflake", "pandas",
                       "query", "analytics", "csv", "parquet", "clickhouse",
                       "redshift", "duckdb", "supabase", "neon"],
    "communication": ["slack", "discord", "gmail", "email", "sms", "twilio", "webhook",
                      "send_message", "telegram", "whatsapp", "messaging"],
    "search": ["search", "scrape", "crawl", "browser", "web", "fetch", "puppeteer",
               "playwright", "exa", "brave", "perplexity"],
    "productivity": ["calendar", "gdrive", "drive", "notion", "jira", "asana",
                     "trello", "linear", "sheets", "docs", "todoist", "obsidian",
                     "confluence"],
}

# --- Aggressive filter patterns ---------------------------------------------

PLACEHOLDER_DESC_RE = re.compile(
    r"^(test|todo|fixme|wip|example|demo|placeholder|description)$", re.IGNORECASE
)
PLACEHOLDER_NAME_RE = re.compile(
    r"^(tool|test|example|demo|foo|bar|baz|new_tool|untitled|temp)\d*$", re.IGNORECASE
)
SPAM_PHRASES = (
    "best ", "ultimate", "premium service", "powered by", "subscribe",
    "free trial", "buy now",
)
DEAD_MARKERS = ("deprecated", "removed", "no longer supported", "this tool does nothing")

# --- Tokenizer ---

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(obj: Any) -> int:
    return len(_enc.encode(json.dumps(obj, sort_keys=True)))


# --- Logging ----------------------------------------------------------------

logger = logging.getLogger("scrape")


def setup_logging(quiet: bool) -> None:
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )


# --- HTTP client + cache ----------------------------------------------------


def make_client(api_key: str) -> httpx.Client:
    return httpx.Client(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(30.0),
    )


def _cache_path(subdir: str, key: str) -> Path:
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    p = CACHE_ROOT / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{h}.json"


def cached_get(client: httpx.Client, url: str, subdir: str) -> dict | None:
    """GET with on-disk cache. Returns parsed JSON or None on failure."""
    cache_file = _cache_path(subdir, url)
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            cache_file.unlink()
    time.sleep(POLITE_DELAY_S)
    try:
        r = client.get(url)
    except httpx.HTTPError as e:
        logger.info(f"http_error url={url} err={e}")
        return None
    if r.status_code != 200:
        logger.info(f"http_status url={url} status={r.status_code}")
        return None
    try:
        data = r.json()
    except json.JSONDecodeError:
        logger.info(f"json_decode_fail url={url}")
        return None
    cache_file.write_text(json.dumps(data))
    return data


# --- Phase 1: List ----------------------------------------------------------


def fetch_query_listings(client: httpx.Client, query: str, max_pages: int = 3) -> list[dict]:
    """Paginate one search query, returning up to LIST_PAGE_SIZE * max_pages results."""
    listings: list[dict] = []
    for page in range(1, max_pages + 1):
        url = f"{REGISTRY}/servers?pageSize={LIST_PAGE_SIZE}&page={page}&q={query}"
        data = cached_get(client, url, "listing")
        if not data or "servers" not in data:
            break
        servers = data["servers"]
        listings.extend(servers)
        pag = data.get("pagination", {})
        total_pages = pag.get("totalPages", 1)
        if page >= total_pages or not servers:
            break
    return listings


def fetch_all_listings(client: httpx.Client) -> list[dict]:
    """Query per-domain to surface canonical servers across all 7 domains.

    The registry's default listing caps at 500 results sorted by popularity, which
    over-represents search/communication MCPs and starves cloud-ops/data-analytics.
    Per-domain keyword queries are how we cover all 7 domain quotas.
    """
    seen_qn: set[str] = set()
    all_listings: list[dict] = []
    for domain, queries in DOMAIN_QUERIES.items():
        domain_total = 0
        for q in queries:
            servers = fetch_query_listings(client, q)
            new_this_query = 0
            for s in servers:
                qn = s.get("qualifiedName")
                if not qn or qn in seen_qn:
                    continue
                seen_qn.add(qn)
                all_listings.append(s)
                new_this_query += 1
                domain_total += 1
            logger.info(f"listing_query domain={domain} q={q} new={new_this_query} cumulative_unique={len(all_listings)}")
        logger.info(f"listing_domain_total domain={domain} new_unique={domain_total}")
    return all_listings


# --- Phase 2: Filter listings ----------------------------------------------


def is_canonical(server: dict) -> bool:
    return bool(server.get("verified")) and (server.get("slug") or "") == ""


# --- Phase 3: Sample per domain --------------------------------------------


def bucket_by_domain(server: dict) -> Domain:
    """Domain-keyword match against displayName + description (best-effort)."""
    haystack = f"{server.get('displayName', '')} {server.get('description', '')}".lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in haystack:
                return domain
    return "misc"


def sample_per_domain(canonical: list[dict], per_domain: int) -> list[dict]:
    """Bucket canonical servers by keyword match, take top N by useCount per bucket."""
    buckets: dict[Domain, list[dict]] = defaultdict(list)
    for srv in canonical:
        buckets[bucket_by_domain(srv)].append(srv)

    sampled: list[dict] = []
    for domain, servers in buckets.items():
        servers.sort(key=lambda s: s.get("useCount") or 0, reverse=True)
        chosen = servers[:per_domain]
        logger.info(f"sample_per_domain domain={domain} bucket_size={len(servers)} chosen={len(chosen)}")
        sampled.extend(chosen)

    # Dedup by qualifiedName (same server can match multiple domains keyword-wise — first wins)
    seen: set[str] = set()
    deduped: list[dict] = []
    for s in sampled:
        qn = s["qualifiedName"]
        if qn in seen:
            continue
        seen.add(qn)
        deduped.append(s)
    return deduped


# --- Phase 4: Fetch details ------------------------------------------------


def fetch_detail(client: httpx.Client, qualified_name: str) -> dict | None:
    url = f"{REGISTRY}/servers/{qualified_name}"
    return cached_get(client, url, "detail")


# --- Phase 5: Normalize (build ToolSchema) ---------------------------------


def normalize_one(server: dict, tool_dict: dict) -> ToolSchema | None:
    """Construct a ToolSchema from one tool entry in a server's detail response."""
    name = tool_dict.get("name")
    description = tool_dict.get("description") or ""
    input_schema = tool_dict.get("inputSchema")
    if not name or not isinstance(input_schema, dict):
        return None
    try:
        return ToolSchema(
            name=name,
            description=description,
            inputSchema=input_schema,
            source_url=f"https://smithery.ai/server/{server['qualifiedName']}",
        )
    except Exception as e:
        logger.info(f"drop reason=normalize_fail name={name} err={e}")
        return None


# --- Phase 6: Aggressive filters -------------------------------------------


def filter_malformed_schema(ts: ToolSchema) -> str | None:
    try:
        jsonschema.Draft7Validator.check_schema(ts.inputSchema)
    except jsonschema.SchemaError as e:
        return f"malformed_schema:{e.message[:60]}"
    if not isinstance(ts.inputSchema, dict):
        return "malformed_schema:not_dict"
    if ts.inputSchema.get("type") != "object":
        return "malformed_schema:type_not_object"
    if "properties" not in ts.inputSchema:
        return "malformed_schema:no_properties"
    return None


def filter_empty_or_placeholder_desc(ts: ToolSchema) -> str | None:
    desc = ts.description or ""
    if not desc or len(desc) < 20:
        return "desc_too_short"
    if PLACEHOLDER_DESC_RE.match(desc.strip()):
        return "desc_placeholder"
    return None


def filter_placeholder_name(ts: ToolSchema) -> str | None:
    if PLACEHOLDER_NAME_RE.match(ts.name):
        return "name_placeholder"
    return None


def filter_non_english(ts: ToolSchema) -> str | None:
    desc = ts.description or ""
    if not desc:
        return None
    non_ascii = sum(1 for c in desc if ord(c) > 127)
    if len(desc) and non_ascii / len(desc) > 0.3:
        return "non_english:ascii_ratio"
    try:
        if langdetect.detect(desc) != "en":
            return "non_english:langdetect"
    except langdetect.lang_detect_exception.LangDetectException:
        return "non_english:detect_fail"
    return None


def filter_spam(ts: ToolSchema) -> str | None:
    desc = (ts.description or "").lower()
    for phrase in SPAM_PHRASES:
        if phrase in desc:
            return f"spam:{phrase.strip()}"
    # crude emoji density
    emoji_count = sum(1 for c in desc if ord(c) > 0x1F000)
    if desc and emoji_count / len(desc) > 0.01:
        return "spam:emoji_density"
    url_count = desc.count("http://") + desc.count("https://")
    if url_count > 1:
        return "spam:url_count"
    return None


def filter_token_bounds(ts: ToolSchema) -> str | None:
    rendered = {
        "name": ts.name,
        "description": ts.description,
        "inputSchema": ts.inputSchema,
    }
    n = count_tokens(rendered)
    if n < 30:
        return f"token_bounds:too_short_{n}"
    if n > 2000:
        return f"token_bounds:too_long_{n}"
    return None


def filter_dead_markers(ts: ToolSchema) -> str | None:
    desc = (ts.description or "").lower()
    for marker in DEAD_MARKERS:
        if marker in desc:
            return f"dead_marker:{marker}"
    return None


HARD_FILTERS = [
    filter_malformed_schema,
    filter_empty_or_placeholder_desc,
    filter_placeholder_name,
    filter_non_english,
    filter_spam,
    filter_token_bounds,
    filter_dead_markers,
]


def schema_hash(ts: ToolSchema) -> str:
    canonical = json.dumps(
        {"name": ts.name, "inputSchema": ts.inputSchema}, sort_keys=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


# --- Phase 8: Categorize ---------------------------------------------------


def categorize(ts: ToolSchema) -> ToolSchema:
    """Assign a domain based on keyword match against name + description."""
    haystack = f"{ts.name} {ts.description or ''}".lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in haystack:
                ts.domain = domain
                return ts
    ts.domain = "misc"
    return ts


# --- Phase 9: Emit ----------------------------------------------------------


def emit_candidates(records: list[ToolSchema], path: Path) -> dict[str, int]:
    """Write candidates JSON and return per-domain counts."""
    records_sorted = sorted(records, key=lambda r: (r.domain, r.name, r.source_url))
    serialized = [r.model_dump(exclude_none=True, mode="json") for r in records_sorted]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialized, indent=2, sort_keys=False))
    counts: dict[str, int] = defaultdict(int)
    for r in records_sorted:
        counts[r.domain] += 1
    return dict(counts)


# --- Main -------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true", help="suppress per-record logs")
    ap.add_argument(
        "--per-domain",
        type=int,
        default=PER_DOMAIN_DETAIL_CAP,
        help=f"max servers per domain to fetch detail for (default: {PER_DOMAIN_DETAIL_CAP})",
    )
    args = ap.parse_args()
    setup_logging(args.quiet)

    api_key = os.environ.get("SMITHERY_API_KEY")
    if not api_key:
        print(
            "ERROR: SMITHERY_API_KEY env var required. "
            "Get a free key at https://smithery.ai/account/api-keys",
            file=sys.stderr,
        )
        return 2

    client = make_client(api_key)

    # 1. List
    listings = fetch_all_listings(client)
    print(f"[1/9] fetched {len(listings)} listings", file=sys.stderr)

    # 2. Filter listings
    canonical = [s for s in listings if is_canonical(s)]
    print(
        f"[2/9] canonical (verified=true, slug=''): {len(canonical)} / {len(listings)}",
        file=sys.stderr,
    )

    # 3. Sample per domain
    sampled = sample_per_domain(canonical, args.per_domain)
    print(f"[3/9] sampled {len(sampled)} servers for detail fetch", file=sys.stderr)

    # 4. Fetch details
    details: list[tuple[dict, dict]] = []
    for i, srv in enumerate(sampled, 1):
        d = fetch_detail(client, srv["qualifiedName"])
        if d:
            details.append((srv, d))
        if i % 10 == 0:
            print(f"[4/9] fetched {i}/{len(sampled)} details", file=sys.stderr)
    print(f"[4/9] fetched {len(details)} details total", file=sys.stderr)

    # 5. Normalize
    candidates: list[ToolSchema] = []
    for srv, detail in details:
        for tool_dict in detail.get("tools", []) or []:
            ts = normalize_one(srv, tool_dict)
            if ts is not None:
                candidates.append(ts)
    print(f"[5/9] normalized to {len(candidates)} ToolSchema records", file=sys.stderr)

    # 6. Aggressive filter
    drop_counts: dict[str, int] = defaultdict(int)
    kept: list[ToolSchema] = []
    for ts in candidates:
        drop_reason: str | None = None
        for f in HARD_FILTERS:
            drop_reason = f(ts)
            if drop_reason:
                break
        if drop_reason:
            drop_counts[drop_reason.split(":")[0]] += 1
            logger.info(f"drop reason={drop_reason} name={ts.name} source={ts.source_url}")
            continue
        kept.append(ts)
    print(f"[6/9] kept {len(kept)} after aggressive filters", file=sys.stderr)
    for reason, n in sorted(drop_counts.items(), key=lambda kv: -kv[1]):
        print(f"        drop {reason}: {n}", file=sys.stderr)

    # 7. Deduplicate
    seen_hashes: set[str] = set()
    deduped: list[ToolSchema] = []
    for ts in kept:
        h = schema_hash(ts)
        if h in seen_hashes:
            logger.info(f"drop reason=duplicate name={ts.name}")
            continue
        seen_hashes.add(h)
        deduped.append(ts)
    print(f"[7/9] deduped to {len(deduped)} unique records", file=sys.stderr)

    # 8. Categorize
    categorized = [categorize(ts) for ts in deduped]
    print("[8/9] categorized", file=sys.stderr)

    # 9. Emit
    counts = emit_candidates(categorized, OUT_PATH)
    print(f"[9/9] wrote {OUT_PATH.relative_to(Path.cwd()) if OUT_PATH.is_relative_to(Path.cwd()) else OUT_PATH}", file=sys.stderr)
    print(f"        total: {sum(counts.values())} candidates", file=sys.stderr)
    for domain in [
        "cloud-ops", "dev-tools", "data-analytics", "communication",
        "search", "productivity", "misc",
    ]:
        print(f"        {domain}: {counts.get(domain, 0)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
