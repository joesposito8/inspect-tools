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
PER_DOMAIN_DETAIL_CAP = 35  # top N verified servers per domain bucket to fetch detail for
PER_MCP_TOOL_CAP = 10  # max tools per MCP server (caps slack/github bloat)

# Vendor allowlist per domain. Researched from awesome-mcp-servers lists +
# Smithery/mcp.so featured + 2026 popularity rankings. Each name is queried
# against Smithery; we pick the best verified match per name.
VENDOR_ALLOWLIST: dict[str, list[str]] = {
    "cloud-ops": [
        "aws", "aws-s3", "aws-cloudwatch", "aws-billing", "aws-cdk", "gcp", "azure",
        "azure-devops", "cloudflare", "wrangler", "vercel", "netlify", "heroku",
        "render", "fly-io", "railway", "digitalocean", "linode", "kubernetes", "helm",
        "argocd", "docker", "docker-hub", "terraform", "pulumi", "ansible", "jenkins",
        "circleci", "github-actions", "buildkite", "datadog", "sentry", "grafana",
        "prometheus", "pagerduty", "new-relic", "splunk", "opsgenie", "honeycomb",
        "portainer",
    ],
    "dev-tools": [
        "github", "gitlab", "bitbucket", "gitea", "git", "jetbrains", "vscode",
        "intellij", "npm", "pypi", "cargo", "maven", "nuget", "packagist",
        "sonarqube", "snyk", "semgrep", "codecov", "sourcegraph", "codacy", "eslint",
        "prettier", "pre-commit", "gradle", "pnpm", "yarn", "bazel", "nx",
        "turborepo", "copilot", "cursor", "ripgrep", "ast-grep", "changesets",
    ],
    "data-analytics": [
        "postgres", "postgresql", "mysql", "mongodb", "mongodb-atlas", "sqlite",
        "redis", "duckdb", "mssql", "oracle", "mariadb", "cockroachdb", "clickhouse",
        "snowflake", "bigquery", "redshift", "databricks", "supabase", "planetscale",
        "neon", "turso", "chroma", "pinecone", "weaviate", "qdrant", "milvus",
        "elasticsearch", "opensearch", "algolia", "meilisearch", "typesense", "dbt",
        "airbyte", "fivetran", "mixpanel", "amplitude", "segment", "posthog", "neo4j",
        "influxdb",
    ],
    "communication": [
        "slack", "discord", "mattermost", "microsoft-teams", "rocketchat", "telegram",
        "whatsapp", "signal", "gmail", "outlook", "sendgrid", "mailgun", "resend",
        "postmark", "twilio", "vonage", "bandwidth", "plivo", "messagebird",
        "zendesk", "intercom", "freshdesk", "helpscout", "front", "zoom",
        "google-meet", "webex", "agentmail", "nylas",
    ],
    "search": [
        "brave-search", "google-search", "bing-search", "exa", "perplexity", "tavily",
        "you-com", "linkup", "jina", "kagi", "serpapi", "duckduckgo", "puppeteer",
        "playwright", "browserbase", "selenium", "stagehand", "hyperbrowser",
        "browser-use", "firecrawl", "scrapegraphai", "apify", "scrapfly",
        "bright-data", "wikipedia", "arxiv", "pubmed", "semantic-scholar",
        "wolfram-alpha", "searxng", "crawl4ai",
    ],
    "productivity": [
        "notion", "confluence", "coda", "quip", "google-docs", "google-sheets",
        "microsoft-excel", "office365", "airtable", "smartsheet", "google-calendar",
        "outlook-calendar", "calendly", "cal-com", "jira", "linear", "asana",
        "trello", "monday", "clickup", "basecamp", "height", "shortcut", "todoist",
        "plane", "wrike", "obsidian", "evernote", "bear", "logseq", "google-drive",
        "dropbox", "box", "onedrive", "sharepoint",
    ],
    "misc": [
        "stripe", "paypal", "square", "adyen", "plaid", "salesforce", "hubspot",
        "pipedrive", "attio", "zoho-crm", "shopify", "woocommerce", "bigcommerce",
        "magento", "mailchimp", "klaviyo", "braze", "customer-io", "typeform",
        "google-forms", "jotform", "figma", "canva", "adobe", "framer", "openai",
        "anthropic", "huggingface", "replicate", "elevenlabs", "runway", "zapier",
        "make", "n8n", "pipedream", "composio", "rube", "auth0", "okta", "clerk",
        "workos", "docusign", "workday", "rippling", "gusto", "quickbooks", "xero",
        "netsuite",
    ],
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


def fetch_query_listings(client: httpx.Client, query: str, max_pages: int = 2) -> list[dict]:
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


def best_vendor_match(vendor_name: str, candidates: list[dict]) -> dict | None:
    """Pick the best Smithery server for a vendor name.

    Preference order:
    1. Exact qualifiedName match (canonical Smithery-hosted)
    2. Verified namespaced match where slug==vendor_name OR qualifiedName endswith /vendor_name
    3. Verified, highest useCount, vendor_name appears in qualifiedName
    """
    vlow = vendor_name.lower().replace("-", "").replace("_", "")
    verified = [s for s in candidates if s.get("verified")]
    if not verified:
        return None

    def name_norm(s: str) -> str:
        return s.lower().replace("-", "").replace("_", "")

    # Tier 1: exact qualifiedName match (canonical, slug='')
    for s in verified:
        if name_norm(s.get("qualifiedName") or "") == vlow:
            return s

    # Tier 2: namespaced where slug exactly matches vendor name
    for s in verified:
        slug = name_norm(s.get("slug") or "")
        if slug == vlow:
            return s

    # Tier 3: vendor_name appears in qualifiedName, take highest useCount
    matching = [s for s in verified if vlow in name_norm(s.get("qualifiedName") or "")]
    if matching:
        return max(matching, key=lambda s: s.get("useCount") or 0)

    return None


def fetch_all_listings(client: httpx.Client) -> list[tuple[dict, str]]:
    """Two-pass listing collection:

    Pass A (vendor allowlist): resolve each vendor name in VENDOR_ALLOWLIST to
    its best Smithery match. Captures brand-recognizable canonical MCPs.

    Pass B (top-N by useCount per domain): for each vendor query, also collect
    the top verified results regardless of name match. Captures popular MCPs
    that don't fit our exact-name expectations (e.g., papersearch/PaperSearcher,
    aurelianflo/core, blockscout/mcp-server).
    """
    seen_qn: set[str] = set()
    resolved: list[tuple[dict, str]] = []

    # Pass A: vendor allowlist (exact-name preference)
    for domain, vendors in VENDOR_ALLOWLIST.items():
        per_domain_found = 0
        for vendor in vendors:
            candidates = fetch_query_listings(client, vendor)
            match = best_vendor_match(vendor, candidates)
            if match is None:
                logger.info(f"vendor_unresolved domain={domain} vendor={vendor}")
                continue
            qn = match["qualifiedName"]
            if qn in seen_qn:
                continue
            seen_qn.add(qn)
            resolved.append((match, domain))
            per_domain_found += 1
            logger.info(f"vendor_resolved domain={domain} vendor={vendor} qn={qn} useCount={match.get('useCount')}")
        logger.info(f"pass_a_domain_total domain={domain} resolved={per_domain_found}/{len(vendors)}")

    # Pass B: top-N popular verified per domain bucket (catches non-allowlist popular MCPs)
    POPULAR_PER_DOMAIN = 25
    for domain, vendors in VENDOR_ALLOWLIST.items():
        bucket_candidates: dict[str, dict] = {}
        for vendor in vendors:
            for s in fetch_query_listings(client, vendor):
                if not s.get("verified"):
                    continue
                qn = s.get("qualifiedName")
                if not qn or qn in seen_qn or qn in bucket_candidates:
                    continue
                bucket_candidates[qn] = s
        # Top-N by useCount among non-already-resolved
        top = sorted(bucket_candidates.values(), key=lambda s: s.get("useCount") or 0, reverse=True)[:POPULAR_PER_DOMAIN]
        per_domain_added = 0
        for s in top:
            qn = s["qualifiedName"]
            seen_qn.add(qn)
            resolved.append((s, domain))
            per_domain_added += 1
            logger.info(f"pass_b_popular domain={domain} qn={qn} useCount={s.get('useCount')}")
        logger.info(f"pass_b_domain_total domain={domain} added={per_domain_added}")

    return resolved


# --- Phase 2: Filter listings ----------------------------------------------


def is_canonical(server: dict) -> bool:
    """Keep only Smithery-verified servers (canonical and namespaced both OK)."""
    return bool(server.get("verified"))


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

    # 1-3. Resolve vendor allowlist to Smithery servers (search + best-match)
    resolved = fetch_all_listings(client)
    print(f"[1-3/9] resolved {len(resolved)} vendor MCPs from allowlist", file=sys.stderr)

    # 4. Fetch details
    details: list[tuple[dict, dict]] = []
    for i, (srv, _domain) in enumerate(resolved, 1):
        d = fetch_detail(client, srv["qualifiedName"])
        if d:
            details.append((srv, d))
        if i % 25 == 0:
            print(f"[4/9] fetched {i}/{len(resolved)} details", file=sys.stderr)
    print(f"[4/9] fetched {len(details)} details total", file=sys.stderr)

    # 5. Normalize (with per-MCP cap to prevent slack/github/discord domination)
    candidates: list[ToolSchema] = []
    for srv, detail in details:
        # Sort tools alphabetically for determinism, take top N per MCP
        all_tools = sorted(detail.get("tools", []) or [], key=lambda t: t.get("name", ""))
        capped = all_tools[:PER_MCP_TOOL_CAP]
        for tool_dict in capped:
            ts = normalize_one(srv, tool_dict)
            if ts is not None:
                candidates.append(ts)
    print(
        f"[5/9] normalized to {len(candidates)} ToolSchema records "
        f"(per-MCP cap={PER_MCP_TOOL_CAP})",
        file=sys.stderr,
    )

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
