#!/usr/bin/env python3
"""Compact LLM-output review: 2 realistic invocations per anchor, one line each.

For each populated anchor, generates two distinct realistic kwarg sets based on
inputSchema property names + types, calls substitute() (the Solver's actual
substitution path), and prints the compact JSON response. ~4 lines per anchor.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inspect_tools._synthesize import substitute  # noqa: E402

CORPUS = REPO_ROOT / "inspect_tools" / "data" / "tool_schemas_v1.json"

# Per-semantic kwarg pools. Two entries each so invocation A and B differ.
POOLS_A = {
    "id":          ["rec_4f8a1c93e207b5d2"],
    "uuid":        ["7c1f3a82-49d0-4ab8-b6e1-f2a8d05c7b14"],
    "name":        ["Aurora Migration Plan"],
    "title":       ["Q3 hardening rollout"],
    "description": ["End-to-end charge orchestration"],
    "query":       ["rust async runtime comparison"],
    "search":      ["distributed tracing best practices"],
    "text":        ["Migration plan ready for review."],
    "body":        ["Thanks for the update; merging tomorrow."],
    "message":     ["Deploy succeeded — promoting to prod."],
    "content":     ["See the runbook for rollback steps."],
    "email":       ["yuki.hoshino@finchhollow.io"],
    "url":         ["https://docs.finchhollow.io/runbooks/api"],
    "owner":       ["rust-lang"],
    "repo":        ["rustfmt"],
    "subreddit":   ["learnrust"],
    "channel":     ["C04G7XR2P"],
    "team":        ["platform"],
    "subject":     ["API key rotation reminder"],
    "tag":         ["orders-api"],
    "callertag":   ["orders-api"],
    "calleetag":   ["payments-gateway"],
    "key":         ["runbook_url"],
    "value":       ["https://docs.finchhollow.io/runbooks/orders"],
    "domain":      ["finchhollow.io"],
    "method":      ["POST"],
    "path":        ["/v1/charges"],
    "language":    ["python"],
    "type":        ["service"],
    "filter":      ["status=active"],
    "limit":       [25],
    "page":        [1],
    "pageSize":    [25],
    "size":        [50],
    "count":       [10],
    "max":         [50],
    "depth":       [2],
    "startdate":   ["2026-04-01"],
    "enddate":     ["2026-05-01"],
    "from_date":   ["2026-04-01"],
    "to_date":     ["2026-05-01"],
    "from":        ["2026-04-01"],
    "to":          ["2026-05-01"],
    "date":        ["2026-04-15"],
    "version":     ["v2.7.1"],
    "status":      ["active"],
    "role":        ["editor"],
    "category":    ["productivity"],
    "tags":        [["urgent", "infra"]],
    "items":       [["item-1", "item-2"]],
}
POOLS_B = {
    "id":          ["evt_a3f17e0b9c2d"],
    "uuid":        ["b41c9d2e-8f7a-4061-9d35-491fab27ce51"],
    "name":        ["Vendor Onboarding"],
    "title":       ["Adopt the new audit policy"],
    "description": ["Compatibility shim removal"],
    "query":       ["postgres index bloat fix"],
    "search":      ["service mesh canary deploy"],
    "text":        ["Closing this out — solution worked, see linked PR."],
    "body":        ["Approved with comments. Two nits inline."],
    "message":     ["Heads-up: oncall rotation shifts next week."],
    "content":     ["The deploy at 14:32 UTC is the culprit; see flamegraph."],
    "email":       ["priya.balasubramanian@ravenstead.dev"],
    "url":         ["https://status.ravenstead.dev/incidents/2026-q2-04"],
    "owner":       ["kubernetes"],
    "repo":        ["client-go"],
    "subreddit":   ["devops"],
    "channel":     ["C09K2NRH4"],
    "team":        ["security"],
    "subject":     ["RFC-12 review window opens Friday"],
    "tag":         ["payments-gateway"],
    "key":         ["pagerduty_service_id"],
    "value":       ["PJZ4Q7K"],
    "domain":      ["ravenstead.dev"],
    "method":      ["GET"],
    "path":        ["/v2/invoices/{invoice_id}"],
    "language":    ["go"],
    "type":        ["team"],
    "filter":      ["assignee=me"],
    "limit":       [100],
    "page":        [2],
    "pageSize":    [50],
    "size":        [200],
    "count":       [50],
    "max":         [100],
    "depth":       [4],
    "startDate":   ["2026-01-01"],
    "endDate":     ["2026-03-31"],
    "from":        ["2026-01-01"],
    "to":          ["2026-03-31"],
    "date":        ["2026-02-14"],
    "version":     ["v4.21.2"],
    "status":      ["completed"],
    "role":        ["viewer"],
    "category":    ["analytics"],
    "tags":        [["q3-okr", "blocked"]],
    "items":       [["foo", "bar", "baz"]],
}


def _lookup_pool(name: str, pool: dict):
    """Match field name to a pool key by lowercase substring."""
    lname = name.lower()
    # try exact name match, then suffix match, then substring
    if lname in pool:
        return pool[lname][0]
    for key, vals in pool.items():
        if lname.endswith(key.lower()) or lname.endswith("_" + key.lower()):
            return vals[0]
    for key, vals in pool.items():
        if key.lower() in lname:
            return vals[0]
    return None


def sample_realistic(input_schema: dict, pool: dict) -> dict:
    """Produce realistic kwargs based on inputSchema property names + types."""
    if not input_schema or not isinstance(input_schema, dict):
        return {}
    props = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    out: dict = {}
    for name, spec in props.items():
        if name not in required and len(out) >= 4:
            continue
        if not isinstance(spec, dict):
            continue
        t = spec.get("type")
        if isinstance(t, list):
            t = next((x for x in t if x != "null"), t[0])
        enum = spec.get("enum")
        if enum:
            out[name] = enum[0]
            continue
        hit = _lookup_pool(name, pool)
        if hit is not None:
            out[name] = hit
            continue
        if t == "string":
            out[name] = f"value-for-{name}"
        elif t == "integer":
            out[name] = 7
        elif t == "number":
            out[name] = 7.5
        elif t == "boolean":
            out[name] = True
        elif t == "array":
            out[name] = []
        elif t == "object":
            out[name] = {}
        else:
            out[name] = f"value-for-{name}"
    return out


def compact_json(obj, max_len: int = 220) -> str:
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if len(s) > max_len:
        return s[:max_len - 1] + "…"
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=None,
                   help="output markdown path (default: .cache/review_packets/llm_outputs[_<batch>].md)")
    p.add_argument("--batch", default=None,
                   help="only render anchors in the given batch id (e.g. b007)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-line", type=int, default=220)
    args = p.parse_args()
    corpus = json.loads(CORPUS.read_text())
    populated = [t for t in corpus if t.get("outputSchema")]
    if args.batch:
        batch_path = REPO_ROOT / ".cache" / "batches" / f"{args.batch}.json"
        if not batch_path.exists():
            sys.exit(f"missing {batch_path}")
        pkt = json.loads(batch_path.read_text())
        keys = {(t["name"], t["source_url"]) for t in pkt["tools"]}
        populated = [t for t in populated
                     if (t["name"], t.get("source_url", "")) in keys]
        default_out = f".cache/review_packets/llm_outputs_{args.batch}.md"
    else:
        default_out = ".cache/review_packets/llm_outputs.md"
    if args.limit:
        populated = populated[:args.limit]
    args.out = args.out or default_out
    lines: list[str] = [f"# LLM responses for {len(populated)} anchors", ""]
    for i, t in enumerate(populated):
        sch = t["outputSchema"]
        pkgs = sch.get("examples", [])
        if not pkgs:
            continue
        kwargs_a = sample_realistic(t.get("inputSchema", {}), POOLS_A)
        kwargs_b = sample_realistic(t.get("inputSchema", {}), POOLS_B)
        pkg_a = pkgs[0]
        pkg_b = pkgs[1] if len(pkgs) > 1 else pkgs[0]
        out_a = substitute(pkg_a, kwargs_a)
        out_b = substitute(pkg_b, kwargs_b)
        vendor = t.get("source_url", "?").rsplit("/", 1)[-1]
        lines.append(f"## {i+1}. `{t['name']}` _({t.get('domain','?')} · {vendor})_")
        lines.append(f"A: `{compact_json(kwargs_a, 110)}` → `{compact_json(out_a, args.max_line)}`")
        lines.append(f"B: `{compact_json(kwargs_b, 110)}` → `{compact_json(out_b, args.max_line)}`")
        lines.append("")
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path} ({len(populated)} anchors, {len(lines)} lines)")


if __name__ == "__main__":
    main()
