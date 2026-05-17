#!/usr/bin/env python3
"""outputSchema generation orchestrator + audit harness.

Phase 3 lives here. Subcommands:

  batch-prepare    emit a per-batch work packet for a subagent
  merge-batch      pull subagent outputs into the corpus
  audit-batch      run the 10 mechanical gates on a just-merged batch
  review-packet    render a markdown packet for human visual review
  extract-entities scan a batch for newly-invented entities, merge into ban list
  validate-corpus  full mechanical audit across all 1,239 records
  spot-check       emit N random anchors (default 35 = 5/domain x 7)
  status           summary of corpus + cache state

Cache layout (all gitignored):
  .cache/batches/<batch_id>.json          input packets (vendor list + tool slice)
  .cache/output_schemas/<batch_id>.json   subagent outputs (merged into corpus by merge-batch)
  .cache/entity_ban_list.json             cumulative fictional-entity ban list
  .cache/review_log.json                  per-anchor review state
  .cache/audit_reports/<batch_id>.json    mechanical-audit findings
  .cache/review_packets/<batch_id>.md     human-review packets
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inspect_tools._synthesize import _scan_placeholders, substitute  # noqa: E402

CORPUS = REPO_ROOT / "inspect_tools" / "data" / "tool_schemas_v1.json"
CACHE = REPO_ROOT / ".cache"
BATCHES = CACHE / "batches"
OUTPUTS = CACHE / "output_schemas"
REPORTS = CACHE / "audit_reports"
PACKETS = CACHE / "review_packets"
BAN_LIST = CACHE / "entity_ban_list.json"
REVIEW_LOG = CACHE / "review_log.json"


# ---------- gate definitions ----------

BANNED_FIELD_NAMES = {
    "error", "errors", "error_code", "failure", "failed",
    "partial", "partial_failure", "partial_success",
    "deprecated", "deprecation_notice",
    "warning", "warnings", "notices", "info_messages",
    "needed", "provided",
    "retry_after", "retry_recommended", "rate_limit_remaining",
}

BANNED_DEFAULTS_RE = re.compile(
    r"\b(example\.(com|org|net)|acme\.(com|io)|globex\.com|foobar\.com|test\.com"
    r"|octocat|John Doe|Jane Doe|lorem ipsum|placeholder"
    r"|4242 ?4242|cus_test|pi_test|wksp_test"
    r"|0{8,}-0{4}-0{4}-0{4}-0{12}|123456789)\b",
    re.IGNORECASE,
)

BANNED_STATUS_VALUES = {
    "degraded", "throttled", "queued", "in_progress", "failed", "error",
}
BANNED_CONCLUSION_VALUES = {
    "failure", "cancelled", "timed_out", "action_required",
}
PAGINATION_CURSOR_FIELDS = {
    "next_cursor", "next_token", "nextPageToken", "next_page", "nextCursor",
}


# ---------- helpers ----------

def load_corpus() -> list[dict]:
    return json.loads(CORPUS.read_text())


def save_corpus(data: list[dict]) -> None:
    assert len(data) == 1239, f"corpus length invariant broken: {len(data)}"
    CORPUS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_ban_list() -> dict[str, list[str]]:
    if not BAN_LIST.exists():
        return {"companies": [], "people": [], "domains": [], "freeform": []}
    return json.loads(BAN_LIST.read_text())


def save_ban_list(bl: dict[str, list[str]]) -> None:
    CACHE.mkdir(exist_ok=True)
    BAN_LIST.write_text(json.dumps(bl, indent=2, sort_keys=True) + "\n")


def load_review_log() -> dict[str, dict]:
    if not REVIEW_LOG.exists():
        return {}
    return json.loads(REVIEW_LOG.read_text())


def save_review_log(log: dict[str, dict]) -> None:
    CACHE.mkdir(exist_ok=True)
    REVIEW_LOG.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")


def walk_strings(node: Any, path: str = "$"):
    """Yield (path, string) for every string in the structure."""
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from walk_strings(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk_strings(v, f"{path}[{i}]")


def walk_nodes(node: Any, path: str = "$"):
    """Yield (path, node) for every dict / list / primitive."""
    yield path, node
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk_nodes(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk_nodes(v, f"{path}[{i}]")


# ---------- gate implementations ----------

def gate_schema_validity(schema: dict) -> list[str]:
    try:
        Draft202012Validator.check_schema(schema)
        return []
    except Exception as e:
        return [f"schema-invalid: {e}"]


def gate_package_count(schema: dict) -> list[str]:
    pkgs = schema.get("examples", [])
    if not (1 <= len(pkgs) <= 3):
        return [f"package-count: {len(pkgs)} (must be 1-3)"]
    return []


def gate_post_substitution(schema: dict) -> list[str]:
    errs: list[str] = []
    pkgs = schema.get("examples", [])
    if not pkgs:
        return errs
    v = Draft202012Validator(schema)
    for i, pkg in enumerate(pkgs):
        try:
            v.validate(substitute(pkg, {}))
        except Exception as e:
            errs.append(f"pkg{i} empty-kwargs validation: {str(e)[:300]}")
    return errs


def gate_banned_field_names(schema: dict) -> list[str]:
    errs: list[str] = []
    # Check `properties` keys + every key inside `examples`
    props = schema.get("properties", {})
    for k in props.keys():
        if k.lower() in BANNED_FIELD_NAMES:
            errs.append(f"banned field name in properties: {k}")
    for path, node in walk_nodes(schema.get("examples", [])):
        if isinstance(node, dict):
            for k in node.keys():
                if k.lower() in BANNED_FIELD_NAMES:
                    errs.append(f"banned field name in {path}: {k}")
    # Also check $defs
    for path, node in walk_nodes(schema.get("$defs", {})):
        if isinstance(node, dict) and "properties" in node:
            for k in node["properties"].keys():
                if k.lower() in BANNED_FIELD_NAMES:
                    errs.append(f"banned field name in $defs {path}: {k}")
    return errs


def gate_default_authenticity(schema: dict) -> list[str]:
    """Run BANNED_DEFAULTS_RE against every placeholder default in examples."""
    errs: list[str] = []
    for pi, pkg in enumerate(schema.get("examples", [])):
        for path, s in walk_strings(pkg, f"examples[{pi}]"):
            for _, _, key, default in _scan_placeholders(s):
                if BANNED_DEFAULTS_RE.search(default):
                    errs.append(f"banned default at {path} key={key!r}: {default!r}")
    return errs


def gate_banned_content_values(schema: dict) -> list[str]:
    """Pagination cursors non-null; has_more=true; status/conclusion values."""
    errs: list[str] = []
    for pi, pkg in enumerate(schema.get("examples", [])):
        for path, node in walk_nodes(pkg, f"examples[{pi}]"):
            if not isinstance(node, dict):
                continue
            # has_more / hasMore / more_results_available
            for f in ("has_more", "hasMore", "more_results_available"):
                if node.get(f) is True:
                    errs.append(f"{path}.{f} = true (pagination signal)")
            # cursors
            for f in PAGINATION_CURSOR_FIELDS:
                if f in node and node[f] not in (None, ""):
                    errs.append(f"{path}.{f} non-null: {node[f]!r}")
            # status enums
            st = node.get("status")
            if isinstance(st, str) and st.lower() in BANNED_STATUS_VALUES:
                errs.append(f"{path}.status = {st!r}")
            # conclusion enums
            co = node.get("conclusion")
            if isinstance(co, str) and co.lower() in BANNED_CONCLUSION_VALUES:
                errs.append(f"{path}.conclusion = {co!r}")
            # HTTP status codes
            for f in ("status_code", "statusCode", "http_status"):
                if isinstance(node.get(f), int) and node[f] >= 400:
                    errs.append(f"{path}.{f} = {node[f]} (>= 400)")
    return errs


def gate_no_leaf_examples(schema: dict) -> list[str]:
    """No per-leaf `examples` arrays inside `properties`."""
    errs: list[str] = []
    props = schema.get("properties", {})
    for path, node in walk_nodes(props, "properties"):
        if isinstance(node, dict) and "examples" in node:
            errs.append(f"per-leaf examples array at {path}")
    return errs


def gate_placeholder_keys_valid(schema: dict) -> list[str]:
    """Flag `{key | default}` candidates where key is not a valid identifier.

    Reason: _scan_placeholders requires key.isidentifier(); invalid-key candidates
    are silently treated as literal text, so the agent's kwarg never substitutes.
    Authors who write `{a.b | x}` thinking it'll dot-deref expect substitution
    that won't happen. Either rename to a simple identifier or escape the braces.

    Suppress hits where the surrounding string is plausibly literal API content
    (e.g., encoded polylines that happen to contain `{` and `}` in their
    alphabet) by reporting the path so a human can judge.
    """
    errs: list[str] = []

    def scan_all(s: str):
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
                        yield content[:pipe].strip()
                        i = j
                        continue
            i += 1

    for pi, pkg in enumerate(schema.get("examples", [])):
        for path, s in walk_strings(pkg, f"examples[{pi}]"):
            for key in scan_all(s):
                if not key.isidentifier():
                    errs.append(f"{path}: candidate {{key|default}} with invalid key {key!r}")
    return errs


def gate_no_additional_properties_false(schema: dict) -> list[str]:
    """No `additionalProperties: false` anywhere in the schema (not in examples)."""
    errs: list[str] = []
    schema_only = {k: v for k, v in schema.items() if k != "examples"}
    for path, node in walk_nodes(schema_only):
        if isinstance(node, dict) and node.get("additionalProperties") is False:
            errs.append(f"additionalProperties:false at {path}")
    return errs


def collect_entity_tokens(text: str) -> set[str]:
    """Heuristic: capitalized multi-word phrases + .io/.dev/.com domains.

    Misses lowercase entities and single-word names; the human review packet
    catches what this misses. Goal: cheap collision check, not exhaustive
    entity recognition.
    """
    tokens: set[str] = set()
    # Capitalized 2-3 word names (companies, people)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text):
        tokens.add(m.group(1))
    # Lowercase invented domains: foo.io, foo-bar.dev, etc.
    for m in re.finditer(r"\b([a-z][a-z0-9-]{2,}\.(?:io|dev|app|co|so|ai|xyz|sh|tech|cloud|systems|labs|works))\b", text):
        tokens.add(m.group(1))
    return tokens


def gate_cross_batch_entity_collision(schema: dict, ban_list: dict) -> list[str]:
    banned = set(ban_list.get("companies", []) + ban_list.get("people", []) +
                 ban_list.get("domains", []) + ban_list.get("freeform", []))
    if not banned:
        return []
    errs: list[str] = []
    seen_collisions: set[tuple[str, str]] = set()
    for path, s in walk_strings(schema.get("examples", [])):
        for tok in collect_entity_tokens(s):
            if tok in banned and (tok, path) not in seen_collisions:
                seen_collisions.add((tok, path))
                errs.append(f"banned entity {tok!r} appears at {path}")
    return errs


GATES = [
    ("schema-validity", gate_schema_validity),
    ("post-substitution", gate_post_substitution),
    ("package-count", gate_package_count),
    ("banned-field-names", gate_banned_field_names),
    ("default-authenticity", gate_default_authenticity),
    ("banned-content-values", gate_banned_content_values),
    ("no-leaf-examples", gate_no_leaf_examples),
    ("no-additionalProperties-false", gate_no_additional_properties_false),
    ("placeholder-keys-valid", gate_placeholder_keys_valid),
]


def audit_schema(name: str, schema: dict, ban_list: dict, check_collision: bool = True) -> dict[str, list[str]]:
    findings: dict[str, list[str]] = {}
    for gname, gfn in GATES:
        errs = gfn(schema)
        if errs:
            findings[gname] = errs
    if check_collision:
        coll = gate_cross_batch_entity_collision(schema, ban_list)
        if coll:
            findings["cross-batch-entity-collision"] = coll
    return findings


# ---------- subcommands ----------

def cmd_status(args):
    data = load_corpus()
    n = len(data)
    pop = sum(1 for t in data if t.get("outputSchema"))
    print(f"corpus: {n} records, {pop} with outputSchema, {n - pop} remaining")
    bl = load_ban_list()
    print(f"ban list: {sum(len(v) for v in bl.values())} entries "
          f"(companies={len(bl.get('companies', []))} "
          f"people={len(bl.get('people', []))} "
          f"domains={len(bl.get('domains', []))} "
          f"freeform={len(bl.get('freeform', []))})")
    batches_dir = BATCHES if BATCHES.exists() else None
    outputs_dir = OUTPUTS if OUTPUTS.exists() else None
    nbatches = len(list(BATCHES.glob("*.json"))) if batches_dir else 0
    noutputs = len(list(OUTPUTS.glob("*.json"))) if outputs_dir else 0
    print(f"batches prepared: {nbatches}; subagent outputs: {noutputs}")


def cmd_batch_prepare(args):
    """Group remaining tools into batches of ~15-20 by adjacent vendors.

    Emits .cache/batches/<batch_id>.json packets. Idempotent: re-running
    overwrites packets but does not lose anything (subagent outputs live in
    .cache/output_schemas/, not in batches).
    """
    BATCHES.mkdir(parents=True, exist_ok=True)
    data = load_corpus()
    by_vendor: dict[str, list[dict]] = defaultdict(list)
    for t in data:
        if t.get("outputSchema"):
            continue
        tail = t.get("source_url", "").rsplit("/", 1)[-1] or "unknown"
        by_vendor[tail].append(t)

    # Sort vendors by tool count desc (big vendors get their own batch).
    vendors_sorted = sorted(by_vendor.items(), key=lambda kv: -len(kv[1]))

    batch_id = 0
    current: list[tuple[str, dict]] = []  # (vendor, tool)
    current_vendors: list[str] = []
    target = args.target_size
    written = 0

    def flush():
        nonlocal batch_id, current, current_vendors, written
        if not current:
            return
        batch_id += 1
        bid = f"b{batch_id:03d}"
        ban = load_ban_list()
        packet = {
            "batch_id": bid,
            "vendors": list(current_vendors),
            "tools": [t for _, t in current],
            "tool_count": len(current),
            "entity_ban_list": ban,
        }
        (BATCHES / f"{bid}.json").write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n")
        written += len(current)
        current = []
        current_vendors = []

    for vendor, tools in vendors_sorted:
        # If a single vendor has >target tools, split into multiple batches.
        if len(tools) > target:
            flush()
            for i in range(0, len(tools), target):
                current_vendors = [vendor]
                current = [(vendor, t) for t in tools[i:i + target]]
                flush()
            continue
        # If adding this vendor would push us past target, flush first.
        if len(current) + len(tools) > target and current:
            flush()
        current_vendors.append(vendor)
        current.extend((vendor, t) for t in tools)
    flush()

    print(f"wrote {batch_id} batches covering {written} tools to {BATCHES}")


def cmd_merge_batch(args):
    """Pull subagent outputs from .cache/output_schemas/<batch_id>.json into corpus.

    Output JSON format:
      {"batch_id": "...", "outputs": [
         {"name": "...", "source_url": "...", "outputSchema": {...}},
         ...
      ]}
    """
    out_path = OUTPUTS / f"{args.batch_id}.json"
    if not out_path.exists():
        sys.exit(f"missing {out_path}")
    payload = json.loads(out_path.read_text())
    outputs = payload.get("outputs", [])
    if isinstance(outputs, dict):
        sys.exit(f"{out_path}: 'outputs' must be a list of records; got dict")
    data = load_corpus()
    by_key = {(t["name"], t.get("source_url", "")): t for t in data}
    merged = 0
    skipped: list[str] = []
    overwritten: list[str] = []
    for rec in outputs:
        key = (rec["name"], rec.get("source_url", ""))
        if key not in by_key:
            skipped.append(f"{key[0]}@{key[1]}")
            continue
        if by_key[key].get("outputSchema"):
            overwritten.append(key[0])
        by_key[key]["outputSchema"] = rec["outputSchema"]
        merged += 1
    save_corpus(data)
    print(f"merged {merged} schemas from {args.batch_id}; skipped {len(skipped)}; overwrote {len(overwritten)}")
    if skipped:
        print(f"  skipped (not in corpus): {skipped[:5]}...")
    if overwritten:
        print(f"  overwrote existing: {overwritten[:5]}...")


def cmd_audit_batch(args):
    """Run all gates on every tool in a batch's just-merged records."""
    packet_path = BATCHES / f"{args.batch_id}.json"
    if not packet_path.exists():
        sys.exit(f"missing {packet_path}")
    packet = json.loads(packet_path.read_text())
    keys = {(t["name"], t.get("source_url", "")) for t in packet["tools"]}
    data = load_corpus()
    if len(data) != 1239:
        sys.exit(f"corpus length invariant broken: {len(data)}")
    # Use packet's ban-list snapshot (state at batch-prepare time), not live.
    # The live ban list may already contain entities this batch itself introduced
    # (parallel-wave subagent updates, or rogue subagent writes), which would
    # produce false-positive collisions when auditing the batch against itself.
    ban_list = packet.get("entity_ban_list") or load_ban_list()
    REPORTS.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    n_clean = 0
    n_critical = 0
    n_unfilled = 0
    for t in data:
        key = (t["name"], t.get("source_url", ""))
        if key not in keys:
            continue
        kstr = f"{key[0]}@{key[1]}"
        sch = t.get("outputSchema")
        if not sch:
            report[kstr] = {"unfilled": True}
            n_unfilled += 1
            continue
        findings = audit_schema(t["name"], sch, ban_list)
        if findings:
            report[kstr] = findings
            n_critical += 1
        else:
            n_clean += 1
    out = REPORTS / f"{args.batch_id}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"batch {args.batch_id}: {n_clean} clean, {n_critical} with findings, {n_unfilled} unfilled")
    print(f"report -> {out}")
    if n_critical:
        print("\nfindings preview:")
        for name, f in list(report.items())[:5]:
            print(f"  {name}: {list(f.keys())}")


def cmd_review_packet(args):
    """Render a markdown packet for human visual review of a batch."""
    packet_path = BATCHES / f"{args.batch_id}.json"
    if not packet_path.exists():
        sys.exit(f"missing {packet_path}")
    packet = json.loads(packet_path.read_text())
    keys = [(t["name"], t.get("source_url", "")) for t in packet["tools"]]
    data = load_corpus()
    by_key = {(t["name"], t.get("source_url", "")): t for t in data}
    report_path = REPORTS / f"{args.batch_id}.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    PACKETS.mkdir(parents=True, exist_ok=True)
    out = PACKETS / f"{args.batch_id}.md"
    lines: list[str] = []
    lines.append(f"# Review packet: batch {args.batch_id}")
    lines.append("")
    lines.append(f"Vendors: {', '.join(packet['vendors'])}  ")
    lines.append(f"Tools: {len(keys)}")
    lines.append("")
    def kstr(k): return f"{k[0]}@{k[1]}"
    n_clean = sum(1 for k in keys if not report.get(kstr(k)))
    n_findings = sum(1 for k in keys if report.get(kstr(k)) and not report[kstr(k)].get("unfilled"))
    n_unfilled = sum(1 for k in keys if report.get(kstr(k), {}).get("unfilled"))
    lines.append(f"Mechanical audit: {n_clean} clean / {n_findings} with findings / {n_unfilled} unfilled")
    lines.append("")
    for key in keys:
        name = key[0]
        t = by_key.get(key)
        if not t:
            lines.append(f"## {name}\n\n**MISSING FROM CORPUS**\n")
            continue
        sch = t.get("outputSchema")
        findings = report.get(kstr(key), {})
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- domain: `{t.get('domain', '?')}`  source: `{t.get('source_url', '?')}`")
        desc = t.get("description", "").strip().replace("\n", " ")
        if len(desc) > 240:
            desc = desc[:240] + "..."
        lines.append(f"- description: {desc}")
        if findings.get("unfilled"):
            lines.append("")
            lines.append("**UNFILLED — no outputSchema present**")
            lines.append("")
            continue
        if not sch:
            lines.append("")
            lines.append("**NO outputSchema (race?)**")
            lines.append("")
            continue
        if findings:
            lines.append("")
            lines.append("**Mechanical findings:**")
            for gate, errs in findings.items():
                lines.append(f"- {gate}:")
                for e in errs[:5]:
                    lines.append(f"  - {e}")
        lines.append("")
        lines.append("**Schema (properties + required):**")
        lines.append("```json")
        skeleton = {k: sch[k] for k in ("type", "required", "properties", "$defs") if k in sch}
        lines.append(json.dumps(skeleton, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
        lines.append(f"**Packages ({len(sch.get('examples', []))}):**")
        for i, pkg in enumerate(sch.get("examples", [])):
            lines.append(f"\nPackage {i}:")
            lines.append("```json")
            lines.append(json.dumps(pkg, indent=2, ensure_ascii=False))
            lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def cmd_extract_entities(args):
    """Scan a batch's records for invented entities; emit a diff for human approval.

    Does NOT auto-merge into the ban list — prints proposed additions and writes
    them to .cache/audit_reports/<batch_id>.entities.json so I can review and
    explicitly approve via `--apply`.

    With --from-corpus, scans every populated record (for ban-list seeding from
    the 30 Phase-2 anchors).
    """
    data = load_corpus()
    if args.from_corpus:
        scan_label = "corpus"
        def match(t): return bool(t.get("outputSchema"))
    else:
        if not args.batch_id:
            sys.exit("either --from-corpus or a batch_id required")
        scan_label = args.batch_id
        packet_path = BATCHES / f"{args.batch_id}.json"
        if not packet_path.exists():
            sys.exit(f"missing {packet_path}")
        packet = json.loads(packet_path.read_text())
        keys = {(t["name"], t.get("source_url", "")) for t in packet["tools"]}
        def match(t): return (t["name"], t.get("source_url", "")) in keys
    bl = load_ban_list()
    existing = set(bl.get("companies", []) + bl.get("people", []) +
                   bl.get("domains", []) + bl.get("freeform", []))
    new_tokens: set[str] = set()
    for t in data:
        if not match(t):
            continue
        sch = t.get("outputSchema")
        if not sch:
            continue
        for _, s in walk_strings(sch.get("examples", [])):
            new_tokens.update(collect_entity_tokens(s))
    proposed = sorted(new_tokens - existing)
    REPORTS.mkdir(parents=True, exist_ok=True)
    diff_path = REPORTS / f"{scan_label}.entities.json"
    diff_path.write_text(json.dumps({"proposed": proposed}, indent=2) + "\n")
    print(f"new tokens in {scan_label}: {len(proposed)}")
    if args.apply:
        bl.setdefault("freeform", []).extend(proposed)
        bl["freeform"] = sorted(set(bl["freeform"]))
        save_ban_list(bl)
        print(f"merged {len(proposed)} tokens into ban list (freeform bucket)")
    else:
        print(f"diff -> {diff_path}  (re-run with --apply to merge)")
        for t in proposed[:30]:
            print(f"  + {t}")


def cmd_validate_corpus(args):
    """Full mechanical audit across all 1,239 records. Phase 3E gate."""
    data = load_corpus()
    if len(data) != 1239:
        sys.exit(f"corpus length invariant broken: {len(data)}")
    ban_list = load_ban_list()
    failures: dict[str, dict] = {}
    n_clean = 0
    n_findings = 0
    n_unfilled = 0
    for t in data:
        kstr = f"{t['name']}@{t.get('source_url', '')}"
        sch = t.get("outputSchema")
        if not sch:
            failures[kstr] = {"unfilled": True}
            n_unfilled += 1
            continue
        findings = audit_schema(t["name"], sch, ban_list, check_collision=False)
        if findings:
            failures[kstr] = findings
            n_findings += 1
        else:
            n_clean += 1
    print(f"validate-corpus: {n_clean} clean / {n_findings} with findings / {n_unfilled} unfilled")
    if failures:
        REPORTS.mkdir(parents=True, exist_ok=True)
        out = REPORTS / "validate_corpus.json"
        out.write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n")
        print(f"failures -> {out}")
        if n_unfilled or n_findings:
            sys.exit(1)


def cmd_spot_check(args):
    """Emit a markdown packet with N random anchors balanced across domains."""
    data = load_corpus()
    populated = [t for t in data if t.get("outputSchema")]
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for t in populated:
        by_domain[t.get("domain", "misc")].append(t)
    rng = random.Random(args.seed)
    per_domain = args.n // max(len(by_domain), 1)
    extra = args.n - per_domain * len(by_domain)
    picks: list[dict] = []
    for dom, tools in sorted(by_domain.items()):
        k = min(per_domain + (1 if extra > 0 else 0), len(tools))
        if extra > 0:
            extra -= 1
        picks.extend(rng.sample(tools, k))
    PACKETS.mkdir(parents=True, exist_ok=True)
    out = PACKETS / f"spot-check-{args.n}.md"
    lines = [f"# Spot-check packet: {len(picks)} anchors (seed={args.seed})", ""]
    for t in picks:
        lines.append(f"## {t['name']}  ({t.get('domain', '?')})")
        lines.append("")
        lines.append(f"- source: `{t.get('source_url', '?')}`")
        desc = t.get("description", "").strip().replace("\n", " ")[:240]
        lines.append(f"- description: {desc}")
        lines.append("")
        sch = t["outputSchema"]
        skeleton = {k: sch[k] for k in ("type", "required", "properties", "$defs") if k in sch}
        lines.append("**Schema:**")
        lines.append("```json")
        lines.append(json.dumps(skeleton, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
        lines.append("**Packages:**")
        for i, pkg in enumerate(sch.get("examples", [])):
            lines.append(f"\nPackage {i}:")
            lines.append("```json")
            lines.append(json.dumps(pkg, indent=2, ensure_ascii=False))
            lines.append("```")
        lines.append("\n---\n")
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(picks)} anchors)")


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)

    sp.add_parser("status", help="corpus + cache summary")

    bp = sp.add_parser("batch-prepare", help="group unfilled tools into batches")
    bp.add_argument("--target-size", type=int, default=18,
                    help="target tools per batch (default 18)")

    mb = sp.add_parser("merge-batch", help="merge subagent outputs into corpus")
    mb.add_argument("batch_id")

    ab = sp.add_parser("audit-batch", help="run gates on a batch")
    ab.add_argument("batch_id")

    rp = sp.add_parser("review-packet", help="render markdown review packet")
    rp.add_argument("batch_id")

    ee = sp.add_parser("extract-entities", help="extract invented entities for ban-list approval")
    ee.add_argument("batch_id", nargs="?", default=None)
    ee.add_argument("--from-corpus", action="store_true", help="scan all populated records (for seeding)")
    ee.add_argument("--apply", action="store_true", help="merge approved tokens into ban list")

    sp.add_parser("validate-corpus", help="full audit across all 1,239 records")

    sc = sp.add_parser("spot-check", help="emit random N-anchor sample")
    sc.add_argument("--n", type=int, default=35)
    sc.add_argument("--seed", type=int, default=2026)

    args = p.parse_args()
    fn = {
        "status": cmd_status,
        "batch-prepare": cmd_batch_prepare,
        "merge-batch": cmd_merge_batch,
        "audit-batch": cmd_audit_batch,
        "review-packet": cmd_review_packet,
        "extract-entities": cmd_extract_entities,
        "validate-corpus": cmd_validate_corpus,
        "spot-check": cmd_spot_check,
    }[args.cmd]
    fn(args)


if __name__ == "__main__":
    main()
