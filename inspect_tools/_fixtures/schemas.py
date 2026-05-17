"""Test-only ToolSchema fixtures spanning all 7 Domain values + edge cases.

Mirrors the production corpus contract (mcp.types.Tool shape + outputSchema.examples
with `{kwarg | default}` placeholders). Used by tests via monkeypatching the
_CORPUS_CACHE in inspect_tools._library.
"""
from inspect_tools.schema import ToolSchema

FIXTURE_SCHEMAS: list[ToolSchema] = [
    # cloud-ops: whole-value placeholders, server-generated literals varied across packages
    ToolSchema(
        name="aws_s3_put_object",
        description="Upload an object to an Amazon S3 bucket. Returns the version ID and ETag.",
        inputSchema={
            "type": "object",
            "properties": {
                "bucket": {"type": "string", "description": "Target S3 bucket name."},
                "key": {"type": "string"},  # missing description — exercises auto-fill
                "body": {"type": "string"},
            },
            "required": ["bucket", "key", "body"],
        },
        outputSchema={
            "type": "object",
            "properties": {
                "bucket": {"type": "string"},
                "key": {"type": "string"},
                "version_id": {"type": "string"},
                "etag": {"type": "string"},
            },
            "required": ["bucket", "key", "version_id", "etag"],
            "examples": [
                {
                    "bucket": "{bucket | test-bucket}",
                    "key": "{key | fixture/unnamed.txt}",
                    "version_id": "3HL4kqtJlcpXroDTDmJ+rmSpXd3dIbrHY",
                    "etag": '"d41d8cd98f00b204e9800998ecf8427e"',
                },
                {
                    "bucket": "{bucket | test-bucket}",
                    "key": "{key | fixture/unnamed.txt}",
                    "version_id": "QUpfdndhfd8438MNFDN93jdnJFkdmqnh893",
                    "etag": '"098f6bcd4621d373cade4e832627b4f6"',
                },
            ],
        },
        domain="cloud-ops",
        content_category="general_popular",
        source_url="https://test.fixture/aws-s3",
    ),
    # dev-tools: same-kwarg-in-both-modes (mirrors production actions_get)
    ToolSchema(
        name="github_actions_get",
        description="Get a workflow run by ID. Returns run metadata.",
        inputSchema={
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repo owner."},
                "repo": {"type": "string", "description": "Repo name."},
                "resource_id": {"type": "integer", "description": "Run ID."},
            },
            "required": ["owner", "repo", "resource_id"],
        },
        outputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "status": {"type": "string"},
                "html_url": {"type": "string"},
            },
            "required": ["id", "html_url"],
            "examples": [
                {
                    "id": "{resource_id | 8472193056}",
                    "status": "completed",
                    "html_url": "https://github.com/{owner | facebook}/{repo | react}/actions/runs/{resource_id | 8472193056}",
                },
            ],
        },
        domain="dev-tools",
        content_category="general_popular",
        source_url="https://test.fixture/github-actions",
    ),
    # data-analytics: non-string defaults (null, 0, [], {})
    ToolSchema(
        name="intercom_create_or_update_company",
        description="Create or update an Intercom company by name.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Company name."},
                "size": {"type": "integer", "description": "Employee count."},
                "monthly_spend": {"type": "integer", "description": "Monthly spend."},
            },
            "required": ["name"],
        },
        outputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "size": {"type": ["integer", "null"]},
                "monthly_spend": {"type": "integer"},
                "plan": {"type": ["string", "null"]},
                "tags": {"type": "array"},
                "custom_attributes": {"type": "object"},
            },
            "required": ["name"],
            "examples": [
                {
                    "name": "{name | New Company}",
                    "size": "{size | null}",
                    "monthly_spend": "{monthly_spend | 0}",
                    "plan": "{plan | null}",
                    "tags": "{tags | []}",
                    "custom_attributes": "{custom_attributes | {}}",
                },
            ],
        },
        domain="data-analytics",
        content_category="general_popular",
        source_url="https://test.fixture/intercom",
    ),
    # communication: deeply nested placeholders inside list-of-dicts (mirrors Gmail draft)
    ToolSchema(
        name="gmail_create_draft",
        description="Create a Gmail draft. Returns the draft message with headers.",
        inputSchema={
            "type": "object",
            "properties": {
                "recipient_email": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["recipient_email", "subject", "body"],
        },
        outputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "message": {"type": "object"},
            },
            "required": ["id", "message"],
            "examples": [
                {
                    "id": "draft_abc123",
                    "message": {
                        "snippet": "{body | Default snippet}",
                        "headers": [
                            {"name": "To", "value": "{recipient_email | nobody@example.com}"},
                            {"name": "From", "value": "Test User <test@fixture.local>"},
                            {"name": "Subject", "value": "{subject | Quick note}"},
                            {"name": "Date", "value": "Mon, 17 Nov 2025 14:37:07 -0500"},
                        ],
                    },
                },
            ],
        },
        domain="communication",
        content_category="general_popular",
        source_url="https://test.fixture/gmail",
    ),
    # search: embedded mode + literal results array
    ToolSchema(
        name="tavily_search",
        description="Run a web search and return top results.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
            },
            "required": ["query"],
        },
        outputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "results": {"type": "array"},
            },
            "required": ["query", "results"],
            "examples": [
                {
                    "query": "{query | how to shard vector indexes}",
                    "results": [
                        {
                            "title": "Sharding strategies for billion-scale vector indexes",
                            "url": "https://example.com/sharding",
                            "score": 0.83,
                        },
                    ],
                },
            ],
        },
        domain="search",
        content_category="general_popular",
        source_url="https://test.fixture/tavily",
    ),
    # productivity: simple whole-value placeholder
    ToolSchema(
        name="notion_create_page",
        description="Create a new page in a Notion workspace.",
        inputSchema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Page title."},
                "parent_id": {"type": "string", "description": "Parent page ID."},
            },
            "required": ["title"],
        },
        outputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["id", "title"],
            "examples": [
                {
                    "id": "page_5b3c1a6e",
                    "title": "{title | Untitled}",
                },
            ],
        },
        domain="productivity",
        content_category="general_popular",
        source_url="https://test.fixture/notion",
    ),
    # misc: WITHOUT outputSchema — exercises {"ok": True} fallback
    ToolSchema(
        name="misc_ping",
        description="Trivial ping with no response packages defined.",
        inputSchema={
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Host to ping."},
            },
            "required": ["host"],
        },
        # no outputSchema — fallback path
        domain="misc",
        content_category="general_popular",
        source_url="https://test.fixture/misc-ping",
    ),
]
