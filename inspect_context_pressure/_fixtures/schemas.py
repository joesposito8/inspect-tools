from inspect_context_pressure._types import ToolSchema

# 8 Category-A schemas across the 7 domains + 2 Category-B vacuous controls.
# Real-shaped MCP-style JSON; ICP-4 ships the scraped corpus that replaces this.

FIXTURE_SCHEMAS: list[ToolSchema] = [
    {
        "name": "aws_s3_put_object",
        "description": "Upload an object to an Amazon S3 bucket. Returns the version ID and ETag of the uploaded object.",
        "parameters": {
            "type": "object",
            "properties": {
                "bucket": {"type": "string", "description": "Target S3 bucket name."},
                "key": {"type": "string", "description": "Object key within the bucket."},
                "body": {"type": "string", "description": "Object body content (UTF-8 text or base64-encoded binary)."},
                "content_type": {"type": "string", "description": "MIME type, e.g. application/json."},
            },
            "required": ["bucket", "key", "body"],
        },
        "domain": "cloud-ops",
        "content_category": "A_general_popular",
        "mcp_server": "anthropic-mcp/aws-toolkit",
    },
    {
        "name": "github_create_pull_request",
        "description": "Open a pull request on a GitHub repository. Returns the PR number and HTML URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "head": {"type": "string", "description": "Branch with the changes."},
                "base": {"type": "string", "description": "Branch to merge into."},
                "body": {"type": "string", "description": "PR description (Markdown)."},
                "draft": {"type": "boolean", "default": False},
            },
            "required": ["owner", "repo", "title", "head", "base"],
        },
        "domain": "dev-tools",
        "content_category": "A_general_popular",
        "mcp_server": "smithery/github",
    },
    {
        "name": "snowflake_execute_query",
        "description": "Execute a SQL statement against a Snowflake warehouse and return the result rows as a JSON array.",
        "parameters": {
            "type": "object",
            "properties": {
                "warehouse": {"type": "string"},
                "database": {"type": "string"},
                "schema": {"type": "string"},
                "query": {"type": "string", "description": "SQL statement to run."},
                "row_limit": {"type": "integer", "default": 1000, "description": "Maximum rows to return."},
            },
            "required": ["warehouse", "database", "schema", "query"],
        },
        "domain": "data-analytics",
        "content_category": "A_general_popular",
        "mcp_server": "smithery/snowflake",
    },
    {
        "name": "slack_post_message",
        "description": "Post a message to a Slack channel as the authenticated user or bot. Supports rich-text blocks.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Channel ID or name (e.g. #general)."},
                "text": {"type": "string", "description": "Plaintext fallback message."},
                "blocks": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Slack Block Kit blocks for rich rendering.",
                },
                "thread_ts": {"type": "string", "description": "If set, post as a reply in this thread."},
            },
            "required": ["channel", "text"],
        },
        "domain": "communication",
        "content_category": "A_general_popular",
        "mcp_server": "anthropic-mcp/slack",
    },
    {
        "name": "google_search",
        "description": "Run a Google web search and return the top organic results with title, snippet, and URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "num_results": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                "site": {"type": "string", "description": "Optional site: filter, e.g. arxiv.org."},
                "language": {"type": "string", "default": "en"},
            },
            "required": ["query"],
        },
        "domain": "search",
        "content_category": "A_general_popular",
        "mcp_server": "smithery/serpapi",
    },
    {
        "name": "notion_create_page",
        "description": "Create a new page in a Notion workspace under the given parent page or database.",
        "parameters": {
            "type": "object",
            "properties": {
                "parent_id": {"type": "string"},
                "title": {"type": "string"},
                "content": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Notion block objects forming the page body.",
                },
                "icon_emoji": {"type": "string"},
            },
            "required": ["parent_id", "title"],
        },
        "domain": "productivity",
        "content_category": "A_general_popular",
        "mcp_server": "anthropic-mcp/notion",
    },
    {
        "name": "stripe_create_charge",
        "description": "Create a charge against a Stripe customer or one-time payment source. Returns the charge ID and status.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "integer", "description": "Amount in the smallest currency unit (e.g. cents)."},
                "currency": {"type": "string", "description": "ISO 4217 currency code, e.g. usd."},
                "source": {"type": "string", "description": "Payment source token or card ID."},
                "customer": {"type": "string", "description": "Stripe customer ID."},
                "description": {"type": "string"},
            },
            "required": ["amount", "currency"],
        },
        "domain": "misc",
        "content_category": "A_general_popular",
        "mcp_server": "smithery/stripe",
    },
    {
        "name": "kubernetes_apply_manifest",
        "description": "Apply a Kubernetes manifest YAML to the configured cluster context. Equivalent to `kubectl apply -f`.",
        "parameters": {
            "type": "object",
            "properties": {
                "manifest_yaml": {"type": "string", "description": "Manifest body (one or more YAML documents)."},
                "namespace": {"type": "string", "default": "default"},
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["manifest_yaml"],
        },
        "domain": "cloud-ops",
        "content_category": "A_general_popular",
        "mcp_server": "anthropic-mcp/kubernetes",
    },
    # --- Category B: vacuous controls (token-matched, semantically empty) ---
    {
        "name": "vacuous_alpha_tool",
        "description": "Performs alpha operation on the input payload. Returns an alpha-typed response object containing the alpha result and an alpha status code.",
        "parameters": {
            "type": "object",
            "properties": {
                "alpha_input": {"type": "string", "description": "Alpha input string of arbitrary length."},
                "alpha_mode": {"type": "string", "enum": ["alpha_one", "alpha_two", "alpha_three"]},
                "alpha_count": {"type": "integer", "default": 1},
            },
            "required": ["alpha_input"],
        },
        "domain": "misc",
        "content_category": "B_vacuous_controls",
        "mcp_server": "icp/control-pool",
    },
    {
        "name": "vacuous_beta_tool",
        "description": "Performs beta operation on the input payload. Returns a beta-typed response object containing the beta result and a beta status code.",
        "parameters": {
            "type": "object",
            "properties": {
                "beta_input": {"type": "string", "description": "Beta input string of arbitrary length."},
                "beta_mode": {"type": "string", "enum": ["beta_one", "beta_two", "beta_three"]},
                "beta_count": {"type": "integer", "default": 1},
            },
            "required": ["beta_input"],
        },
        "domain": "misc",
        "content_category": "B_vacuous_controls",
        "mcp_server": "icp/control-pool",
    },
]
