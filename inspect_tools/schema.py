from typing import Literal

from mcp.types import Tool as MCPTool

Domain = Literal[
    "cloud-ops",
    "dev-tools",
    "data-analytics",
    "communication",
    "search",
    "productivity",
    "misc",
]

ContentCategory = Literal["general_popular"]


class ToolSchema(MCPTool):
    domain: Domain = "misc"
    content_category: ContentCategory = "general_popular"
    source_url: str
