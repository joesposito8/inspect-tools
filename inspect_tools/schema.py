import re
from typing import Literal

from mcp.types import Tool as MCPTool
from pydantic import field_validator

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

# Anthropic / OpenAI / Gemini tool-name compatibility regex.
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


class ToolSchema(MCPTool):
    domain: Domain = "misc"
    content_category: ContentCategory = "general_popular"
    source_url: str

    @field_validator("name")
    @classmethod
    def _name_provider_compatible(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError(
                f"tool name {v!r} violates provider regex ^[a-zA-Z0-9_-]{{1,128}}$"
            )
        return v
