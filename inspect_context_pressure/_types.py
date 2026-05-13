from typing import TypedDict


class ToolSchema(TypedDict):
    """Tool schema shape consumed by the Solver.

    Mirrors the Pydantic ``ToolSchema`` that ICP-2's schema library will ship.
    Once that lands this TypedDict is replaced with a one-line import; downstream
    callers that read these fields by key continue to work unchanged.
    """

    name: str
    description: str
    parameters: dict
    domain: str
    content_category: str
    mcp_server: str
