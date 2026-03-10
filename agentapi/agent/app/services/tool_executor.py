from pathlib import Path
from app.tools.registry import TOOL_REGISTRY


def execute_tool(tool_name: str, file_path: Path) -> dict:
    tool = TOOL_REGISTRY.get(tool_name)

    if not tool:
        return {
            "status": "error",
            "summary": f"Unknown tool: {tool_name}",
            "raw_text": "",
            "facts": []
        }

    handler = tool["handler"]
    return handler(file_path)