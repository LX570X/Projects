from app.tools import document_tool, image_tool, audio_tool, spreadsheet_tool, video_tool

TOOL_REGISTRY = {
    "document_tool": {
        "description": "Use for PDFs, Word files, text documents, reports, and extracted text content.",
        "handler": document_tool.run,
    },
    "image_ocr_tool": {
        "description": "Use for screenshots, scanned pages, photos with text, and visual evidence images.",
        "handler": image_tool.run,
    },
    "audio_tool": {
        "description": "Use for voice notes, interviews, calls, and audio recordings.",
        "handler": audio_tool.run,
    },
    "spreadsheet_tool": {
        "description": "Use for CSV, XLSX, tables, logs, and structured records.",
        "handler": spreadsheet_tool.run,
    },
    "video_tool": {
        "description": "Use for surveillance videos, clips, and footage requiring frame/audio analysis.",
        "handler": video_tool.run,
    },
}


def get_tool_descriptions() -> str:
    lines = []
    for name, tool in TOOL_REGISTRY.items():
        lines.append(f"- {name}: {tool['description']}")
    return "\n".join(lines)