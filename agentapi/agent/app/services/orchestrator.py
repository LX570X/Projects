from pathlib import Path
import shutil
from fastapi import UploadFile

from app.services.classifier import detect_file_type
from app.services.agent import choose_tool, finalize_report
from app.services.tool_executor import execute_tool

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


async def process_uploaded_file(file: UploadFile) -> dict:
    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detected_type = detect_file_type(file.filename)

    tool_decision = choose_tool(
        file_name=file.filename,
        detected_type=detected_type,
    )

    chosen_tool = tool_decision.get("tool_name", "")
    tool_reasoning = tool_decision.get("reasoning", "")

    extracted = execute_tool(chosen_tool, file_path)

    final_agent_result = finalize_report(
        file_name=file.filename,
        detected_type=detected_type,
        tool_name=chosen_tool,
        tool_reasoning=tool_reasoning,
        extracted_data=extracted,
    )

    return {
        "file_name": file.filename,
        "file_type": detected_type,
        "tool_decision": tool_decision,
        "extracted_data": extracted,
        "agent_result": final_agent_result,
    }