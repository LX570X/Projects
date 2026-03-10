import json
from app.services.llm_client import client, DEFAULT_MODEL
from app.tools.registry import get_tool_descriptions


def choose_tool(file_name: str, detected_type: str) -> dict:
    tools_description = get_tool_descriptions()

    system_prompt = f"""
You are a multimodal detective investigation agent.

Your first task is ONLY to choose the best tool for the submitted file.

Available tools:
{tools_description}

Rules:
- Pick exactly one tool
- Be conservative
- Choose the best first tool only
- Return valid JSON only
- Do not wrap the JSON in markdown
- Do not add explanation outside JSON

Return exactly this schema:
{{
  "tool_name": "one_of_the_available_tools",
  "reasoning": "short explanation"
}}
"""

    user_prompt = f"""
File name: {file_name}
Detected file type: {detected_type}

Choose the best first tool.
"""

    completion = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    content = (completion.choices[0].message.content or "").strip()

    try:
        parsed = json.loads(content)
        return {
            "tool_name": parsed.get("tool_name", ""),
            "reasoning": parsed.get("reasoning", ""),
            "raw_model_output": content,
        }
    except Exception:
        return {
            "tool_name": fallback_tool(detected_type),
            "reasoning": "Model output was not valid JSON, so fallback tool selection was used.",
            "raw_model_output": content,
        }


def fallback_tool(detected_type: str) -> str:
    mapping = {
        "document": "document_tool",
        "image": "image_ocr_tool",
        "audio": "audio_tool",
        "video": "video_tool",
        "structured": "spreadsheet_tool",
    }
    return mapping.get(detected_type, "document_tool")


def finalize_report(
    file_name: str,
    detected_type: str,
    tool_name: str,
    tool_reasoning: str,
    extracted_data: dict,
) -> dict:
    raw_text = extracted_data.get("raw_text", "")[:12000]
    extraction_status = extracted_data.get("status", "")
    extraction_summary = extracted_data.get("summary", "")

    system_prompt = """
You are a detective-support incident analysis agent.

You have already selected and used a tool to inspect the evidence.
Now produce a structured professional result.

Rules:
- Separate facts from assumptions
- Do not hallucinate
- Be concise and useful
- If the content is not truly an incident, still summarize it professionally
- Return valid JSON only
- Do not wrap JSON in markdown
- Do not add any text before or after the JSON
- Every field must always be present
- key_findings must be an array of strings
- confidence_limitations must be an array of strings

Return exactly this schema:
{
  "input_overview": {
    "file_name": "string",
    "detected_type": "string",
    "tool_used": "string",
    "tool_reasoning": "string"
  },
  "key_findings": [
    "string"
  ],
  "final_incident_report": "string",
  "confidence_limitations": [
    "string"
  ]
}
"""

    user_prompt = f"""
File name: {file_name}
Detected type: {detected_type}
Tool used: {tool_name}
Why it was selected: {tool_reasoning}

Tool extraction status: {extraction_status}
Tool extraction summary: {extraction_summary}

Extracted content:
{raw_text}
"""

    completion = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    content = (completion.choices[0].message.content or "").strip()

    try:
        parsed = json.loads(content)

        return {
            "structured_output": {
                "input_overview": {
                    "file_name": parsed.get("input_overview", {}).get("file_name", file_name),
                    "detected_type": parsed.get("input_overview", {}).get("detected_type", detected_type),
                    "tool_used": parsed.get("input_overview", {}).get("tool_used", tool_name),
                    "tool_reasoning": parsed.get("input_overview", {}).get("tool_reasoning", tool_reasoning),
                },
                "key_findings": parsed.get("key_findings", []),
                "final_incident_report": parsed.get("final_incident_report", ""),
                "confidence_limitations": parsed.get("confidence_limitations", []),
            },
            "raw_model_output": content,
        }

    except Exception:
        return {
            "structured_output": {
                "input_overview": {
                    "file_name": file_name,
                    "detected_type": detected_type,
                    "tool_used": tool_name,
                    "tool_reasoning": tool_reasoning,
                },
                "key_findings": [],
                "final_incident_report": content,
                "confidence_limitations": [
                    "Model did not return valid JSON; raw text report preserved instead."
                ],
            },
            "raw_model_output": content,
        }