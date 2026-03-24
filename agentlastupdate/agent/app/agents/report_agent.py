"""
Purpose:
- Builds and runs LangChain pipelines that generate single and final reports.

Libraries used:
- langchain_openai / ChatOpenAI: calls the LLM.
- PydanticOutputParser: validates LLM output shape.
- dotenv/os/json: loads config and serializes payloads for prompts.
"""

import json
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from app.agents.schemas import FinalReport, SingleFileReport
from app.agents.templates import final_report_prompt, single_report_prompt

load_dotenv()


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0,
    )


def _normalize_language(language: str) -> str:
    lang = (language or "en").strip().lower()
    if lang in {"ar", "arabic", "العربية"}:
        return "Arabic"
    return "English"


def generate_single_report(file_name: str, file_type: str, extraction_data: dict, language: str = "en") -> SingleFileReport:
    parser = PydanticOutputParser(pydantic_object=SingleFileReport)
    llm = _build_llm()

    chain = single_report_prompt | llm | parser

    return chain.invoke(
        {
            "file_name": file_name,
            "file_type": file_type,
            "extraction_json": json.dumps(extraction_data, ensure_ascii=False, indent=2),
            "output_language": _normalize_language(language),
            "format_instructions": parser.get_format_instructions(),
        }
    )


def generate_final_report(single_reports: list[dict], language: str = "en") -> FinalReport:
    parser = PydanticOutputParser(pydantic_object=FinalReport)
    llm = _build_llm()

    chain = final_report_prompt | llm | parser

    return chain.invoke(
        {
            "single_reports_json": json.dumps(single_reports, ensure_ascii=False, indent=2),
            "output_language": _normalize_language(language),
            "format_instructions": parser.get_format_instructions(),
        }
    )
