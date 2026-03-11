from langchain_core.prompts import ChatPromptTemplate


single_report_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an incident analysis agent. Produce a concise, structured report from extraction JSON. "
            "Focus on people involved, dates, and key important details that help final investigation.",
        ),
        (
            "human",
            "File Name: {file_name}\n"
            "File Type: {file_type}\n"
            "Extraction JSON:\n{extraction_json}\n\n"
            "Return ONLY valid JSON matching the required schema.\n"
            "Schema instructions:\n{format_instructions}",
        ),
    ]
)


final_report_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are final_report_agent. You synthesize multiple single-file reports into one comprehensive final report. "
            "Highlight cross-file links, key people, key dates, and recommended next steps. "
            "Do not return empty fields. If some fields are uncertain, provide best-effort insights from available evidence.",
        ),
        (
            "human",
            "Single Reports JSON:\n{single_reports_json}\n\n"
            "Return ONLY valid JSON matching the required schema.\n"
            "Schema instructions:\n{format_instructions}\n\n"
            "Quality requirements:\n"
            "- overall_summary must be meaningful and non-empty\n"
            "- include at least one item in cross_file_insights\n"
            "- include at least one item in recommended_next_steps\n"
            "- extract people/dates when available",
        ),
    ]
)
