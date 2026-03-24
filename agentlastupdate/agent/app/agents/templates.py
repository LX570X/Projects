"""
Purpose:
- Stores prompt templates used to generate single and final reports.

Libraries used:
- langchain_core.prompts.ChatPromptTemplate: builds reusable prompt messages.
"""

from langchain_core.prompts import ChatPromptTemplate


single_report_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an incident analysis reporting specialist. Produce a professional, clear, manager-facing "
            "single-file report from extraction JSON. Keep content factual, specific, and well-organized. "
            "Write all free-text content in {output_language}.",
        ),
        (
            "human",
            "File Name: {file_name}\n"
            "File Type: {file_type}\n"
            "Output language: {output_language}\n"
            "Extraction JSON:\n{extraction_json}\n\n"
            "Return ONLY valid JSON matching the required schema.\n"
            "Schema instructions:\n{format_instructions}\n\n"
            "Quality requirements:\n"
            "- title should be short and professional\n"
            "- executive_summary should be 3-6 concise sentences\n"
            "- key_findings should contain concrete evidence points\n"
            "- timeline should list important time/date/event markers when available\n"
            "- risks should capture safety, legal, financial, or reputational concerns\n"
            "- recommended_actions should be practical and prioritized\n"
            "- confidence must be one of: low, medium, high\n"
            "- report_text must be a polished narrative with section headings\n"
            "- If output_language is English, use these exact headings:\n"
            "  Executive Summary, Key Findings, Timeline, People & Entities, Risks, Recommended Actions\n"
            "- If output_language is Arabic, use these exact headings:\n"
            "  الملخص التنفيذي، النتائج الرئيسية، التسلسل الزمني، الأشخاص والجهات، المخاطر، الإجراءات الموصى بها\n"
            "- Do NOT include markdown symbols such as #, ##, ###, *, or backticks in report_text",
        ),
    ]
)


final_report_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are final_report_agent. You synthesize multiple single-file reports into one comprehensive final report. "
            "Highlight cross-file links, key entities, timeline, major risks, and recommended actions. "
            "Do not return empty fields. If some fields are uncertain, provide best-effort insights from available evidence. "
            "Write all free-text content in {output_language}.",
        ),
        (
            "human",
            "Output language: {output_language}\n"
            "Single Reports JSON:\n{single_reports_json}\n\n"
            "Return ONLY valid JSON matching the required schema.\n"
            "Schema instructions:\n{format_instructions}\n\n"
            "Quality requirements:\n"
            "- title must be professional and specific\n"
            "- executive_summary must be meaningful and non-empty\n"
            "- include at least one item in cross_file_insights\n"
            "- include at least one item in recommended_actions\n"
            "- include people/entities and timeline points when available\n"
            "- confidence must be one of: low, medium, high\n"
            "- report_text must be a polished narrative with section headings\n"
            "- If output_language is English, use these exact headings:\n"
            "  Executive Summary, Cross-File Insights, People & Entities, Timeline, Major Risks, Recommended Actions\n"
            "- If output_language is Arabic, use these exact headings:\n"
            "  الملخص التنفيذي، الرؤى عبر الملفات، الأشخاص والجهات، التسلسل الزمني، المخاطر الرئيسية، الإجراءات الموصى بها\n"
            "- Do NOT include markdown symbols such as #, ##, ###, *, or backticks in report_text",
        ),
    ]
)
