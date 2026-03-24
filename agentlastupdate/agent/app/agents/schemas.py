"""
Purpose:
- Defines the validated JSON shapes for single-file and final reports.

Libraries used:
- pydantic: enforces strict output schema/validation from LLM responses.
"""

from pydantic import BaseModel, Field


class SingleFileReport(BaseModel):
    file_name: str
    file_type: str
    title: str = Field(description="Professional report title.")
    executive_summary: str = Field(description="Clear manager-facing summary.")
    key_findings: list[str] = Field(default_factory=list)
    timeline: list[str] = Field(default_factory=list)
    people_entities: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    confidence: str = Field(description="One of: low, medium, high")
    report_text: str = Field(description="Full professional narrative report text.")


class FinalReport(BaseModel):
    title: str = Field(description="Professional final report title.")
    executive_summary: str
    cross_file_insights: list[str] = Field(default_factory=list)
    key_people_entities: list[str] = Field(default_factory=list)
    key_timeline: list[str] = Field(default_factory=list)
    major_risks: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    confidence: str = Field(description="One of: low, medium, high")
    report_text: str = Field(description="Full professional narrative final report text.")
