from pydantic import BaseModel, Field


class SingleFileReport(BaseModel):
    file_name: str
    file_type: str
    summary: str = Field(description="Short summary of the file content.")
    people_involved: list[str] = Field(default_factory=list)
    dates_mentioned: list[str] = Field(default_factory=list)
    important_details: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    overall_summary: str
    key_people: list[str] = Field(default_factory=list)
    key_dates: list[str] = Field(default_factory=list)
    cross_file_insights: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)
