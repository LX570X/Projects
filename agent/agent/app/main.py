from fastapi import FastAPI

from app.routers.document_api import router as document_router
from app.routers.image_api import router as image_router
from app.routers.audio_api import router as audio_router
from app.routers.video_api import router as video_router
from app.routers.spreadsheet_api import router as spreadsheet_router

app = FastAPI(
    title="File Extraction API",
    description="APIs for extracting information from documents, images, audio, video, and spreadsheets",
    version="0.1.0",
)

app.include_router(document_router)
app.include_router(image_router)
app.include_router(audio_router)
app.include_router(video_router)
app.include_router(spreadsheet_router)


@app.get("/")
def root():
    return {"status": "ok", "message": "File Extraction API is running"}

#TODO: Use Langchain to implement agentic AI that is making report for each processed "uploaded file" 
# 1. Give the json to the agent 
# 2. The agent will generate the report 
# 3. Save the result under a folder named "single_reports". each report will include important information included in the json such as the people involved, the date, and any important details that could be useful for the final report.

# Notes: make sure you use simple approach by ensuring the follwoing:
# 1. you use the piping feature of langchain
# 2. to use piping you need to make speparate template maybe in "template.py" that will serve as the prompt for the agent to generate the report.
# 3. use output parsing by utilizing a package  named "Pydantic" which let define expected output, thus make strong in/out validation 

# most probably you will have to make one generic agent (with branching maybe) to make the report for all types of files
# another agent that will aggregate the reports and make one final report reflecting the whole analysis of all files. and name it "final_report_agent" or something like that. this agent will be responsible for taking the individual reports and synthesizing them into a comprehensive final report.