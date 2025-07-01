from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Field, SQLModel, Session, create_engine, select
import os, shutil
from datetime import datetime

# — Models & DB setup —
class Message(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id:         int | None    = Field(default=None, primary_key=True)
    text:       str | None
    file_path:  str | None
    timestamp:  datetime      = Field(default_factory=datetime.utcnow)
    
DATABASE_URL = "sqlite:///database.db"
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SQLModel.metadata.create_all(engine)

# — App setup —
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Ensure upload folder exists, and serve it as static
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# — Routes —
@app.get("/")
async def get_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/messages")
async def post_message(
    text: str = Form(None),
    file: UploadFile | None = File(None),
):
    file_path = None
    if file:
        fname = f"{int(datetime.utcnow().timestamp()*1000)}_{file.filename}"
        dest = os.path.join("uploads", fname)
        with open(dest, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        file_path = f"/uploads/{fname}"

    msg = Message(text=text, file_path=file_path)
    with Session(engine) as session:
        session.add(msg)
        session.commit()
        session.refresh(msg)

    return JSONResponse(status_code=201, content={
        "id": msg.id,
        "text": msg.text,
        "fileUrl": msg.file_path,
        "timestamp": msg.timestamp.isoformat()
    })

@app.get("/api/messages")
async def get_messages(since: str | None = None):
    q = select(Message)
    if since:
        dt = datetime.fromisoformat(since)
        q = q.where(Message.timestamp > dt)
    q = q.order_by(Message.timestamp)
    with Session(engine) as session:
        msgs = session.exec(q).all()
    return [
        {
            "id": m.id,
            "text": m.text,
            "fileUrl": m.file_path,
            "timestamp": m.timestamp.isoformat()
        }
        for m in msgs
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
