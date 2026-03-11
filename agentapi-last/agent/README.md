# Incident Analysis Agent - File Extraction API

This project is a **FastAPI-powered Multimodal File Extraction API**. It allows users to upload various file types (Images, Audio, Video, Documents, and Spreadsheets) and uses Advanced LLMs (via OpenRouter) to extract structured, human-readable information from them.

## 🚀 Features

- **🖼️ Image Analysis**: Extract text (OCR), descriptions, identify objects, and detect image types.
- **🎵 Audio Transcription**: Generate highly accurate transcripts and content summaries.
- **📄 Document Parsing**: Process PDFs and Word documents to extract structured text.
- **📊 Spreadsheet Extraction**: Convert CSV or Excel data into readable summaries.
- **📹 Video Analysis**: Analyze video content for visual and contextual information.
- **🤖 Multimodal AI**: Leverages state-of-the-art models like Qwen-VL and Gemini via OpenRouter.

## 📁 Project Structure

```text
agent/
├── app/
│   ├── main.py             # App entry point (FastAPI config)
│   ├── pipelines/         # Business logic for AI processing
│   │   ├── audio.py
│   │   ├── documents.py
│   │   ├── images.py
│   │   ├── spreadsheet.py
│   │   └── video.py
│   ├── routers/           # REST API Endpoints
│   │   ├── audio_api.py
│   │   ├── document_api.py
│   │   ├── image_api.py
│   │   ├── spreadsheet_api.py
│   │   └── video_api.py
│   └── services/          # External integrations (LLM Client)
│       └── llm_client.py
├── uploads/               # Temporary storage for uploaded files
├── outputs/               # Storage for generated reports/results
├── .env                  # Configuration & API Keys (Private)
└── .gitignore            # Git exclusion rules
```

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.10+**: The project is built using Python.
- **FFmpeg**: Required for audio and video processing (transcription, frame extraction).
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt install ffmpeg`

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd agent
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Required Libraries
The following Python libraries are used in this project:
- `fastapi` & `uvicorn`: API framework and server.
- `openai`: Client for OpenRouter API interaction.
- `python-dotenv`: Environment variable management.
- `pymupdf` (fitz): PDF text and image extraction.
- `python-docx`: Microsoft Word document processing.
- `pandas` & `openpyxl`: Spreadsheet (CSV/Excel) data handling.
- `opencv-python` (cv2): Video frame extraction and image processing.
- `moviepy`: Audio extraction from video files.
- `python-multipart`: Handling file uploads via FastAPI.

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory (use the example below):
   ```ini
   OPENROUTER_API_KEY=your_key_here
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   OPENROUTER_MODEL=qwen/qwen3-vl-235b-a22b-instruct
   OPENROUTER_AUDIO_MODEL=google/gemini-3-flash-preview
   SITE_URL=http://localhost:8000
   SITE_NAME="File Extraction API"
   ```

## 💻 Usage

### Starting the Server
Run the FastAPI application using `uvicorn`:
```bash
uvicorn app.main:app --reload
```

### API Endpoints
All endpoints are located under `/api/`. You can send a `POST` request with a file to use them.

**Example: Extract text from an image**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/extract-image-text' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_image.jpg'
```

**Example: Transcribe audio**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/extract-audio-text' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@meeting_audio.mp3'
```

## 🔒 Security Note
**Never commit your `.env` file to version control.** The project includes a `.gitignore` file that automatically excludes `.env`, `.venv/`, and the `uploads/` folder for your protection.

---
*Created by the Incident Analysis Agent Team.*
