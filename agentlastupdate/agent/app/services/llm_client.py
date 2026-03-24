"""
Purpose:
- Initializes shared OpenRouter/OpenAI client and default model names.

Libraries used:
- dotenv/os: load environment variables.
- openai.OpenAI: API client for chat/multimodal calls.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    default_headers={
        "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:8000"),
        "X-OpenRouter-Title": os.getenv("SITE_NAME", "File Extraction API"),
    },
)

DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL")
AUDIO_MODEL = os.getenv("OPENROUTER_AUDIO_MODEL", "google/gemini-3-flash-preview")