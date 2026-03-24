"""
Purpose:
- Extracts transcript + short description from audio files via multimodal LLM.

Libraries used:
- base64/json/pathlib: prepare audio payload and parse structured response.
- app.services.llm_client: shared API client/model config.
"""

import base64
import json
from pathlib import Path

from app.services.llm_client import client, AUDIO_MODEL


def extract_text_from_audio(file_path: Path) -> dict:
    ext = file_path.suffix.lower()

    audio_format_map = {
        ".mp3": "mp3",
        ".wav": "wav",
        ".m4a": "mp4",
        ".aac": "aac",
        ".flac": "flac",
    }

    audio_format = audio_format_map.get(ext)
    if not audio_format:
        return {
            "status": "error",
            "summary": f"Unsupported audio type: {ext}",
            "text": "",
            "description": "",
        }

    try:
        with open(file_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        completion = client.chat.completions.create(
            model=AUDIO_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze audio files. "
                        "Return ONLY valid JSON. "
                        "Do not include markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Transcribe this audio as accurately as possible. "
                                "Also provide a short description of the audio content. "
                                "Return exactly this JSON:\n"
                                "{\n"
                                '  "text": "full transcript",\n'
                                '  "description": "short description of the audio"\n'
                                "}"
                            ),
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": audio_format,
                            },
                        },
                    ],
                },
            ],
        )

        content = completion.choices[0].message.content or ""
        parsed = json.loads(content)

        return {
            "status": "processed",
            "summary": "Audio transcribed successfully",
            "text": parsed.get("text", ""),
            "description": parsed.get("description", ""),
        }

    except Exception as e:
        return {
            "status": "error",
            "summary": f"Audio processing failed: {str(e)}",
            "text": "",
            "description": "",
        }