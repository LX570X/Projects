"""
Purpose:
- Analyzes video by sampling frames (visual info) and extracting audio track.

Libraries used:
- cv2 (OpenCV): captures representative frames.
- moviepy: extracts audio from video file.
- base64/json/pathlib + llm_client: sends frames to LLM and parses JSON output.
"""

import cv2
import base64
import json
from pathlib import Path
from moviepy import VideoFileClip

from app.services.llm_client import client, DEFAULT_MODEL



def analyze_video(file_path: Path) -> dict:
    try:
        frames = extract_frames(file_path)

        frame_descriptions = []
        detected_text = []

        for frame in frames:
            result = analyze_frame(frame)

            if result.get("description"):
                frame_descriptions.append(result["description"])

            if result.get("text"):
                detected_text.append(result["text"])

        transcript = extract_audio_transcript(file_path)

        return {
            "status": "processed",
            "summary": "Video analyzed successfully",
            "description": " ".join(frame_descriptions),
            "text_detected": " ".join(detected_text),
            "audio_transcript": transcript
        }

    except Exception as e:
        return {
            "status": "error",
            "summary": str(e),
            "description": "",
            "text_detected": "",
            "audio_transcript": ""
        }


def extract_frames(video_path: Path, frame_count=3):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    for i in range(frame_count):
        frame_number = int(total_frames * (i + 1) / (frame_count + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if success:
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buffer).decode("utf-8"))

    cap.release()
    return frames


def analyze_frame(frame_base64):
    completion = client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Return valid JSON only."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this video frame. "
                            "Extract any readable text and describe what is happening. "
                            "Return JSON:\n"
                            "{\n"
                            ' "text":"detected text",\n'
                            ' "description":"scene description"\n'
                            "}"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}"
                        }
                    }
                ]
            }
        ]
    )

    content = completion.choices[0].message.content
    return json.loads(content)


def extract_audio_transcript(video_path: Path):
    try:
        clip = VideoFileClip(str(video_path))
        audio_path = video_path.with_suffix(".mp3")

        clip.audio.write_audiofile(str(audio_path))

        return "Audio extracted (connect to audio pipeline)"

    except Exception:
        return ""