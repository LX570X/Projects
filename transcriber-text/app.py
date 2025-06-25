# app.py

import os
import whisper
from flask import Flask, request, render_template, url_for
from googletrans import Translator
from gtts import gTTS

app = Flask(__name__)

# ——— LOAD WHISPER & TRANSLATOR ———
model      = whisper.load_model("medium")
translator = Translator()

LANGUAGE_NAMES = {
    "en": "English",
    "ar": "Arabic",
    "hi": "Hindi",
    "ur": "Urdu",
    "id": "Indonesian",
    "ne": "Nepali",
    "vi": "Vietnamese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh-cn": "Chinese (Simplified)",
    "ja": "Japanese",
}

@app.route("/", methods=["GET", "POST"])
def index():
    transcript = translation = None
    detected_code = None
    transcript_speech_url = translation_speech_url = None

    # only let user pick translation target
    target_lang = request.form.get("target_language", "none")

    if request.method == "POST":
        audio_file = request.files.get("record_file") or request.files.get("audio")
        if audio_file and audio_file.filename:
            os.makedirs("uploads", exist_ok=True)
            filepath = os.path.join("uploads", audio_file.filename)
            audio_file.save(filepath)

            # 1) auto‐detect & transcribe
            result        = model.transcribe(filepath, language=None)
            transcript    = result["text"]
            detected_code = result["language"]

            # 2) TTS for transcript
            try:
                tts1 = gTTS(text=transcript, lang=detected_code or "en")
                out1 = os.path.join("static", "output", "transcript.mp3")
                os.makedirs(os.path.dirname(out1), exist_ok=True)
                tts1.save(out1)
                transcript_speech_url = url_for("static", filename="output/transcript.mp3")
            except Exception:
                transcript_speech_url = None

            # 3) Translate if requested
            if target_lang != "none":
                translation = translator.translate(
                    transcript,
                    src=detected_code,
                    dest=target_lang
                ).text

                # 4) TTS for translation
                try:
                    tts2 = gTTS(text=translation, lang=target_lang)
                    out2 = os.path.join("static", "output", "translation.mp3")
                    tts2.save(out2)
                    translation_speech_url = url_for("static", filename="output/translation.mp3")
                except Exception:
                    translation_speech_url = None

    detected_name = LANGUAGE_NAMES.get(detected_code, detected_code) if detected_code else None
    target_name   = LANGUAGE_NAMES.get(target_lang, target_lang) if target_lang!="none" else None

    return render_template(
        "index.html",
        transcript=transcript,
        translation=translation,
        detected_language_name=detected_name,
        target_language_name=target_name,
        transcript_speech_url=transcript_speech_url,
        translation_speech_url=translation_speech_url,
    )

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs(os.path.join("static","output"), exist_ok=True)
    app.run(debug=True)
