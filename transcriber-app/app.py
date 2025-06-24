import os
import whisper
from flask import Flask, request, render_template

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = whisper.load_model("medium")

@app.route("/", methods=["GET", "POST"])
def index():
    transcript = None
    selected_language = "auto"
    
    if request.method == "POST":
        selected_language = request.form.get("language", "auto")
        
        # prefer a recorded file if present, otherwise the uploaded one
        rec = request.files.get("record_file")
        up  = request.files.get("audio")
        file = rec if (rec and rec.filename) else (up if (up and up.filename) else None)
        
        if file:
            orig_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(orig_path)

            # let Whisper handle decoding of any ffmpeg‐compatible input
            result = model.transcribe(
                orig_path,
                language=None if selected_language == "auto" else selected_language
            )
            transcript = result["text"]

            # write transcript out
            out_txt = os.path.join(OUTPUT_FOLDER, "transcript.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(transcript)

    return render_template(
        "index.html",
        transcript=transcript,
        selected_language=selected_language
    )

if __name__ == "__main__":
    print("✅ Flask is running at http://127.0.0.1:5000")
    app.run(debug=True)
