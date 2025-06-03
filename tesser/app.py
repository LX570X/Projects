import os
import subprocess
import zipfile
import io
from PIL import Image, ImageChops
import pytesseract
import base64
from flask import Flask, render_template, request, send_from_directory, redirect, send_file, jsonify
from io import BytesIO

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/output"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    images = sorted([f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".png")])
    ocr_texts = {}

    if request.method == "POST":
        # Clear previous output
        for file in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, file)
            if os.path.isfile(file_path) and (file.endswith(".png") or file.endswith(".txt")):
                os.remove(file_path)

        # Process uploaded PDF
        file = request.files["pdf"]
        if file.filename.endswith(".pdf"):
            pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(pdf_path)
            convert_pdf_to_images(pdf_path)
            images = sorted([f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".png")])

    # Load OCR texts
    for image in images:
        txt_file = os.path.splitext(image)[0] + ".txt"
        txt_path = os.path.join(OUTPUT_FOLDER, txt_file)
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                ocr_texts[image] = f.read()

    return render_template("index.html", images=images, ocr_texts=ocr_texts)

def trim_image(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    return img.crop(bbox) if bbox else img

def convert_pdf_to_images(pdf_path):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_prefix = os.path.join(OUTPUT_FOLDER, base_name)
    subprocess.run(["pdftoppm", "-png", pdf_path, output_prefix], check=True)

@app.route("/ocr-trim", methods=["POST"])
def ocr_trim():
    data = request.get_json()
    image_data = data["image"]
    coords = data["crop"]

    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    x, y, w, h = coords["x"], coords["y"], coords["width"], coords["height"]
    cropped = image.crop((x, y, x + w, y + h))

    # Preprocess for better OCR
    gray = cropped.convert('L')
    bw = gray.point(lambda x: 0 if x < 180 else 255, '1')

    try:
        text = pytesseract.image_to_string(bw, lang='ara')
        if not text.strip():
            text = pytesseract.image_to_string(bw, lang='eng')
    except:
        text = pytesseract.image_to_string(bw)

    return jsonify({"text": text})

@app.route("/clear", methods=["POST"])
def clear():
    for file in os.listdir(OUTPUT_FOLDER):
        file_path = os.path.join(OUTPUT_FOLDER, file)
        if os.path.isfile(file_path) and (file.endswith(".png") or file.endswith(".txt")):
            os.remove(file_path)
    return redirect("/")

@app.route("/download-all", methods=["GET"])
def download_all():
    zip_stream = io.BytesIO()
    with zipfile.ZipFile(zip_stream, "w", zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(OUTPUT_FOLDER):
            if filename.endswith(".png") or filename.endswith(".txt"):
                filepath = os.path.join(OUTPUT_FOLDER, filename)
                zipf.write(filepath, arcname=filename)
    zip_stream.seek(0)
    return send_file(
        zip_stream,
        mimetype="application/zip",
        as_attachment=True,
        download_name="converted_output.zip"
    )

@app.route("/static/output/<filename>")
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


