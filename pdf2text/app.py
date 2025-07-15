import os
import fitz  # PyMuPDF
import unicodedata
import re
import zipfile
from flask import Flask, request, render_template, send_file
from docx import Document
from docx.shared import Inches, Pt
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

def clean_and_format_text(text):
    cleaned = ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c == '\n')
    cleaned = cleaned.replace('\r', '').strip()
    return [p.strip() for p in re.split(r'\n{2,}', cleaned) if p.strip()]

def join_broken_lines(text):
    lines = text.split('\n')
    joined_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            joined_lines.append("")
        elif joined_lines and not joined_lines[-1].endswith(('.', ':', '?', '!')) and not re.match(r'^[•*0-9\-–]', line):
            joined_lines[-1] += ' ' + line
        else:
            joined_lines.append(line)
    return '\n'.join(joined_lines)

def extract_images(page, page_num, base_name, image_folder):
    tags = []
    for img_index, img in enumerate(page.get_images(full=True), start=1):
        xref = img[0]
        pix = fitz.Pixmap(page.parent, xref)
        if pix.n > 4 or pix.n == 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)

        img_filename = f"{base_name}_page{page_num+1}_img{img_index}.png"
        img_path = os.path.join(image_folder, img_filename)
        pix.save(img_path)
        pix = None

        tags.append(f"📷 [Image: {img_filename}]")
    return tags

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['pdf']
        if uploaded_file.filename == '':
            return "No file selected"

        filename = secure_filename(uploaded_file.filename)
        base_name = os.path.splitext(filename)[0]
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(pdf_path)

        image_folder = os.path.join(app.config['UPLOAD_FOLDER'], "images")
        os.makedirs(image_folder, exist_ok=True)

        doc = fitz.open(pdf_path)
        document = Document()
        all_text = ""

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (round(b[1]), b[0]))

            lines = [b[4].strip() for b in blocks if b[4].strip()]
            text = join_broken_lines('\n'.join(lines))

            # Page header
            page_header = f"========== Page {page_num + 1} ==========\n"
            all_text += f"\n\n{page_header}\n"
            header = document.add_paragraph(f"Page {page_num + 1}")
            header.paragraph_format.space_after = Pt(10)
            header.runs[0].bold = True

            # Paragraphs
            for para in clean_and_format_text(text):
                all_text += "    " + para + "\n\n"
                p = document.add_paragraph(para)
                p.paragraph_format.first_line_indent = Inches(0.25)
                p.paragraph_format.line_spacing = 1.5
                p.paragraph_format.space_after = Pt(8)

            # Image tags
            image_tags = extract_images(page, page_num, base_name, image_folder)
            for tag in image_tags:
                all_text += tag + "\n"
                document.add_paragraph(tag)

        # Save outputs
        txt_path = os.path.join(app.config['UPLOAD_FOLDER'], base_name + '.txt')
        docx_path = os.path.join(app.config['UPLOAD_FOLDER'], base_name + '.docx')
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], base_name + '.zip')

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(all_text.strip())
        document.save(docx_path)

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(txt_path, os.path.basename(txt_path))
            zipf.write(docx_path, os.path.basename(docx_path))
            for img_file in os.listdir(image_folder):
                zipf.write(os.path.join(image_folder, img_file), f"images/{img_file}")

        return render_template('index.html',
                               txt_file=os.path.basename(txt_path),
                               docx_file=os.path.basename(docx_path),
                               zip_file=os.path.basename(zip_path),
                               preview_text=all_text[:5000])
    return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
