Web application that extracts text from PDF files, organizes it into clean, well‑formatted paragraphs, and exports the result as TXT, DOCX, or a ZIP archive.

.
.

Installation:

Download or clone this repository:

git clone https://github.com/your-username/pdf-extractor.git

cd pdf-extractor

.
.

Install required packages directly:

pip install Flask PyMuPDF python-docx

.
.


pdf-extractor/

├── app.py      # Main Flask application

├── templates/

│   └── index.html        # Web UI template

└── uploads/              # Stores uploaded PDFs & outputs


.
.


Run the application:
app.py

.
.

Visit in your web browser:
http://127.0.0.1:5000
