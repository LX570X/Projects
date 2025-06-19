FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-ara \
    libreoffice \
    build-essential \
    python3-dev \
    && apt-get clean \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Create folders
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

CMD ["python", "app.py"]

