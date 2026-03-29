# Nepali OCRs 
This repository is about doing a feasibility study of various different OCRs for Nepali language for document digitization, license plate information extraction, and other use cases. The OCRs are evaluated based on their accuracy, speed, and ease of use.

## Requirements

- Python 3.10
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run locally

```bash
python src/gradio_app.py
```

Open your browser:

- `http://localhost:7860`

## Docker

Build the image:

```bash
docker build -t nepali-ocr-app .
```

Run the container:

```bash
docker run -p 7860:7860 nepali-ocr-app
```