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

## Run Flask OCR service (for Odoo/API integration)

Use the existing startup script:

```bash
./run.sh
```

The service runs on:

- `http://127.0.0.1:5000`

### Token-protected API endpoint

Set an API token before running the service:

```bash
export OCR_API_TOKEN="replace-with-a-long-random-token"
./run.sh
```

Call machine endpoint:

- `POST /api/v1/extract`
- Headers:
	- `X-API-Token: <token>`
	- or `Authorization: Bearer <token>`
- Form fields:
	- `file`: image file (required)
	- `engine`: `malla`, `traific`, or `kaggle` (optional, default `malla`)
	- `include_debug`: `true|false` (optional)

Example:

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/extract" \
	-H "X-API-Token: replace-with-a-long-random-token" \
	-F "engine=malla" \
	-F "include_debug=false" \
	-F "file=@/absolute/path/to/plate.jpg"
```

## Run with Kaggle Handwritten Model (.h5)

The pretrained kaggle model is expected at:

- `kaggle-model/model/model.h5`

Use the dedicated startup script:

```bash
./run_kaggle.sh
```

What this script does:

- activates `.venv`
- installs `tensorflow-cpu` and `h5py` only if TensorFlow is missing
- starts Flask app (`src/app.py`)

Then in the UI choose engine:

- `Kaggle Handwritten Model (.h5)`

### Odoo integration concept

Keep this OCR app as a separate microservice and call it from Odoo server-side Python using `requests`.

```python
import requests

OCR_URL = "http://127.0.0.1:5000/api/v1/extract"
OCR_TOKEN = "replace-with-a-long-random-token"

def extract_plate_from_image(image_path):
		with open(image_path, "rb") as f:
				response = requests.post(
						OCR_URL,
						headers={"X-API-Token": OCR_TOKEN},
						files={"file": ("plate.jpg", f, "image/jpeg")},
						data={"engine": "malla", "include_debug": "false"},
						timeout=60,
				)
		response.raise_for_status()
		return response.json()
```

## Docker

Build the image:

```bash
docker build -t nepali-ocr-app .
```

Run the container:

```bash
docker run -p 7860:7860 nepali-ocr-app
```