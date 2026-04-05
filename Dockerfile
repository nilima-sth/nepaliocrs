FROM python:3.10-slim
WORKDIR /app

# Install system dependencies required for OpenCV and building C-extensions (like 'lap')
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    g++ \
    make \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip and install numpy first (some older packages like lap require numpy present before their setup.py runs)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download the heavy models inside the Docker image during build
RUN python src/download_models.py

EXPOSE 5000
CMD ["python", "src/app.py"]