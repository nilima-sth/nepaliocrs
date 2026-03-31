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
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
CMD ["python", "src/gradio_app.py"]