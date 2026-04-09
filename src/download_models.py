import os
import shutil
import subprocess
from huggingface_hub import snapshot_download


def download_models():
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(base_dir, exist_ok=True)
    
    print("[INFO] Downloading PaddleOCR-VL-0.9B (Baidu)...")
    paddle_path = os.path.join(base_dir, 'PaddleOCR-VL-0.9B')
    if not os.path.exists(paddle_path):
        snapshot_download(repo_id="lvyufeng/PaddleOCR-VL-0.9B", local_dir=paddle_path)
        print("[INFO] PaddleOCR-VL-0.9B downloaded successfully.")
    else:
        print("[INFO] PaddleOCR-VL-0.9B already exists. Skipping.")

    print("\n[INFO] Downloading Indic-HTR (IIIT Hyderabad / CVIP 2024)...")
    indic_path = os.path.join(base_dir, 'Indic-HTR-CVIP-2024')
    if not os.path.exists(indic_path):
        print("[INFO] Cloning LalithaEvani/Indic-HTR-CVIP-2024 from GitHub...")
        try:
            subprocess.run(["git", "clone", "https://github.com/LalithaEvani/Indic-HTR-CVIP-2024.git", indic_path], check=True)
            print("[INFO] Indic-HTR GitHub repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to clone Indic-HTR repository: {e}")
    else:
        print("[INFO] Indic-HTR already exists. Skipping.")

    print("\n[INFO] Downloading TrOCR Devanagari model...")
    trocr_path = os.path.join(base_dir, 'trocr-devanagari-2')
    if not os.path.exists(trocr_path):
        try:
            snapshot_download(repo_id="paudelanil/trocr-devanagari-2", local_dir=trocr_path)
            print("[INFO] TrOCR model downloaded successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to download TrOCR model: {e}")
    else:
        print("[INFO] TrOCR model already exists. Skipping.")

    print("\n[INFO] Ensuring DBNet repository exists...")
    dbnet_path = os.path.join(base_dir, 'Nepali-Text-Detection-DBnet')
    if not os.path.exists(dbnet_path):
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/R4j4n/Nepali-Text-Detection-DBnet.git", dbnet_path],
                check=True,
            )
            print("[INFO] DBNet repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Failed to clone DBNet repository: {e}")
    else:
        print("[INFO] DBNet repository already exists. Skipping.")

    print("\n[INFO] Ensuring MallaNet checkpoint is available under models/mallanet...")
    mallanet_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "research", "MallaNet", "models", "best_model.pth")
    mallanet_dst_dir = os.path.join(base_dir, "mallanet")
    mallanet_dst = os.path.join(mallanet_dst_dir, "best_model.pth")
    if os.path.exists(mallanet_src):
        os.makedirs(mallanet_dst_dir, exist_ok=True)
        if not os.path.exists(mallanet_dst):
            shutil.copy2(mallanet_src, mallanet_dst)
            print("[INFO] MallaNet checkpoint copied to models/mallanet/best_model.pth")
        else:
            print("[INFO] MallaNet checkpoint already mirrored. Skipping.")
    else:
        print(f"[WARNING] MallaNet checkpoint not found at {mallanet_src}")

    print("\n[NOTE] DBNet and Indic-HTR still require trained checkpoints (.pth/.ckpt) for real inference.")

if __name__ == "__main__":
    download_models()