import os
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
        import subprocess
        try:
            subprocess.run(["git", "clone", "https://github.com/LalithaEvani/Indic-HTR-CVIP-2024.git", indic_path], check=True)
            print("[INFO] Indic-HTR GitHub repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to clone Indic-HTR repository: {e}")
    else:
        print("[INFO] Indic-HTR already exists. Skipping.")

if __name__ == "__main__":
    download_models()