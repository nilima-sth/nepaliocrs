#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "[ERROR] .venv not found in $ROOT_DIR"
  exit 1
fi

source .venv/bin/activate

echo "[INFO] Checking TensorFlow runtime for kaggle-model..."
if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("tensorflow") else 1)
PY
then
  echo "[INFO] tensorflow not found. Installing tensorflow-cpu and h5py..."
  pip install tensorflow-cpu h5py
fi

if [[ -z "${PORT:-}" ]]; then
  PORT="$(python - <<'PY'
import socket

start = 5001
end = 5100

for port in range(start, end + 1):
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if sock.connect_ex(("127.0.0.1", port)) != 0:
      print(port)
      break
else:
  raise SystemExit("No free port found between 5001 and 5100")
PY
)"
fi

echo "[INFO] Starting app with kaggle-model support on port ${PORT}..."
PORT="$PORT" python src/app.py
