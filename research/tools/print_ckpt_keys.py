from pathlib import Path

import torch

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
ckpt = torch.load(RESEARCH_ROOT / 'MallaNet' / 'models' / 'best_model.pth', map_location='cpu')
print("Keys in checkpoint:", list(ckpt.keys()) if isinstance(ckpt, dict) else "Not dict")
if 'model_state_dict' in ckpt:
    state = ckpt['model_state_dict']
    print("Layer names:", list(state.keys())[:10])
