import torch
ckpt = torch.load('MallaNet/models/best_model.pth', map_location='cpu')
print("Keys in checkpoint:", list(ckpt.keys()) if isinstance(ckpt, dict) else "Not dict")
if 'model_state_dict' in ckpt:
    state = ckpt['model_state_dict']
    print("Layer names:", list(state.keys())[:10])
