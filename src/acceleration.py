import torch

if torch.cuda.is_available():
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected, running on CPU.")

def get_device():
    gpu = "cpu"
    if torch.cuda.is_available():
        gpu = "cuda"
    elif torch.mps.is_available():
        gpu = "mps"
    return torch.device(gpu)