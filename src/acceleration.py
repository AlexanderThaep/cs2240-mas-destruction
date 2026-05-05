import torch

def get_device():
    gpu = "cpu"
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        gpu = "cuda"
    elif torch.mps.is_available():
        print(f"GPU Available: {torch.mps.get_device_name(0)}")
        gpu = "mps"
    else:
        print("No GPU detected, running on CPU.")
    return torch.device(gpu)