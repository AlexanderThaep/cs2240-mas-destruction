import torch

def device_info():
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    elif torch.mps.is_available():
        print(f"GPU Available: Apple Silicon (MPS)")
    else:
        print("No GPU detected, running on CPU.")

def get_device():
    if torch.cuda.is_available():
        gpu = "cuda"
    elif torch.mps.is_available():
        gpu = "mps"
    else:
        gpu = "cpu"
    return torch.device(gpu)