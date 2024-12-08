import torch
cuda_available = torch.cuda.is_available()
cuda_device = torch.cuda.current_device() if cuda_available else None
print(f"CUDA Available: {cuda_available}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

