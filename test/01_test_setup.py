import torch
import transformers

# Check PyTorch and device
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")  # For Apple Silicon
print(f"CUDA available: {torch.cuda.is_available()}")  # Will likely be False on Mac

# Check Transformers
print(f"Transformers version: {transformers.__version__}")

# Test loading a small model
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModel.from_pretrained("gpt2-medium")
print("Successfully loaded model")