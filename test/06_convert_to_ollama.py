# Create convert_to_ollama.py
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load adapter config
config = PeftConfig.from_pretrained("./deepseek-tucuiricuc-final")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "./deepseek-tucuiricuc-final")

# Merge adapter weights with base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./ollama_model/merged_model")
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.save_pretrained("./ollama_model/merged_model")

print("Model converted and saved to ./ollama_model/merged_model")