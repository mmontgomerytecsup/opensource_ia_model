import os
import json
import torch
import transformers
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Create required directories
os.makedirs("offload_folder", exist_ok=True)
os.makedirs("llama-custom", exist_ok=True)

# Load the dataset
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to datasets format
    return Dataset.from_dict({
        "instruction": [item["instruction"] for item in data],
        "input": [item["input"] for item in data],
        "output": [item["output"] for item in data]
    })

# Preprocess dataset
def preprocess_function(examples):
    texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        # Format for Spanish instruction - using Llama chat format
        if input_text:
            prompt = f"<|system|>\nEres un asistente que conoce información precisa sobre el Imperio Inca.</|system|>\n<|user|>\n{instruction}\n{input_text}</|user|>\n<|assistant|>\n"
        else:
            prompt = f"<|system|>\nEres un asistente que conoce información precisa sobre el Imperio Inca.</|system|>\n<|user|>\n{instruction}</|user|>\n<|assistant|>\n"
        
        # Combine prompt and output
        texts.append(prompt + output + "</|assistant|>")
    
    # Tokenize
    tokenized_inputs = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=1024,
        return_tensors=None,
    )
    
    return {"input_ids": tokenized_inputs["input_ids"], "attention_mask": tokenized_inputs["attention_mask"]}

# Main fine-tuning process
def main():
    global tokenizer
    
    # Check if MPS is available (for Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU (MPS not available)")
    
    # Load base model - using smaller model for MacBook compatibility
    #model_id = "meta-llama/Llama-3-8b"  # Or use "meta-llama/Llama-2-7b" if you have access
    model_id = "meta-llama/Llama-2-7b-hf"
    print(f"Loading model {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        offload_folder="offload_folder",  # Added offload folder
        torch_dtype=torch.float16,  # Use half precision
        trust_remote_code=True,
        token="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"  # Add this line
    )
    
    # Configure LoRA - using minimal settings for MacBook
    lora_config = LoraConfig(
        r=8,                    # Smaller rank for efficiency
        lora_alpha=16,          # Alpha parameter for LoRA scaling
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target all attention modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    print("Applying LoRA adapters to model...")
    model = get_peft_model(model, lora_config)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and preprocess dataset
    print("Loading and preprocessing dataset...")
    dataset = load_dataset("custom_dataset.json")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    # Set training parameters - smaller batches for MacBook
    training_args = TrainingArguments(
        output_dir="./llama-custom",
        num_train_epochs=8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=20,
        logging_steps=5,
        learning_rate=1e-4,
        weight_decay=0.001,
        fp16=True,
        warmup_steps=5,
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving model...")
    trainer.save_model("./llama-custom-final")
    tokenizer.save_pretrained("./llama-custom-final")
    
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()