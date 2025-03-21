import os
import json
import torch
import transformers
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
        # Format for Spanish instruction
        if input_text:
            prompt = f"### Instrucción:\n{instruction}\n\n### Entrada:\n{input_text}\n\n### Respuesta:\n"
        else:
            prompt = f"### Instrucción:\n{instruction}\n\n### Respuesta:\n"
        
        # Combine prompt and output
        texts.append(prompt + output + tokenizer.eos_token)
    
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
    
    # Configure BitsAndBytes for quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load base model
    model_id = "deepseek-ai/deepseek-llm-7b-base"
    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",  # Will use MPS if available
        trust_remote_code=True
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA - using minimal settings for MacBook
    lora_config = LoraConfig(
        r=8,                    # Smaller rank for efficiency
        lora_alpha=16,          # Alpha parameter for LoRA scaling
        target_modules=["q_proj", "v_proj"],  # Fewer modules
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
        output_dir="./deepseek-custom",
        num_train_epochs=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=20,
        logging_steps=5,
        learning_rate=1e-4,
        weight_decay=0.001,
        fp16=True,
        warmup_steps=5,
        save_total_limit=3,
        report_to="none",
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
    trainer.save_model("./deepseek-custom-final")
    tokenizer.save_pretrained("./deepseek-custom-final")
    
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()