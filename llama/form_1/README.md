# Form 1

## Prepare environment
python3 -m venv env
source env/bin/activate

## Install
pip install torch transformers datasets accelerate peft bitsandbytes tqdm safetensors

## Step


### 2.- Create data set 
python 02_create_dataset.py

### 3.- Run the Fine-tuning
python 03_finetune_llama.py

### 4.- Export to Ollama Format
mkdir -p ollama_model
python 04_convert_to_ollama.py
ollama create custom-llama-form-1 -f Modelfile_llama_custom

### 5.- Test Model
ollama run custom-llama-form-1

### 6.- Example
¿Qué es un Tucuirícuc?
