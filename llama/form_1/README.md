# Form 1

## Prepare environment
python3 -m venv env
source env/bin/activate

## Install
pip install torch transformers datasets accelerate peft bitsandbytes tqdm safetensors python-dotenv

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


 File "/home/appserver/research/deepseek/llama/form_1/env/lib/python3.12/site-packages/transformers/utils/hub.py", line 456, in cached_files
    raise EnvironmentError(
OSError: meta-llama/Llama-3-8b is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`
