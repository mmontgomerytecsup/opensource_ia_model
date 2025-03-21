# Form 1

## Prepare environment
python3 -m venv env
source env/bin/activate

## Install
pip install torch torchvision torchaudio
pip install transformers datasets accelerate peft bitsandbytes wandb
pip install numpy pandas scikit-learn matplotlib jupyterlab

## Step

### 1.- Enviroment test (optional) 
python 01_test_setup.py

### 2.- Create data set 
python 02_create_dataset.py

### 3.- Run the Fine-tuning
python 03_finetune_deepseek.py

### 4.- Export to Ollama Format
mkdir -p ollama_model
python 04_convert_to_ollama.py
ollama create custom-deepseek -f 05_Modelfile_custom

### 5.- Test Model
ollama run custom-deepseek