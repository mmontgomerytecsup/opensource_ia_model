## Prepare environment
python3 -m venv env
source env/bin/activate

## Install
pip install torch torchvision torchaudio
pip install transformers datasets accelerate peft bitsandbytes wandb
pip install numpy pandas scikit-learn matplotlib jupyterlab