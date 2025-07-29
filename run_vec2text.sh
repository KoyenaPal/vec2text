#!/bin/bash
#SBATCH --account=bfbd-dtai-gh
#SBATCH --mem=480G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:1
#SBATCH --time=48:00:00
#SBATCH --job-name=vec2text_run
#SBATCH --output=vec2text_run_%j.out
#SBATCH --error=vec2text_run_%j.err
#SBATCH --time=48:00:00              # 48 hours
#SBATCH --partition=ghx4             # Replace with correct partition name if needed

# ---- Activate Conda Environment ----
source ~/.bashrc

# Activate your environment (adjust this line to your setup)
conda init
conda activate gradient_engine          # OR: conda activate myenv
export PYTHONPATH=~/vec2text
nvidia-smi
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"

# Run your command
torchrun vec2text/run.py --batch_size 128 --all_grads --reduction_version_SVD --wandb --from_gradients
