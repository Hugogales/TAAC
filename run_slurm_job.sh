#!/bin/bash
#
# SLURM Job Submission Script for TAAC Training
#
# This script takes one argument: a unique job name for the experiment.
# This job name is used to create a dedicated directory for logs, models,
# and statistics, keeping your experiments organized.

# --- SBATCH Directives ---
#SBATCH --partition=teaching
#SBATCH --gpus=4
#SBATCH --account=undergrad_research
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --job-name=H-BOX-07
#SBATCH --output=jobs/boxjump/TAAC_07/slurm.out
#SBATCH --error=jobs/boxjump/TAAC_07/slurm.err

echo "Running on node: $(hostname)"
echo "Time: $(date)"

# Activate virtual environment (if required by your setup)
echo "Initializing conda environment for SLURM..."
source /home/garrido-lestacheh/miniconda3/etc/profile.d/conda.sh

# Activate TAAC environment
echo "Activating taac environment..."
conda activate taac
echo "Conda environment activated"

# Verify environment
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Testing PyTorch import..."
python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run the main training script, passing the job name
python scripts/train.py --config=configs/boxjump.yaml

echo "Job finished with exit code $?."
echo "Time: $(date)"