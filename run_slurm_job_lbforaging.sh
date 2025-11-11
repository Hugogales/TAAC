#!/bin/bash
#
# SLURM Job Submission Script for TAAC Training
#
# This script takes one argument: a unique job name for the experiment.
# This job name is used to create a dedicated directory for logs, models,
# and statistics, keeping your experiments organized.
# nameing = T-LF-01 = lbforaging/taac/01
# naming = M-LF-01 = lbforaging/maac/01
# naming = P-LF-02 = lbforaging/ppo/02

# --- SBATCH Directives ---
#SBATCH --partition=teaching
#SBATCH --gpus=2
#SBATCH --account=undergrad_research
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --job-name=M-LF-109
#SBATCH --output=jobs/lbforaging/MAAC/109/slurm.out
#SBATCH --error=jobs/lbforaging/MAAC/109/slurm.err

echo "Running on node: $(hostname)"
echo "Time: $(date)"

# Initialize conda in this non-interactive shell
echo "Initializing conda environment for SLURM..."
if [ -f /usr/local/miniforge/miniforge3/etc/profile.d/conda.sh ]; then
    source /usr/local/miniforge/miniforge3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
    source "$HOME/mambaforge/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda.sh. Tried /usr/local/miniforge/miniforge3 and $HOME paths."
    exit 1
fi

# Activate TAAC environment
echo "Activating taac environment..."
conda activate taac || { echo "ERROR: Failed to activate 'taac' conda env."; conda info --envs; exit 1; }
echo "Conda environment activated"

# Limit thread oversubscription for BLAS/OpenMP libs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Verify environment
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Testing PyTorch import..."
python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run the main training script, passing the job name
python scripts/train.py --config=configs/lbforaging.yaml

echo "Job finished with exit code $?."
echo "Time: $(date)"