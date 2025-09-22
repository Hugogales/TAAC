#!/bin/bash
#
# SLURM Job Submission Script for TAAC Training
#
# This script takes one argument: a unique job name for the experiment.
# This job name is used to create a dedicated directory for logs, models,
# and statistics, keeping your experiments organized.
# nameing = E-BOX-01 = boxjump/evaluate/01
# naming = E-BOX-01 = boxjump/evaluate/01
# naming = E-BOX-02 = boxjump/evaluate/02

# --- SBATCH Directives ---
#SBATCH --partition=teaching
#SBATCH --gpus=3
#SBATCH --account=undergrad_research
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --job-name=E-BOX-05
#SBATCH --output=jobs/boxjump/evaluate/05/slurm.out
#SBATCH --error=jobs/boxjump/evaluate/05/slurm.err

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

# Ensure plotting deps are available in the active Python env
echo "Ensuring seaborn and matplotlib are available in $(which python)"
if ! python -c "import seaborn, matplotlib" >/dev/null 2>&1; then
    echo "Installing seaborn and matplotlib via pip..."
    python -m pip install --upgrade pip setuptools wheel | cat
    python -m pip install --no-input seaborn matplotlib | cat
fi
# Fallback to conda if still missing (e.g., restricted pip)
if ! python -c "import seaborn, matplotlib" >/dev/null 2>&1; then
    echo "pip install failed or packages still missing; trying conda-forge..."
    conda install -y -c conda-forge seaborn matplotlib || true
fi
# Final verification
python - <<'PY'
try:
    import seaborn, matplotlib
    print(f"Seaborn {seaborn.__version__} | Matplotlib {matplotlib.__version__}")
except Exception as e:
    print(f"Plotting deps missing: {e}")
PY

# Verify environment
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Testing PyTorch import..."
python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run the main training script, passing the job name
python scripts/evaluate.py --config=configs/boxjump.yaml

echo "Job finished with exit code $?."
echo "Time: $(date)"