#!/bin/bash
#
# SLURM Job Submission Script for TAAC Training
#
# This script takes one argument: a unique job name for the experiment.
# This job name is used to create a dedicated directory for logs, models,
# and statistics, keeping your experiments organized.
#
# --- USAGE ---
# 1. Choose a unique job name, e.g., "4_box_tower_run"
#
# 2. Create the log directory for the job before submitting. SLURM needs
#    this directory to exist to write the output/error files.
#    mkdir -p experiments/boxjump/4_box_tower_run
#
# 3. Submit the job with sbatch, setting the output paths to the directory
#    you just created. Pass the job name as an argument to this script.
#    sbatch --job-name="4_box_tower_run" \
#           --output="experiments/boxjump/4_box_tower_run/slurm.out" \
#           --error="experiments/boxjump/4_box_tower_run/slurm.err" \
#           run_slurm_job.sh "4_box_tower_run"
#

# --- SBATCH Directives ---
# These can be overridden from the command line (e.g., sbatch --gpus=2 ...)
#SBATCH --partition=teaching
#SBATCH --gpus=4
#SBATCH --account=undergrad_research
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=10

# --- Script Logic ---

# Check if job name is provided
if [ -z "$1" ]; then
    echo "Error: Please provide a job name as the first argument."
    echo "Usage: sbatch run_slurm_job.sh <job_name>"
    exit 1
fi

JOB_NAME=$1

echo "Starting SLURM job: ${JOB_NAME}"
echo "Running on node: $(hostname)"
echo "Time: $(date)"

# Activate virtual environment (if required by your setup)
# source .venv/Scripts/activate

# Run the main training script, passing the job name
singularity exec --nv -B /data:/data /data/containers/msoe-tf2x.sif python scripts/train.py --config=configs/boxjump.yaml --job_name="${JOB_NAME}"

echo "Job ${JOB_NAME} finished with exit code $?."
echo "Time: $(date)"