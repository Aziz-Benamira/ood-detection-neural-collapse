#!/bin/bash
#SBATCH --job-name=ood_train_200ep
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"
echo "=========================================="

# Check GPU
nvidia-smi

# Activate virtual environment
source /home/ensta/ensta-ben-amira/envs/agentic_ai/bin/activate

# Navigate to project directory
cd /home/ensta/ensta-ben-amira/ood_detection/project

# Create logs directory if it doesn't exist
mkdir -p logs

# Run full 200 epoch training
python3 main.py \
    --epochs 200 \
    --data-dir ./data \
    --output-dir ./outputs \
    --batch-size 128 \
    --num-workers 8 \
    --seed 42

echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
