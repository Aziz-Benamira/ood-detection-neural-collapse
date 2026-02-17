#!/bin/bash
#SBATCH --job-name=ood_test_1ep
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"
echo "=========================================="

# Check GPU
nvidia-smi

# Navigate to project directory
cd /home/ensta/ensta-ben-aissa/ood-detection-neural-collapse

# Create logs directory if it doesn't exist
mkdir -p logs

# Run 1 epoch test
python3 main.py \
    --epochs 1 \
    --data-dir ./data \
    --output-dir ./outputs \
    --batch-size 128 \
    --num-workers 4 \
    --seed 42

echo "Finished at: $(date)"

