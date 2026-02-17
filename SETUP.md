# Setup Guide for GPU Cluster

Complete setup instructions for running this project on a GPU cluster.

---

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/ood_detection.git
cd ood_detection/project
```

---

## 2. Environment Setup

### Option A: Using conda (recommended)

```bash
# Create environment
conda create -n ood_env python=3.10 -y
conda activate ood_env

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install numpy scipy scikit-learn matplotlib seaborn tqdm
```

### Option B: Using virtualenv

```bash
# Create virtual environment
python3 -m venv ~/venvs/ood_env
source ~/venvs/ood_env/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True
```

---

## 3. Quick Test Run (1 epoch)

Test that everything works before launching the full 200-epoch run:

```bash
# Quick 1-epoch test
python main.py --epochs 1 --batch-size 128 --num-workers 4
```

This should:
- Download CIFAR-100, SVHN, DTD (~500MB total)
- Train for 1 epoch (~2-3 minutes on GPU)
- Run OOD evaluation
- Generate plots in `outputs/plots/`

If this completes successfully, you're ready for the full run!

---

## 4. Full Training (200 epochs)

### Interactive session (if your cluster allows)

```bash
# Activate environment
source ~/venvs/ood_env/bin/activate  # or: conda activate ood_env

# Run full pipeline
python main.py --epochs 200 --batch-size 128 --num-workers 4
```

Expected runtime: **2-4 hours** on a single modern GPU (L40S, H100, A100, V100)

### SLURM batch job (recommended for clusters)

Create `run_training.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=ood_training
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu

# === ADAPT THESE TO YOUR CLUSTER ===
module load python/3.10
module load cuda/11.8
source ~/venvs/ood_env/bin/activate
# ====================================

cd $SLURM_SUBMIT_DIR

# Run training
python main.py \
    --epochs 200 \
    --batch-size 128 \
    --num-workers 4 \
    --data-dir ./data \
    --output-dir ./outputs

echo "Training completed!"
```

Submit the job:

```bash
mkdir -p logs
sbatch run_training.sh
```

Monitor progress:

```bash
# Check SLURM queue
squeue -u $USER

# Watch live logs
tail -f outputs/logs/main.log

# Or SLURM output
tail -f logs/slurm_*.out
```

---

## 5. Resume Training (if interrupted)

If training gets interrupted, resume from the last checkpoint:

```bash
python main.py --resume outputs/checkpoints/checkpoint_epoch100.pth
```

---

## 6. Evaluation Only (skip training)

If you already have a trained model:

```bash
python main.py \
    --skip-train \
    --checkpoint outputs/checkpoints/best_model.pth \
    --skip-across-layers  # optional: skip the bonus analysis
```

This will:
- Load the trained model
- Run OOD evaluation on SVHN and DTD
- Perform Neural Collapse analysis
- Generate all plots and metrics

Runtime: **5-10 minutes**

---

## 7. Results Location

After completion, check:

```bash
ls -lh outputs/
```

You should see:

```
outputs/
├── checkpoints/
│   ├── best_model.pth              (best model by test accuracy)
│   ├── checkpoint_epoch200.pth     (final checkpoint)
├── logs/
│   └── main.log                    (detailed training logs)
├── plots/
│   ├── training_curves.png
│   ├── final_ood_comparison_svhn.png
│   ├── neural_collapse_nc1_nc4.png
│   └── ... (10+ visualization files)
└── results/
    ├── summary.json                (all metrics in one file)
    ├── training_history.json
    └── ood_results.json
```

Key file to check: **`outputs/results/summary.json`**

```bash
cat outputs/results/summary.json
```

Example output:
```json
{
  "ood_auroc_svhn": {
    "Energy": 0.876,
    "ViM": 0.776,
    "NECO": 0.833,
    ...
  },
  "neural_collapse": {
    "NC1": 1145083.28,
    "NC3_mean_cos": 0.977,
    "NC4_agreement": 0.99994,
    ...
  }
}
```

---

## 8. Troubleshooting

### CUDA out of memory

Reduce batch size:

```bash
python main.py --epochs 200 --batch-size 64  # or 32
```

### Datasets won't download (firewall/proxy)

Download manually and extract to `data/`:

- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
- SVHN: http://ufldl.stanford.edu/housenumbers/
- DTD: https://www.robots.ox.ac.uk/~vgg/data/dtd/

### "No module named src"

Make sure you're in the `project/` directory:

```bash
cd /path/to/ood_detection/project
python main.py ...
```

### Plots not showing up

The code saves plots as PNG files (headless mode). They won't display but will be saved to `outputs/plots/`. Download them to view:

```bash
scp username@cluster:/path/to/outputs/plots/*.png ./local_folder/
```

---

## 9. Cluster-Specific Notes

### ENSTA Cluster

```bash
# Load modules
module load python/3.10
module load cuda/11.8

# Activate environment
source ~/ensta/envs/agentic_ai/bin/activate

# Submit job
sbatch run_training.sh
```

### Generic SLURM Cluster

Adjust these in your SLURM script:
- `--partition`: check available partitions with `sinfo`
- `--gres=gpu:1`: change to your GPU type (e.g., `gpu:a100:1`)
- `module load`: check available modules with `module avail`

---

## 10. Expected Training Time

| GPU Type | Approximate Time (200 epochs) |
|----------|------------------------------|
| H100     | ~1.5-2 hours                 |
| A100     | ~2-3 hours                   |
| L40S     | ~2.5-3.5 hours               |
| V100     | ~3-4 hours                   |
| RTX 3090 | ~4-5 hours                   |
| T4       | ~8-10 hours                  |

Single epoch takes ~1-2 minutes on modern GPUs.

---

## 11. Citation

If you use this code, please cite the original papers:

```bibtex
@article{papyan2020prevalence,
  title={Prevalence of neural collapse during the terminal phase of deep learning training},
  author={Papyan, Vardan and Han, XY and Donoho, David L},
  journal={PNAS},
  year={2020}
}

@inproceedings{ammar2024neco,
  title={NECO: NEural Collapse Based Out-of-distribution detection},
  author={Ammar, Mouïn Ben and others},
  booktitle={ICLR},
  year={2024}
}
```

---

**Questions?** Open an issue on GitHub or contact the author.
