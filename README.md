# OOD Detection & Neural Collapse

**Practical Work — Theory of Deep Learning (MVA + ENSTA)**  
Author: Aziz BEN AMIRA — February 2026

---

## 🚀 New to This Project? Start Here

**Complete setup guide for GPU clusters:** see [SETUP.md](SETUP.md)

**Quick start:**
```bash
# 1. Clone and setup
git clone https://github.com/your-username/ood_detection.git
cd ood_detection/project
pip install -r requirements.txt

# 2. Run full pipeline (200 epochs, ~3h on GPU)
python main.py --epochs 200

# 3. Check results
cat outputs/results/summary.json
```

---

## Overview

This project implements:

1. **ResNet-18 training** on CIFAR-100 (200 epochs, cosine annealing LR)
2. **6 OOD detection methods**: MSP, Max Logit, Energy, Mahalanobis, ViM, NECO
3. **Neural Collapse analysis** (NC1–NC5): within-class collapse, Simplex ETF, self-duality, NCC
4. **Bonus**: Neural Collapse across layers
5. Full visualizations, AUROC metrics, and result export

**Datasets**: CIFAR-100 (in-distribution), SVHN and DTD (out-of-distribution)

---

## Project Structure

```
project/
├── README.md                   ← you are here
├── requirements.txt            ← pip dependencies
├── config.py                   ← all hyperparameters & paths
├── main.py                     ← full pipeline (train + eval + analysis)
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── resnet_cifar.py     ← ResNet-18 adapted for CIFAR (32×32)
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py         ← CIFAR-100, SVHN, DTD loaders
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py          ← training loop + checkpointing + resume
│   ├── ood/
│   │   ├── __init__.py
│   │   ├── scores.py           ← MSP, MaxLogit, Energy, Mahalanobis, ViM, NECO
│   │   └── evaluation.py       ← AUROC computation + batch evaluation
│   ├── neural_collapse/
│   │   ├── __init__.py
│   │   ├── metrics.py          ← NC1-NC4 measurement functions
│   │   └── analysis.py         ← NC1-NC5 orchestration + across-layers bonus
│   └── utils/
│       ├── __init__.py
│       ├── feature_extraction.py  ← penultimate layer + multi-layer hooks
│       └── visualization.py       ← all plots saved as PNG (headless-safe)
└── scripts/
    ├── train.py                ← standalone training script
    ├── evaluate_ood.py         ← standalone OOD evaluation
    └── analyze_nc.py           ← standalone Neural Collapse analysis
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (recommended)

```bash
python main.py --epochs 200 --data-dir ./data --output-dir ./outputs
```

This will:
- Train ResNet-18 on CIFAR-100 for 200 epochs
- Evaluate all 6 OOD detection methods on SVHN (and DTD if available)
- Run the full Neural Collapse analysis (NC1–NC5 + across layers)
- Save all plots, logs, checkpoints, and JSON results under `./outputs/`

### 3. Or run each step separately

```bash
# Step 1: Train
python scripts/train.py --epochs 200 --data-dir ./data --output-dir ./outputs

# Step 2: Evaluate OOD (requires trained model)
python scripts/evaluate_ood.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --data-dir ./data --output-dir ./outputs

# Step 3: Neural Collapse analysis (requires trained model)
python scripts/analyze_nc.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --data-dir ./data --output-dir ./outputs
```

### 4. Resume training from a checkpoint

```bash
python scripts/train.py --resume outputs/checkpoints/checkpoint_epoch100.pth
```

### 5. Skip training and reuse an existing checkpoint

```bash
python main.py --skip-train --checkpoint outputs/checkpoints/best_model.pth
```

---

## GPU Cluster Usage (SLURM / sbatch)

Create a file `job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=ood_nc
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu

# Load modules (adapt to your cluster)
module load python/3.10
module load cuda/11.8

# Activate virtualenv
source ~/venvs/ood/bin/activate

# Run the full pipeline
cd /path/to/project
python main.py \
    --epochs 200 \
    --data-dir ./data \
    --output-dir ./outputs \
    --num-workers 4 \
    --batch-size 128
```

Submit with:

```bash
sbatch job.sh
```

Monitor progress:

```bash
# Live logs
tail -f outputs/logs/main.log

# Check SLURM output
tail -f slurm_*.out
```

---

## Outputs

After a full run, `outputs/` will contain:

```
outputs/
├── checkpoints/
│   ├── best_model.pth              ← best model weights (by test accuracy)
│   ├── checkpoint_epoch010.pth     ← full checkpoint (model + optimizer + scheduler)
│   ├── checkpoint_epoch020.pth
│   └── ...
├── logs/
│   ├── main.log                    ← or train.log / evaluate_ood.log / analyze_nc.log
├── plots/
│   ├── training_curves.png         ← loss, accuracy, LR schedule
│   ├── ood_scores_svhn.png         ← score distributions for each method
│   ├── final_ood_comparison_svhn.png  ← AUROC bar chart
│   ├── neco_analysis_svhn.png      ← NECO score + similarity patterns
│   ├── neural_collapse_nc1_nc4.png ← 6-panel NC analysis
│   └── nc_across_layers.png        ← NC1 vs layer depth (bonus)
└── results/
    ├── training_history.json       ← per-epoch metrics
    ├── ood_results.json            ← AUROC for each method × dataset
    ├── nc_results.json             ← all NC metrics
    └── summary.json                ← combined summary
```

---

## Command-Line Arguments

All scripts support `--help` for full argument docs. Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 200 | Number of training epochs |
| `--batch-size` | 128 | Mini-batch size |
| `--lr` | 0.1 | Initial learning rate |
| `--weight-decay` | 5e-4 | L2 regularization |
| `--data-dir` | `./data` | Dataset root directory |
| `--output-dir` | `./outputs` | All outputs go here |
| `--checkpoint` | — | Path to model weights |
| `--resume` | — | Resume training from checkpoint |
| `--temperature` | 1.0 | Temperature for energy score |
| `--num-workers` | 2 | Data loading workers |
| `--seed` | 42 | Random seed |
| `--skip-train` | — | Skip training (main.py only) |
| `--skip-across-layers` | — | Skip bonus NC-across-layers |

---

## Methods Implemented

### OOD Detection

| Method | Reference | Key Idea |
|---|---|---|
| **MSP** | Hendrycks & Gimpel, ICLR 2017 | Max softmax probability |
| **Max Logit** | Hendrycks et al., ICML 2022 | Max raw logit value |
| **Energy** | Liu et al., NeurIPS 2020 | LogSumExp of logits |
| **Mahalanobis** | Lee et al., NeurIPS 2018 | Feature-space distance with shared covariance |
| **ViM** | Wang et al., CVPR 2022 | Null-space residual + virtual logit |
| **NECO** | Ammar et al., ICLR 2024 | Neural Collapse inspired cosine similarity gap |

### Neural Collapse (Papyan, Han & Donoho, PNAS 2020)

| Property | What it measures |
|---|---|
| **NC1** | Within-class variability collapse (Σ_W → 0) |
| **NC2** | Class means → Simplex ETF (equinorm + equiangular) |
| **NC3** | Classifier weights align with class means (self-duality) |
| **NC4** | Predictions simplify to Nearest Class Center |
| **NC5** | NCC generalizes to test data |

---

## Expected Results

With 200 epochs of training on CIFAR-100:

- **Test accuracy**: ~78–80%
- **OOD AUROC (SVHN)**: typically 80–95% depending on method
- **NC1 metric**: should be small (< 0.1), indicating within-class collapse
- **NC3 mean cosine**: should be close to 1.0 (classifier-feature alignment)
- **NC4 agreement**: should be > 95% (model ≈ NCC classifier)
