# OOD Detection & Neural Collapse for CIFAR-100

Research project implementing Out-of-Distribution (OOD) detection methods and Neural Collapse analysis.

---

## Project Overview

This project trains a ResNet-18 classifier on CIFAR-100 and studies:
1. **6 OOD Detection Methods**: MSP, Max Logit, Energy, Mahalanobis, ViM, NECO
2. **Neural Collapse**: NC1-NC4 analysis + bonus layer-wise study
3. **Performance**: 78.65% test accuracy, 0.813-0.876 AUROC on OOD datasets

---

## Project Structure

```
project/
├── main.py                 # Full pipeline orchestrator
├── config.py              # Hyperparameters & paths
├── requirements.txt       # Dependencies
│
├── src/
│   ├── models/           # ResNet-18 for CIFAR
│   ├── data/             # Data loaders (CIFAR-100, SVHN, DTD)
│   ├── training/         # Training loop with checkpointing
│   ├── ood/              # 6 OOD scoring methods
│   ├── neural_collapse/  # NC1-NC5 metrics
│   └── utils/            # Feature extraction, visualization
│
├── scripts/              # Standalone CLI scripts
│   ├── train.py
│   ├── evaluate_ood.py
│   └── analyze_nc.py
│
├── outputs/
│   ├── plots/            # All visualizations (9 figures)
│   ├── results/          # summary.json, training_history.json
│   └── checkpoints/      # Saved models (excluded from repo due to size)
│
├── analysis.md           # Personal insights & observations
├── experiments.py        # Temperature tuning, ViM debugging
├── STUDY_GUIDE.md        # Defense preparation notes
└── TODO.md              # Work-in-progress tracking
```

---

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training (200 epochs)

```bash
# Local execution
python main.py --epochs 200

# Or via SLURM (GPU cluster)
sbatch run_200epochs.sh
```

### Standalone Scripts

```bash
# Train only
python scripts/train.py --epochs 200 --output-dir ./outputs

# Evaluate OOD (requires trained model)
python scripts/evaluate_ood.py --checkpoint ./outputs/checkpoints/best_model.pth

# Analyze Neural Collapse
python scripts/analyze_nc.py --checkpoint ./outputs/checkpoints/best_model.pth
```

---

## Results Summary

### Training Performance
- **Test Accuracy**: 78.65% (CIFAR-100, 100 classes)
- **Best Epoch**: 161 (saved to `best_model.pth`)
- **Training Time**: ~20 minutes on NVIDIA L40S GPU

### OOD Detection (AUROC)

| Method | SVHN | DTD |
|--------|------|-----|
| Energy | **0.876** | 0.784 |
| NECO | 0.840 | **0.813** |
| Max Logit | 0.869 | 0.784 |
| MSP | 0.846 | 0.779 |
| Mahalanobis | 0.688 | 0.781 |
| ViM | 0.225 | 0.451 |

### Neural Collapse Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| NC1 | 1,126,273 | Partial collapse (expected at 78% acc) |
| NC2 (angle) | -0.0101 |  Perfect simplex! |
| NC3 (alignment) | 0.977 | Strong weight-mean alignment |
| NC4 (agreement) | 100% (train) |  NCC = Softmax |
| NC5 (test) | 96.89% | Generalizes to test data |

---

## Key Insights

1. **Neural Collapse is real**: Despite only 78% accuracy, NC2-NC5 show strong geometric structure
2. **Simple methods work**: Energy score (0.876 AUROC) outperforms complex ViM
3. **Dataset matters**: NECO excels on textures (DTD), Energy on street numbers (SVHN)
4. **ViM needs debugging**: Poor performance suggests hyperparameter tuning needed

---

## References

- **Neural Collapse**: Papyan et al. (2020) - "Prevalence of Neural Collapse during the terminal phase of deep learning training"
- **NECO**: Inspired by Neural Collapse geometry for OOD detection
- **Energy Score**: Liu et al. (2020) - "Energy-based Out-of-distribution Detection"
- **Mahalanobis**: Lee et al. (2018) - "A Simple Unified Framework for Detecting OOD Samples"
- **ViM**: Wang et al. (2022) - "ViM: Out-Of-Distribution with Virtual-logit Matching"

---

## Files to Review

- **`analysis.md`**: Personal observations and deep dive into results
- **`STUDY_GUIDE.md`**: Defense preparation with key concepts
- **`experiments.py`**: Temperature tuning and ViM debugging experiments
- **`outputs/plots/`**: 9 visualizations (training curves, OOD scores, NC analysis)

---

## Known Issues

1. **ViM performance**: AUROC of 0.225 (needs hyperparameter tuning)
2. **NC1 not collapsed**: Expected at 78% accuracy; would need >95% for full collapse
3. **Large checkpoints**: Intermediate checkpoints excluded from repo (~2.6GB total)

---

##  Collaboration

**Share with mate**: This repo contains full working code + results. To reproduce:
1. Download CIFAR-100 (automatic via torchvision)
2. Download OOD datasets: SVHN, DTD (handled by `get_dataloaders()`)
3. Run `main.py` or SLURM scripts

**What's included**:
-  Complete working pipeline
-  All results (plots, metrics, training logs)
-  Personal analysis and experiments
-  Study materials for defense

**What's excluded** (too large):
-  Intermediate checkpoints (20 × 129MB each)
-  Raw datasets (downloaded on-demand)


##  License

For educational purposes (MVA + ENSTA course project).
