#!/usr/bin/env python3
"""
Run the full Neural Collapse analysis (NC1–NC5 + bonus across layers).
Usage
    python scripts/analyze_nc.py \\
        --checkpoint outputs/checkpoints/best_model.pth \\
        --data-dir ./data \\
        --output-dir ./outputs
"""

import argparse
import json
import logging
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SEED, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS,
    CIFAR100_MEAN, CIFAR100_STD,
    DATA_DIR, OUTPUT_DIR,
)
from src.models import build_resnet18_cifar
from src.data import get_dataloaders
from src.utils.feature_extraction import extract_features_and_logits
from src.ood.scores import compute_class_statistics
from src.neural_collapse.analysis import run_full_nc_analysis, measure_nc_across_layers
from src.utils.visualization import plot_neural_collapse, plot_nc_across_layers


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Neural Collapse analysis (NC1–NC5 + bonus)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained model weights (.pth)")
    p.add_argument("--data-dir", type=str, default=DATA_DIR)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--skip-across-layers", action="store_true",
                   help="Skip the bonus NC-across-layers analysis")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "analyze_nc.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)


def main():
    args = parse_args()

    log_dir = os.path.join(args.output_dir, "logs")
    plot_dir = os.path.join(args.output_dir, "plots")
    results_dir = os.path.join(args.output_dir, "results")

    for d in [log_dir, plot_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    setup_logging(log_dir)
    logger = logging.getLogger(__name__)

    #   Reproducibility  
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #   Device  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    #   Data  
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cifar_mean=CIFAR100_MEAN,
        cifar_std=CIFAR100_STD,
    )

    #   Model  
    model = build_resnet18_cifar(num_classes=NUM_CLASSES).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    logger.info("Loaded model from %s", args.checkpoint)

    #   Feature extraction  
    logger.info("Extracting features from CIFAR-100 train set...")
    train_features, train_logits, train_labels = extract_features_and_logits(
        model, loaders['train'], device,
    )

    logger.info("Extracting features from CIFAR-100 test set...")
    id_features, id_logits, id_labels = extract_features_and_logits(
        model, loaders['test'], device,
    )

    #   Compute class means (used by NC5)  
    class_means, _, _ = compute_class_statistics(
        train_features, train_labels, NUM_CLASSES,
    )

    #   NC1 – NC5  
    nc_results = run_full_nc_analysis(
        model=model,
        train_features=train_features,
        train_labels=train_labels,
        id_features=id_features,
        id_logits=id_logits,
        id_labels=id_labels,
        class_means=class_means,
        num_classes=NUM_CLASSES,
    )

    #   Plots  
    plot_neural_collapse(
        per_class_var=nc_results['per_class_var'],
        cos_sim_matrix=nc_results['nc2_cos_sim_matrix'],
        nc3_cos_sims=nc_results['nc3_cos_sims'],
        nc3_mean_cos=nc_results['nc3_mean_cos'],
        nc4_model_acc=nc_results['nc4_model_acc'],
        nc4_ncc_acc=nc_results['nc4_ncc_acc'],
        nc4_agreement=nc_results['nc4_agreement'],
        train_features=train_features,
        train_labels=train_labels,
        num_classes=NUM_CLASSES,
        output_dir=plot_dir,
    )

    #   Bonus: NC across layers  
    if not args.skip_across_layers:
        nc1_per_layer, pcv_per_layer = measure_nc_across_layers(
            model=model,
            train_loader=loaders['train'],
            device=device,
            train_features_penult=train_features,
            train_labels=train_labels,
            num_classes=NUM_CLASSES,
        )
        plot_nc_across_layers(nc1_per_layer, pcv_per_layer, plot_dir)

        nc_results['nc_across_layers'] = {k: float(v) for k, v in nc1_per_layer.items()}

    #   Save results as JSON  
    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for k, v in nc_results.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (float, int, str)):
            serializable[k] = v
        elif isinstance(v, dict):
            serializable[k] = {
                kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                for kk, vv in v.items()
            }

    results_path = os.path.join(results_dir, "nc_results.json")
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results saved → %s", results_path)

    logger.info("\nPlots saved to %s", plot_dir)
    logger.info("All done!")


if __name__ == "__main__":
    main()
