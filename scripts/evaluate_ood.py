#!/usr/bin/env python3
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
    CIFAR100_MEAN, CIFAR100_STD, ENERGY_TEMPERATURE,
    DATA_DIR, OUTPUT_DIR,
)
from src.models import build_resnet18_cifar
from src.data import get_dataloaders
from src.utils.feature_extraction import extract_features_and_logits
from src.ood.evaluation import evaluate_all_ood_methods, compute_auroc
from src.ood.scores import score_msp, score_max_logit, score_energy, score_neco
from src.ood.scores import compute_class_statistics, compute_neco_parameters, score_mahalanobis
from src.ood.scores import compute_vim_parameters, score_vim
from src.utils.visualization import (
    plot_ood_score_distributions,
    plot_final_ood_comparison,
    plot_neco_analysis,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate OOD detection methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained model weights (.pth)")
    p.add_argument("--data-dir", type=str, default=DATA_DIR)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--temperature", type=float, default=ENERGY_TEMPERATURE,
                   help="Temperature for energy score")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "evaluate_ood.log")

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

    #    Reproducibility   
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #    Device   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    #    Data   
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cifar_mean=CIFAR100_MEAN,
        cifar_std=CIFAR100_STD,
    )

    #Model 
    model = build_resnet18_cifar(num_classes=NUM_CLASSES).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    # Handle both raw state_dict and full checkpoint
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    logger.info("Loaded model from %s", args.checkpoint)

    #Feature extraction   
    logger.info("Extracting features from CIFAR-100 train set...")
    train_features, train_logits, train_labels = extract_features_and_logits(
        model, loaders['train'], device,
    )

    logger.info("Extracting features from CIFAR-100 test set (in-distribution)...")
    id_features, id_logits, id_labels = extract_features_and_logits(
        model, loaders['test'], device,
    )

    #Evaluate on SVHN   
    logger.info("Extracting features from SVHN (out-of-distribution)...")
    ood_features_svhn, ood_logits_svhn, _ = extract_features_and_logits(
        model, loaders['svhn'], device,
    )

    logger.info("\nOOD Detection: CIFAR-100 (ID) vs SVHN (OOD)")


    svhn_results = evaluate_all_ood_methods(
        model=model,
        train_features=train_features,
        train_logits=train_logits,
        train_labels=train_labels,
        id_features=id_features,
        id_logits=id_logits,
        ood_features=ood_features_svhn,
        ood_logits=ood_logits_svhn,
        num_classes=NUM_CLASSES,
        temperature=args.temperature,
    )

    #Plots
    plot_ood_score_distributions(
        {k: v for k, v in svhn_results.items() if not k.startswith('_')},
        plot_dir, ood_name="SVHN",
    )
    plot_final_ood_comparison(
        {k: v for k, v in svhn_results.items() if not k.startswith('_')},
        plot_dir, ood_name="SVHN",
    )

    # NECO analysis plot
    fitted = svhn_results['_fitted']
    global_mean = train_features.mean(axis=0)
    cm = fitted['class_means'] - global_mean
    neco_viz_means = cm / (np.linalg.norm(cm, axis=1, keepdims=True) + 1e-8)

    plot_neco_analysis(
        id_neco=svhn_results['NECO']['id'],
        ood_neco=svhn_results['NECO']['ood'],
        auroc_neco=svhn_results['NECO']['auroc'],
        id_features=id_features,
        ood_features=ood_features_svhn,
        neco_normalized_means=neco_viz_means,
        neco_global_mean=global_mean,
        output_dir=plot_dir,
        ood_name="SVHN",
    )

    #Evaluate on DTD (if available)   
    dtd_aurocs = {}
    if loaders['dtd'] is not None:
        logger.info("\nExtracting features from DTD (out-of-distribution)...")
        ood_features_dtd, ood_logits_dtd, _ = extract_features_and_logits(
            model, loaders['dtd'], device,
        )

        logger.info("\nOOD Detection: CIFAR-100 (ID) vs DTD (OOD)")
        dtd_results = evaluate_all_ood_methods(
            model=model,
            train_features=train_features,
            train_logits=train_logits,
            train_labels=train_labels,
            id_features=id_features,
            id_logits=id_logits,
            ood_features=ood_features_dtd,
            ood_logits=ood_logits_dtd,
            num_classes=NUM_CLASSES,
            temperature=args.temperature,
        )

        plot_ood_score_distributions(
            {k: v for k, v in dtd_results.items() if not k.startswith('_')},
            plot_dir, ood_name="DTD",
        )
        plot_final_ood_comparison(
            {k: v for k, v in dtd_results.items() if not k.startswith('_')},
            plot_dir, ood_name="DTD",
        )

        plot_neco_analysis(
            id_neco=dtd_results['NECO']['id'],
            ood_neco=dtd_results['NECO']['ood'],
            auroc_neco=dtd_results['NECO']['auroc'],
            id_features=id_features,
            ood_features=ood_features_dtd,
            neco_normalized_means=neco_viz_means,
            neco_global_mean=global_mean,
            output_dir=plot_dir,
            ood_name="DTD",
        )

        dtd_aurocs = {k: v['auroc'] for k, v in dtd_results.items() if not k.startswith('_')}

    #Final summary   
    
    logger.info("FINAL OOD DETECTION COMPARISON")
    methods = [k for k in svhn_results if not k.startswith('_')]
    logger.info("%-20s %15s %15s", "Method", "AUROC (SVHN)", "AUROC (DTD)")
    logger.info("-" * 55)
    for m in methods:
        svhn_val = svhn_results[m]['auroc']
        dtd_val = dtd_aurocs.get(m, float('nan'))
        logger.info("%-20s %15.4f %15.4f", m, svhn_val, dtd_val)
    logger.info("-" * 55)

    svhn_aurocs = {k: v['auroc'] for k, v in svhn_results.items() if not k.startswith('_')}
    best_method = max(svhn_aurocs, key=svhn_aurocs.get)
    logger.info("Best method (SVHN): %s (AUROC = %.4f)", best_method, svhn_aurocs[best_method])

    # Save results as JSON
    serializable = {
        'svhn': svhn_aurocs,
        'dtd': dtd_aurocs if dtd_aurocs else None,
    }
    results_path = os.path.join(results_dir, "ood_results.json")
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results saved → %s", results_path)

    logger.info("\nPlots saved to %s", plot_dir)
    logger.info("All done!")


if __name__ == "__main__":
    main()
