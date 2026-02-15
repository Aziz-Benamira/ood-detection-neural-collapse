#!/usr/bin/env python3
"""
Entry point — run the full pipeline (train + evaluate OOD + analyze NC).

This is a convenience wrapper around the individual scripts.  For more
fine-grained control, use the scripts directly:

    python scripts/train.py        --help
    python scripts/evaluate_ood.py --help
    python scripts/analyze_nc.py   --help

Usage
-----
Full pipeline:
    python main.py --epochs 200 --data-dir ./data --output-dir ./outputs

Skip training (reuse existing checkpoint):
    python main.py --skip-train --checkpoint outputs/checkpoints/best_model.pth

Author: Aziz BEN AMIRA
Course: Theory of Deep Learning (MVA + ENSTA)
"""

import argparse
import json
import logging
import os
import sys

import torch
import numpy as np

from config import (
    SEED, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS,
    CIFAR100_MEAN, CIFAR100_STD, ENERGY_TEMPERATURE,
    NUM_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, LR_MIN,
    CHECKPOINT_EVERY, PRINT_EVERY,
    DATA_DIR, OUTPUT_DIR,
)
from src.models import build_resnet18_cifar
from src.data import get_dataloaders
from src.training import Trainer
from src.utils.feature_extraction import extract_features_and_logits
from src.ood.evaluation import evaluate_all_ood_methods
from src.ood.scores import compute_class_statistics
from src.neural_collapse.analysis import run_full_nc_analysis, measure_nc_across_layers
from src.utils.visualization import (
    plot_training_curves,
    plot_ood_score_distributions,
    plot_final_ood_comparison,
    plot_neco_analysis,
    plot_neural_collapse,
    plot_nc_across_layers,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="OOD Detection & Neural Collapse — full pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data
    p.add_argument("--data-dir", type=str, default=DATA_DIR)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    # training
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--momentum", type=float, default=MOMENTUM)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--lr-min", type=float, default=LR_MIN)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    p.add_argument("--print-every", type=int, default=PRINT_EVERY)
    # resume / skip
    p.add_argument("--resume", type=str, default=None,
                   help="Resume training from this checkpoint")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training, use --checkpoint instead")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Pretrained weights (used when --skip-train)")
    p.add_argument("--skip-across-layers", action="store_true",
                   help="Skip the bonus NC-across-layers analysis")
    # misc
    p.add_argument("--temperature", type=float, default=ENERGY_TEMPERATURE)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "main.log")

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

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    log_dir = os.path.join(args.output_dir, "logs")
    plot_dir = os.path.join(args.output_dir, "plots")
    results_dir = os.path.join(args.output_dir, "results")

    for d in [checkpoint_dir, log_dir, plot_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    setup_logging(log_dir)
    logger = logging.getLogger(__name__)

    # ---- Reproducibility ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- Device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)
    if device.type == 'cuda':
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # ---- Data ----
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ---- Model ----
    model = build_resnet18_cifar(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("ResNet-18: %s parameters", f"{total_params:,}")

    # ==================================================================
    # STEP 1: TRAINING
    # ==================================================================
    if not args.skip_train:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: TRAINING")
        logger.info("=" * 60)

        trainer = Trainer(
            model=model,
            device=device,
            num_epochs=args.epochs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            lr_min=args.lr_min,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            print_every=args.print_every,
        )

        if args.resume:
            trainer.load_checkpoint(args.resume)

        history = trainer.train(loaders['train'], loaders['test'])
        plot_training_curves(history, plot_dir)

        # Load best model for subsequent analysis
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        # Load from provided checkpoint
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        logger.info("Loaded pretrained model from %s", ckpt_path)

    model.eval()

    # ==================================================================
    # STEP 2: FEATURE EXTRACTION
    # ==================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: FEATURE EXTRACTION")
    logger.info("=" * 60)

    logger.info("Extracting features from CIFAR-100 train set...")
    train_features, train_logits, train_labels = extract_features_and_logits(
        model, loaders['train'], device,
    )

    logger.info("Extracting features from CIFAR-100 test set...")
    id_features, id_logits, id_labels = extract_features_and_logits(
        model, loaders['test'], device,
    )

    logger.info("Extracting features from SVHN...")
    ood_features_svhn, ood_logits_svhn, _ = extract_features_and_logits(
        model, loaders['svhn'], device,
    )

    ood_features_dtd, ood_logits_dtd = None, None
    if loaders['dtd'] is not None:
        logger.info("Extracting features from DTD...")
        ood_features_dtd, ood_logits_dtd, _ = extract_features_and_logits(
            model, loaders['dtd'], device,
        )

    # ==================================================================
    # STEP 3: OOD EVALUATION
    # ==================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: OOD EVALUATION")
    logger.info("=" * 60)

    svhn_results = evaluate_all_ood_methods(
        model, train_features, train_logits, train_labels,
        id_features, id_logits,
        ood_features_svhn, ood_logits_svhn,
        num_classes=NUM_CLASSES, temperature=args.temperature,
    )

    public_svhn = {k: v for k, v in svhn_results.items() if not k.startswith('_')}
    plot_ood_score_distributions(public_svhn, plot_dir, "SVHN")
    plot_final_ood_comparison(public_svhn, plot_dir, "SVHN")

    fitted = svhn_results['_fitted']
    plot_neco_analysis(
        svhn_results['NECO']['id'], svhn_results['NECO']['ood'],
        svhn_results['NECO']['auroc'],
        id_features, ood_features_svhn,
        fitted['neco_normalized_means'], fitted['neco_global_mean'],
        plot_dir, "SVHN",
    )

    if ood_features_dtd is not None:
        dtd_results = evaluate_all_ood_methods(
            model, train_features, train_logits, train_labels,
            id_features, id_logits,
            ood_features_dtd, ood_logits_dtd,
            num_classes=NUM_CLASSES, temperature=args.temperature,
        )
        public_dtd = {k: v for k, v in dtd_results.items() if not k.startswith('_')}
        plot_ood_score_distributions(public_dtd, plot_dir, "DTD")
        plot_final_ood_comparison(public_dtd, plot_dir, "DTD")

        fitted_dtd = dtd_results['_fitted']
        plot_neco_analysis(
            dtd_results['NECO']['id'], dtd_results['NECO']['ood'],
            dtd_results['NECO']['auroc'],
            id_features, ood_features_dtd,
            fitted_dtd['neco_normalized_means'], fitted_dtd['neco_global_mean'],
            plot_dir, "DTD",
        )

    # ==================================================================
    # STEP 4: NEURAL COLLAPSE ANALYSIS
    # ==================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: NEURAL COLLAPSE ANALYSIS")
    logger.info("=" * 60)

    class_means = fitted['class_means']

    nc_results = run_full_nc_analysis(
        model, train_features, train_labels,
        id_features, id_logits, id_labels,
        class_means, num_classes=NUM_CLASSES,
    )

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

    # ---- Bonus: NC across layers ----
    if not args.skip_across_layers:
        nc1_per_layer, pcv_per_layer = measure_nc_across_layers(
            model, loaders['train'], device,
            train_features, train_labels, NUM_CLASSES,
        )
        plot_nc_across_layers(nc1_per_layer, pcv_per_layer, plot_dir)

    # ==================================================================
    # SAVE SUMMARY
    # ==================================================================
    svhn_aurocs = {k: v['auroc'] for k, v in svhn_results.items() if not k.startswith('_')}
    summary = {
        'ood_auroc_svhn': svhn_aurocs,
        'neural_collapse': {
            'NC1': nc_results['nc1_metric'],
            'NC2_norm_cv': nc_results['nc2_norm_std'],
            'NC2_cos_mean_offdiag': nc_results['nc2_cos_mean_offdiag'],
            'NC3_mean_cos': nc_results['nc3_mean_cos'],
            'NC4_agreement': nc_results['nc4_agreement'],
            'NC5_test_agreement': nc_results['nc5_test_agreement'],
        },
    }
    if ood_features_dtd is not None:
        summary['ood_auroc_dtd'] = {
            k: v['auroc'] for k, v in dtd_results.items() if not k.startswith('_')
        }

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved → %s", summary_path)

    logger.info("\nAll plots saved to %s", plot_dir)
    logger.info("All done!")


if __name__ == "__main__":
    main()
