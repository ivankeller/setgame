#!/usr/bin/env python3
"""
Train the Set card attribute classifier.

Usage (from the setgame/ directory, with the virtualenv active):

    python scripts/train_card_classifier.py

    # Or with custom paths / hyperparameters:
    python scripts/train_card_classifier.py \\
        --originals  ../data/1_segmented_cards \\
        --labels     ../data/2_labels \\
        --augmented  ../data/3_augmented \\
        --output     ../data/models/set_card_classifier.pth \\
        --phase1-epochs 5 \\
        --phase2-epochs 25 \\
        --batch-size 32
"""

import argparse
from setgame.classify_card.train import train

PROJECT_ROOT = "/Users/ivankeller/Projects/setgame_project"

DEFAULTS = dict(
    originals_dir  = f"{PROJECT_ROOT}/data/1_segmented_cards",
    labels_dir     = f"{PROJECT_ROOT}/data/2_labels",
    augmented_dir  = f"{PROJECT_ROOT}/data/3_augmented",
    output_path    = f"{PROJECT_ROOT}/data/models/set_card_classifier.pth",
    val_fraction   = 0.2,
    img_size       = 224,
    batch_size     = 32,
    num_workers    = 4,
    phase1_epochs  = 5,
    phase2_epochs  = 20,
    seed           = 42,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Set card attribute classifier")
    p.add_argument("--originals",      default=DEFAULTS["originals_dir"])
    p.add_argument("--labels",         default=DEFAULTS["labels_dir"])
    p.add_argument("--augmented",      default=DEFAULTS["augmented_dir"],
                   help="Pass '' to disable augmented data")
    p.add_argument("--output",         default=DEFAULTS["output_path"])
    p.add_argument("--val-fraction",   type=float, default=DEFAULTS["val_fraction"])
    p.add_argument("--img-size",       type=int,   default=DEFAULTS["img_size"])
    p.add_argument("--batch-size",     type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--num-workers",    type=int,   default=DEFAULTS["num_workers"])
    p.add_argument("--phase1-epochs",  type=int,   default=DEFAULTS["phase1_epochs"])
    p.add_argument("--phase2-epochs",  type=int,   default=DEFAULTS["phase2_epochs"])
    p.add_argument("--seed",           type=int,   default=DEFAULTS["seed"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        originals_dir  = args.originals,
        labels_dir     = args.labels,
        augmented_dir  = args.augmented or None,
        output_path    = args.output,
        val_fraction   = args.val_fraction,
        img_size       = args.img_size,
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
        phase1_epochs  = args.phase1_epochs,
        phase2_epochs  = args.phase2_epochs,
        seed           = args.seed,
    )