#!/usr/bin/env python3
"""
Full Set game solver pipeline.

Given a photo of a board of Set cards, this script:
  1. Segments individual cards from the board image.
  2. Classifies the 4 attributes of each card (number, shape, shading, color).
  3. Finds all valid Sets among the cards.
  4. Draws rectangles around each solution's cards, one distinct color per solution.
  5. Saves the annotated image and optionally displays it.

Usage:
    uv run python scripts/solve_board.py -i <board_image> -m <model_weights>

Example:
    uv run python scripts/solve_board.py \\
        -i ../data/boards/my_game.jpg \\
        -m ../data/models/set_card_classifier.pth \\
        -o ../data/output/my_game_solution.jpg
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from loguru import logger

from setgame.utils import read_image
from setgame.draw_solution.draw_solution import (
    extract_cards_with_bboxes,
    draw_all_solutions,
    numpy_to_pil,
)
from setgame.classify_card.predict import load_model, predict_card
from setgame.find_set.find_set import Card, find_all_Sets


def solve(
    board_path: str,
    model_path: str,
    output_path: str | None = None,
    display: bool = True,
    background_thres: float = 0.25,
) -> Image.Image:
    """Run the full pipeline on a board image and return the annotated image.

    Parameters
    ----------
    board_path : str
        Path to the input board image (JPG or PNG).
    model_path : str
        Path to the trained classifier weights (.pth).
    output_path : str or None
        If given, save the annotated image to this path.
    display : bool
        If True, open the annotated image in the system viewer.
    background_thres : float
        Segmentation sensitivity (passed to extract_cards_with_bboxes).

    Returns
    -------
    PIL.Image.Image
        Annotated board image with color-coded rectangles for each solution.
    """
    # ── 1. Load board image ──────────────────────────────────────────────────
    logger.info(f"Loading board image: {board_path}")
    board_img = read_image(board_path)  # numpy RGB

    # ── 2. Segment individual cards ──────────────────────────────────────────
    card_imgs, bboxes = extract_cards_with_bboxes(board_img, background_thres)
    logger.info(f"Segmented {len(card_imgs)} cards from the board.")
    if len(card_imgs) < 3:
        logger.error("Fewer than 3 cards detected — check the image or background_thres.")
        sys.exit(1)

    # ── 3. Load classifier ───────────────────────────────────────────────────
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Loading classifier from: {model_path}  (device: {device})")
    model = load_model(model_path, device=device)

    # ── 4. Classify each card ────────────────────────────────────────────────
    cards = []
    for i, card_img in enumerate(card_imgs):
        pil_img = Image.fromarray(card_img)
        attrs = predict_card(pil_img, model, device=device)
        card = Card.from_dict(attrs)
        logger.debug(f"  Card {i:02d}: {attrs}")
        cards.append(card)

    # ── 5. Find a valid Set ──────────────────────────────────────────────────
    all_sets = find_all_Sets(cards)
    if not all_sets:
        logger.warning("No valid Set found among the detected cards.")
        sys.exit(0)

    logger.info(f"Found {len(all_sets)} valid set(s). Highlighting all of them.")
    for i, solution in enumerate(all_sets):
        logger.info(f"  Set {i + 1}: {' | '.join(str(c) for c in solution)}")

    # ── 6. Draw rectangles for every solution in a distinct color ────────────
    solutions_bboxes = [
        [bboxes[cards.index(card)] for card in solution]
        for solution in all_sets
    ]
    annotated_img = draw_all_solutions(board_img, solutions_bboxes)

    # ── 7. Save and/or display ───────────────────────────────────────────────
    result = numpy_to_pil(annotated_img)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        logger.info(f"Saved annotated image to: {output_path}")

    if display:
        result.show()

    return result


def parse_args():
    p = argparse.ArgumentParser(description="Set game board solver")
    p.add_argument("-i", "--input",  required=True,  help="Path to the board image (JPG or PNG)")
    p.add_argument("-m", "--model",  required=True,  help="Path to the trained classifier weights (.pth)")
    p.add_argument("-o", "--output", default=None,   help="Path to save the annotated output image")
    p.add_argument("--no-display",   action="store_true", help="Do not open the result image")
    p.add_argument("--background-thres", type=float, default=0.25,
                   help="Segmentation background threshold (default: 0.25)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    solve(
        board_path       = args.input,
        model_path       = args.model,
        output_path      = args.output,
        display          = not args.no_display,
        background_thres = args.background_thres,
    )