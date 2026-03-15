"""
Utilities to extract card bounding boxes from a board image and draw
solution rectangles on the original image.
"""

from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image

from setgame.segmentboard.segmentboard import (
    add_frame,
    binarize_image,
    img_ratio,
    lower_cut_points,
    mean_row_col,
    turn,
)

# (xmin, ymin, xmax, ymax) in pixel coordinates of the original board image
BBox = Tuple[int, int, int, int]


def extract_cards_with_bboxes(
    board_img: np.ndarray,
    background_thres: float = 0.25,
    max_ratio: float = 3,
) -> Tuple[List[np.ndarray], List[BBox]]:
    """Segment a board image into individual cards and return their bounding boxes.

    This mirrors the logic of segmentboard.extract_cards but additionally
    returns the position of each card in the original board image.

    Parameters
    ----------
    board_img : np.ndarray
        RGB board image.
    background_thres : float
        Threshold to discriminate background from card content.
    max_ratio : float
        Maximum long/short edge ratio accepted as a valid card crop.

    Returns
    -------
    cards : list of np.ndarray
        Individual card images in portrait orientation (same as training data).
    bboxes : list of BBox
        Bounding box (xmin, ymin, xmax, ymax) of each card in the original image.
    """
    framed_bin_img = add_frame(binarize_image(board_img))
    mean_row, mean_col = mean_row_col(framed_bin_img)
    xs_cuts = lower_cut_points(mean_row, background_thres).reshape((-1, 2))
    ys_cuts = lower_cut_points(mean_col, background_thres).reshape((-1, 2))

    cards, bboxes = [], []
    for xcut in xs_cuts:
        for ycut in ys_cuts:
            xmin, xmax = int(xcut[0]), int(xcut[1])
            ymin, ymax = int(ycut[0]), int(ycut[1])
            cropped = board_img[ymin:ymax, xmin:xmax]
            if img_ratio(cropped) < max_ratio:
                cards.append(turn(cropped, orient='portrait'))
                bboxes.append((xmin, ymin, xmax, ymax))

    return cards, bboxes


def draw_rectangles(
    board_img: np.ndarray,
    bboxes: List[BBox],
    color: Tuple[int, int, int] = (220, 20, 20),
    thickness: int = 6,
) -> np.ndarray:
    """Draw filled rectangles around the given bounding boxes on a copy of the board image.

    Parameters
    ----------
    board_img : np.ndarray
        Original RGB board image.
    bboxes : list of BBox
        Bounding boxes to highlight.
    color : (R, G, B)
        Rectangle border color. Default is red.
    thickness : int
        Border thickness in pixels.

    Returns
    -------
    np.ndarray
        Annotated copy of the board image (RGB).
    """
    annotated = board_img.copy()
    for xmin, ymin, xmax, ymax in bboxes:
        cv2.rectangle(
            annotated,
            pt1=(xmin, ymin),
            pt2=(xmax, ymax),
            color=color,
            thickness=thickness,
        )
    return annotated


# Visually distinct colors (RGB) cycled across solutions
SOLUTION_COLORS = [
    (220,  20,  20),  # red
    ( 30, 144, 255),  # dodger blue
    ( 50, 205,  50),  # lime green
    (255, 165,   0),  # orange
    (148,   0, 211),  # violet
    (255,  20, 147),  # deep pink
    (  0, 206, 209),  # dark turquoise
    (255, 215,   0),  # gold
]


def draw_all_solutions(
    board_img: np.ndarray,
    solutions_bboxes: List[List[BBox]],
    thickness: int = 15,
) -> np.ndarray:
    """Draw rectangles for every solution, each in a distinct color.

    Parameters
    ----------
    board_img : np.ndarray
        Original RGB board image.
    solutions_bboxes : list of list of BBox
        One inner list per solution; each inner list contains 3 bboxes.
    thickness : int
        Border thickness in pixels.

    Returns
    -------
    np.ndarray
        Annotated copy of the board image (RGB).
    """
    annotated = board_img.copy()
    for i, bboxes in enumerate(solutions_bboxes):
        color = SOLUTION_COLORS[i % len(SOLUTION_COLORS)]
        for xmin, ymin, xmax, ymax in bboxes:
            cv2.rectangle(
                annotated,
                pt1=(xmin, ymin),
                pt2=(xmax, ymax),
                color=color,
                thickness=thickness,
            )
    return annotated


def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """Convert an RGB numpy array to a PIL Image."""
    return Image.fromarray(img)