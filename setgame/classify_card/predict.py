"""
Inference utilities for the Set card attribute classifier.
"""

from pathlib import Path

import torch
from PIL import Image

from setgame.classify_card.dataset import (
    ATTRIBUTES,
    INVERSE_LABEL_MAPS,
    make_transforms,
    white_balance_from_border,
)
from setgame.classify_card.model import SetCardClassifier


def load_model(weights_path: str, device: torch.device | None = None) -> SetCardClassifier:
    """Load a trained SetCardClassifier from a weights file.

    Parameters
    ----------
    weights_path : str
        Path to a .pth file saved by train.train().
    device : torch.device or None
        Target device. Defaults to CPU.

    Returns
    -------
    SetCardClassifier
        Model in eval mode on the requested device.
    """
    if device is None:
        device = torch.device("cpu")
    model = SetCardClassifier(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_card(
    image: Image.Image,
    model: SetCardClassifier,
    device: torch.device | None = None,
    white_balance: bool = True,
) -> dict[str, str]:
    """Predict the four attributes of a single Set card image.

    Parameters
    ----------
    image : PIL.Image.Image
        Segmented card image (RGB).
    model : SetCardClassifier
        Trained model in eval mode.
    device : torch.device or None
        Device to run inference on. Defaults to CPU.
    white_balance : bool
        Whether to apply border-based white balance correction.

    Returns
    -------
    dict
        Keys are attribute names; values are string labels, e.g.:
        {"number": "2", "shape": "oval", "shading": "striped", "color": "red"}
    """
    if device is None:
        device = next(model.parameters()).device

    if white_balance:
        image = white_balance_from_border(image)

    transform = make_transforms(train=False)
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)

    return {
        attr: INVERSE_LABEL_MAPS[attr][logits[attr].argmax(1).item()]
        for attr in ATTRIBUTES
    }


def predict_card_from_path(
    image_path: str,
    model: SetCardClassifier,
    device: torch.device | None = None,
    white_balance: bool = True,
) -> dict[str, str]:
    """Convenience wrapper: load an image from disk and predict its attributes.

    Parameters
    ----------
    image_path : str
        Path to the card image file.

    Returns
    -------
    dict
        Same format as predict_card().
    """
    image = Image.open(image_path).convert("RGB")
    return predict_card(image, model, device, white_balance)