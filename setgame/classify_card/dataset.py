"""
Dataset utilities for Set card attribute classification.

Labels are read from JSON files (one per image) with the format:
    {"number": "2", "color": "purple", "shape": "squiggle", "shading": "striped"}

Augmented images in data/3_augmented/aug{N}/ share filenames with originals
and therefore inherit their labels from data/2_labels/.
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Maps attribute string values → integer class indices
LABEL_MAPS = {
    "number":  {"1": 0, "2": 1, "3": 2},
    "shape":   {"diamond": 0, "squiggle": 1, "oval": 2},
    "shading": {"solid": 0, "striped": 1, "open": 2},
    "color":   {"red": 0, "green": 1, "purple": 2},
}

# Inverse maps for decoding predictions back to strings
INVERSE_LABEL_MAPS = {
    attr: {v: k for k, v in m.items()}
    for attr, m in LABEL_MAPS.items()
}

ATTRIBUTES = list(LABEL_MAPS.keys())


def white_balance_from_border(img: Image.Image, border_fraction: float = 0.05) -> Image.Image:
    """Correct white balance using the card's white border as an illuminant reference.

    The card border (outer N% of pixels) is assumed to be white. The 95th
    percentile per channel is used as the illuminant estimate (robust to
    shadow/noise), then each channel is scaled so that the illuminant maps to
    255.

    Parameters
    ----------
    img : PIL.Image.Image
        Input RGB card image.
    border_fraction : float
        Fraction of the image height/width to sample as the border.

    Returns
    -------
    PIL.Image.Image
        White-balance-corrected RGB image.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    bh = max(1, int(h * border_fraction))
    bw = max(1, int(w * border_fraction))

    border_pixels = np.concatenate([
        arr[:bh, :].reshape(-1, 3),    # top strip
        arr[-bh:, :].reshape(-1, 3),   # bottom strip
        arr[:, :bw].reshape(-1, 3),    # left strip
        arr[:, -bw:].reshape(-1, 3),   # right strip
    ])

    illuminant = np.percentile(border_pixels, 95, axis=0).clip(1, None)
    corrected = (arr * (255.0 / illuminant)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(corrected)


def load_label(label_path: Path) -> dict:
    """Load a label JSON file and return a dict of {attribute: class_index}.

    Parameters
    ----------
    label_path : Path
        Path to the JSON label file.

    Returns
    -------
    dict
        Keys are attribute names; values are integer class indices.
    """
    with open(label_path) as f:
        raw = json.load(f)
    return {attr: LABEL_MAPS[attr][raw[attr]] for attr in ATTRIBUTES}


def build_sample_list(
    originals_dir: str,
    labels_dir: str,
    augmented_dir: str | None = None,
) -> list[tuple[Path, Path]]:
    """Return a list of (image_path, label_path) pairs for all labeled cards.

    Images in augmented_dir/aug{N}/ that share a stem with a label file are
    included automatically.

    Parameters
    ----------
    originals_dir : str
        Directory containing the original segmented card images.
    labels_dir : str
        Directory containing the JSON label files.
    augmented_dir : str or None
        Parent directory of aug0/, aug1/, … sub-directories. Pass None to
        skip augmented data.

    Returns
    -------
    list of (image_path, label_path) tuples
    """
    originals_dir = Path(originals_dir)
    labels_dir = Path(labels_dir)

    # Only include images that have a corresponding label file
    labeled_stems = {p.stem for p in labels_dir.glob("*.json")}

    samples = []

    # Original images
    for img_path in sorted(originals_dir.glob("*.jpg")):
        if img_path.stem in labeled_stems:
            samples.append((img_path, labels_dir / f"{img_path.stem}.json"))

    # Augmented images (same filenames, labels come from originals dir)
    if augmented_dir is not None:
        aug_parent = Path(augmented_dir)
        for aug_subdir in sorted(aug_parent.iterdir()):
            if not aug_subdir.is_dir() or aug_subdir.name.startswith("."):
                continue
            for img_path in sorted(aug_subdir.glob("*.jpg")):
                if img_path.stem in labeled_stems:
                    samples.append((img_path, labels_dir / f"{img_path.stem}.json"))

    return samples


def train_val_split(
    samples: list[tuple[Path, Path]],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """Split samples into train and validation sets.

    Validation uses only original images (no augmented copies of a card that
    appears in val should be in train, so the split is done by image stem).

    Parameters
    ----------
    samples : list of (image_path, label_path)
        Full sample list as returned by build_sample_list().
    val_fraction : float
        Fraction of *original* images to hold out for validation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_samples, val_samples : two lists of (image_path, label_path)
        val_samples contains only original images; train_samples contains the
        remaining originals plus all augmented images.
    """
    # Separate originals from augmented (augmented paths contain aug{N} in their parent dir name)
    originals = [s for s in samples if not s[0].parent.name.startswith("aug")]
    augmented = [s for s in samples if s[0].parent.name.startswith("aug")]

    rng = random.Random(seed)
    shuffled = list(originals)
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction))
    val_stems = {s[0].stem for s in shuffled[:n_val]}

    val_samples = [s for s in originals if s[0].stem in val_stems]
    train_originals = [s for s in originals if s[0].stem not in val_stems]

    # Keep only augmented images whose stem is NOT in val_stems
    train_augmented = [s for s in augmented if s[0].stem not in val_stems]

    train_samples = train_originals + train_augmented
    return train_samples, val_samples


# ImageNet normalisation statistics (used by all timm pretrained models)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def make_transforms(train: bool, img_size: int = 224) -> transforms.Compose:
    """Return the torchvision transform pipeline for training or validation.

    Color jitter is limited to brightness and contrast only — hue and
    saturation are intentionally left at 0 because color (red/green/purple)
    is one of the four attributes being classified.

    Parameters
    ----------
    train : bool
        If True, include data-augmentation transforms; otherwise only resize
        and normalise.
    img_size : int
        Target spatial size for the model input.

    Returns
    -------
    transforms.Compose
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomRotation(degrees=20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.0,  # must stay 0: saturation shift could confuse color labels
                hue=0.0,         # must stay 0: hue shift would corrupt color labels
            ),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])


class SetCardDataset(Dataset):
    """PyTorch Dataset for Set card attribute classification.

    Each item is a (image_tensor, labels) pair where labels is a dict:
        {"number": int, "shape": int, "shading": int, "color": int}

    Parameters
    ----------
    samples : list of (image_path, label_path) tuples
        As returned by build_sample_list() / train_val_split().
    transform : torchvision.transforms.Compose
        Transform pipeline applied after white-balance correction.
    white_balance : bool
        Whether to apply per-card white balance correction from the border.
    """

    def __init__(self, samples: list, transform=None, white_balance: bool = True):
        self.samples = samples
        self.transform = transform
        self.white_balance = white_balance

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.white_balance:
            image = white_balance_from_border(image)
        if self.transform:
            image = self.transform(image)
        labels = load_label(label_path)
        return image, labels
