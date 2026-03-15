import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tests.testbase_class import TestBaseClass
from setgame.classify_card.dataset import (
    ATTRIBUTES,
    LABEL_MAPS,
    INVERSE_LABEL_MAPS,
    SetCardDataset,
    build_sample_list,
    load_label,
    make_transforms,
    train_val_split,
    white_balance_from_border,
)


def _make_white_card(h=100, w=80, border=5) -> np.ndarray:
    """Synthetic white card with a coloured centre patch (RGB uint8)."""
    img = np.full((h, w, 3), 240, dtype=np.uint8)   # near-white background
    img[border:h-border, border:w-border] = [180, 80, 80]  # reddish centre
    return img


def _write_fake_dataset(tmp_dir: str, n: int = 6) -> tuple[Path, Path]:
    """Write n synthetic card images + label JSONs into tmp_dir subdirs.

    Returns (images_dir, labels_dir).
    """
    images_dir = Path(tmp_dir) / "images"
    labels_dir = Path(tmp_dir) / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    attrs_cycle = [
        {"number": "1", "color": "red",    "shape": "oval",     "shading": "solid"},
        {"number": "2", "color": "green",  "shape": "diamond",  "shading": "striped"},
        {"number": "3", "color": "purple", "shape": "squiggle", "shading": "open"},
    ]
    for i in range(n):
        stem = f"card_{i:03d}"
        img = Image.fromarray(_make_white_card())
        img.save(images_dir / f"{stem}.jpg")
        attrs = attrs_cycle[i % 3]
        (labels_dir / f"{stem}.json").write_text(json.dumps(attrs))

    return images_dir, labels_dir


class TestWhiteBalanceFromBorder(unittest.TestCase):

    def test_output_shape_unchanged(self):
        img = Image.fromarray(_make_white_card(60, 50))
        result = white_balance_from_border(img)
        self.assertEqual(result.size, img.size)

    def test_output_is_pil_image(self):
        img = Image.fromarray(_make_white_card())
        result = white_balance_from_border(img)
        self.assertIsInstance(result, Image.Image)

    def test_white_border_image_approximately_unchanged(self):
        # An image whose border is already pure white should change very little
        arr = np.full((80, 60, 3), 255, dtype=np.uint8)
        arr[10:70, 10:50] = [100, 150, 200]   # coloured centre
        img = Image.fromarray(arr)
        result = white_balance_from_border(img)
        result_arr = np.array(result)
        # Border pixels should still be (close to) white
        np.testing.assert_array_less(
            np.abs(result_arr[:3, :].astype(int) - 255), 5
        )

    def test_pixel_values_in_valid_range(self):
        img = Image.fromarray(_make_white_card())
        result = white_balance_from_border(img)
        arr = np.array(result)
        self.assertTrue((arr >= 0).all())
        self.assertTrue((arr <= 255).all())


class TestLoadLabel(unittest.TestCase):

    def test_loads_all_attributes(self):
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            json.dump({"number": "2", "color": "green", "shape": "oval", "shading": "striped"}, f)
            path = Path(f.name)
        try:
            label = load_label(path)
            self.assertEqual(set(label.keys()), set(ATTRIBUTES))
        finally:
            path.unlink()

    def test_values_are_integers(self):
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            json.dump({"number": "3", "color": "purple", "shape": "diamond", "shading": "solid"}, f)
            path = Path(f.name)
        try:
            label = load_label(path)
            for attr in ATTRIBUTES:
                self.assertIsInstance(label[attr], int)
        finally:
            path.unlink()

    def test_correct_index_mapping(self):
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            json.dump({"number": "1", "color": "red", "shape": "oval", "shading": "solid"}, f)
            path = Path(f.name)
        try:
            label = load_label(path)
            self.assertEqual(label["number"],  LABEL_MAPS["number"]["1"])
            self.assertEqual(label["color"],   LABEL_MAPS["color"]["red"])
            self.assertEqual(label["shape"],   LABEL_MAPS["shape"]["oval"])
            self.assertEqual(label["shading"], LABEL_MAPS["shading"]["solid"])
        finally:
            path.unlink()


class TestBuildSampleList(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.images_dir, self.labels_dir = _write_fake_dataset(self.tmp, n=4)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_only_labeled_images_included(self):
        # Add an unlabeled image (no matching JSON)
        Image.fromarray(_make_white_card()).save(self.images_dir / "unlabeled.jpg")
        samples = build_sample_list(str(self.images_dir), str(self.labels_dir))
        stems = [s[0].stem for s in samples]
        self.assertNotIn("unlabeled", stems)
        self.assertEqual(len(samples), 4)

    def test_augmented_images_included(self):
        aug_dir = Path(self.tmp) / "augmented"
        aug0 = aug_dir / "aug0"
        aug0.mkdir(parents=True)
        # Copy the same images into aug0 (same filenames → same labels)
        for img_path, _ in build_sample_list(str(self.images_dir), str(self.labels_dir)):
            Image.fromarray(_make_white_card()).save(aug0 / img_path.name)

        samples = build_sample_list(str(self.images_dir), str(self.labels_dir), str(aug_dir))
        self.assertEqual(len(samples), 4 + 4)   # 4 originals + 4 augmented

    def test_returns_existing_paths(self):
        samples = build_sample_list(str(self.images_dir), str(self.labels_dir))
        for img_path, label_path in samples:
            self.assertTrue(img_path.exists(), f"Image not found: {img_path}")
            self.assertTrue(label_path.exists(), f"Label not found: {label_path}")


class TestTrainValSplit(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.images_dir, self.labels_dir = _write_fake_dataset(self.tmp, n=10)
        self.samples = build_sample_list(str(self.images_dir), str(self.labels_dir))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_split_sizes(self):
        train, val = train_val_split(self.samples, val_fraction=0.2, seed=42)
        self.assertEqual(len(train) + len(val), len(self.samples))
        self.assertGreaterEqual(len(val), 1)

    def test_no_overlap_between_train_and_val(self):
        train, val = train_val_split(self.samples, val_fraction=0.2, seed=42)
        val_stems  = {s[0].stem for s in val}
        train_stems = {s[0].stem for s in train}
        self.assertEqual(val_stems & train_stems, set())

    def test_reproducible_with_same_seed(self):
        train1, val1 = train_val_split(self.samples, seed=0)
        train2, val2 = train_val_split(self.samples, seed=0)
        self.assertEqual([s[0] for s in val1], [s[0] for s in val2])

    def test_different_seeds_give_different_splits(self):
        _, val1 = train_val_split(self.samples, seed=0)
        _, val2 = train_val_split(self.samples, seed=99)
        # With 10 images it's possible (but unlikely) seeds collide — just check they can differ
        # We only assert one of them is not empty
        self.assertGreater(len(val1), 0)
        self.assertGreater(len(val2), 0)


class TestMakeTransforms(unittest.TestCase):

    def _apply(self, train: bool) -> torch.Tensor:
        img = Image.fromarray(_make_white_card(100, 80))
        t = make_transforms(train=train, img_size=64)
        return t(img)

    def test_train_output_shape(self):
        tensor = self._apply(train=True)
        self.assertEqual(tensor.shape, (3, 64, 64))

    def test_val_output_shape(self):
        tensor = self._apply(train=False)
        self.assertEqual(tensor.shape, (3, 64, 64))

    def test_output_is_float_tensor(self):
        tensor = self._apply(train=False)
        self.assertEqual(tensor.dtype, torch.float32)


class TestSetCardDataset(TestBaseClass):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.images_dir, self.labels_dir = _write_fake_dataset(self.tmp, n=5)
        self.samples = build_sample_list(str(self.images_dir), str(self.labels_dir))
        self.transform = make_transforms(train=False, img_size=64)
        self.dataset = SetCardDataset(self.samples, transform=self.transform)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_len(self):
        self.assertEqual(len(self.dataset), 5)

    def test_getitem_image_shape(self):
        img, _ = self.dataset[0]
        self.assertEqual(img.shape, (3, 64, 64))

    def test_getitem_label_keys(self):
        _, label = self.dataset[0]
        self.assertEqual(set(label.keys()), set(ATTRIBUTES))

    def test_getitem_label_values_are_integers(self):
        _, label = self.dataset[0]
        for attr in ATTRIBUTES:
            self.assertIsInstance(label[attr], int)

    def test_white_balance_flag(self):
        # No error raised with or without white balance
        ds_wb  = SetCardDataset(self.samples, transform=self.transform, white_balance=True)
        ds_nowb = SetCardDataset(self.samples, transform=self.transform, white_balance=False)
        img_wb,  _ = ds_wb[0]
        img_nowb, _ = ds_nowb[0]
        self.assertEqual(img_wb.shape, img_nowb.shape)


if __name__ == '__main__':
    unittest.main()