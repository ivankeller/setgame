import os
import unittest

import cv2
import numpy as np
from PIL import Image

from tests.testbase_class import TestBaseClass
from setgame.draw_solution.draw_solution import (
    SOLUTION_COLORS,
    draw_all_solutions,
    draw_rectangles,
    extract_cards_with_bboxes,
    numpy_to_pil,
)


def _make_board_img(h=200, w=400, n_cards=3) -> np.ndarray:
    """Synthetic board: white cards on a dark background, arranged in a row."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    card_w = w // (n_cards + 1)
    margin = 10
    for i in range(n_cards):
        x1 = (i + 1) * card_w - card_w // 2
        x2 = x1 + card_w - margin
        y1, y2 = 20, h - 20
        img[y1:y2, x1:x2] = 240   # near-white card
    return img


class TestExtractCardsWithBboxes(TestBaseClass):

    def setUp(self):
        self.board_path = os.path.join(self.RESOURCE_DIR, 'test_board.jpg')
        self.board_img  = cv2.cvtColor(cv2.imread(self.board_path), cv2.COLOR_BGR2RGB)

    def test_returns_correct_card_count(self):
        cards, bboxes = extract_cards_with_bboxes(self.board_img)
        self.assertEqual(len(cards), 12)
        self.assertEqual(len(bboxes), len(cards))

    def test_bboxes_within_image_bounds(self):
        h, w = self.board_img.shape[:2]
        _, bboxes = extract_cards_with_bboxes(self.board_img)
        for xmin, ymin, xmax, ymax in bboxes:
            self.assertGreaterEqual(xmin, 0)
            self.assertGreaterEqual(ymin, 0)
            self.assertLessEqual(xmax, w)
            self.assertLessEqual(ymax, h)
            self.assertLess(xmin, xmax)
            self.assertLess(ymin, ymax)

    def test_bboxes_and_cards_consistent(self):
        cards, bboxes = extract_cards_with_bboxes(self.board_img)
        for card, (xmin, ymin, xmax, ymax) in zip(cards, bboxes):
            # Card image dimensions must match the bbox (possibly rotated by turn())
            card_h, card_w = card.shape[:2]
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin
            # After turn(), portrait: height >= width, so short/long edges must match
            self.assertEqual(min(card_h, card_w), min(bbox_h, bbox_w))
            self.assertEqual(max(card_h, card_w), max(bbox_h, bbox_w))

    def test_cards_are_portrait_oriented(self):
        cards, _ = extract_cards_with_bboxes(self.board_img)
        for card in cards:
            h, w = card.shape[:2]
            self.assertGreaterEqual(h, w, msg="Card should be in portrait orientation")

    def test_cards_are_rgb(self):
        cards, _ = extract_cards_with_bboxes(self.board_img)
        for card in cards:
            self.assertEqual(card.ndim, 3)
            self.assertEqual(card.shape[2], 3)


class TestDrawRectangles(unittest.TestCase):

    def setUp(self):
        self.img = np.zeros((100, 200, 3), dtype=np.uint8)
        self.bbox = (10, 10, 50, 50)

    def test_output_shape_same_as_input(self):
        result = draw_rectangles(self.img, [self.bbox])
        self.assertEqual(result.shape, self.img.shape)

    def test_does_not_modify_original(self):
        original = self.img.copy()
        draw_rectangles(self.img, [self.bbox])
        np.testing.assert_array_equal(self.img, original)

    def test_rectangle_pixels_changed(self):
        result = draw_rectangles(self.img, [self.bbox], color=(255, 0, 0))
        # Some pixels along the rectangle border must be red
        xmin, ymin, xmax, ymax = self.bbox
        border_pixels = result[ymin, xmin:xmax]
        self.assertTrue((border_pixels[:, 0] > 0).any(), "Expected red pixels on top border")

    def test_no_bboxes_returns_copy(self):
        result = draw_rectangles(self.img, [])
        np.testing.assert_array_equal(result, self.img)


class TestDrawAllSolutions(unittest.TestCase):

    def setUp(self):
        self.img = np.zeros((200, 400, 3), dtype=np.uint8)
        self.solutions = [
            [(10, 10, 60, 80)],
            [(100, 10, 150, 80)],
            [(200, 10, 250, 80)],
        ]

    def test_does_not_modify_original(self):
        original = self.img.copy()
        draw_all_solutions(self.img, self.solutions)
        np.testing.assert_array_equal(self.img, original)

    def test_output_shape_same_as_input(self):
        result = draw_all_solutions(self.img, self.solutions)
        self.assertEqual(result.shape, self.img.shape)

    def test_each_solution_drawn_in_different_color(self):
        result = draw_all_solutions(self.img, self.solutions)
        # Sample the top-left corner pixel of each bbox and verify they differ
        colors_found = set()
        for bboxes in self.solutions:
            xmin, ymin, _, _ = bboxes[0]
            pixel = tuple(result[ymin, xmin])
            colors_found.add(pixel)
        self.assertGreater(len(colors_found), 1, "Expected different colors per solution")

    def test_color_cycles_for_more_than_palette(self):
        # More solutions than colors in SOLUTION_COLORS — must not raise
        many_solutions = [[(i * 10, 10, i * 10 + 8, 30)] for i in range(len(SOLUTION_COLORS) + 2)]
        try:
            draw_all_solutions(self.img, many_solutions)
        except Exception as e:
            self.fail(f"draw_all_solutions raised unexpectedly: {e}")

    def test_empty_solutions_returns_copy(self):
        result = draw_all_solutions(self.img, [])
        np.testing.assert_array_equal(result, self.img)


class TestNumpyToPil(unittest.TestCase):

    def test_returns_pil_image(self):
        arr = np.zeros((50, 80, 3), dtype=np.uint8)
        result = numpy_to_pil(arr)
        self.assertIsInstance(result, Image.Image)

    def test_size_matches(self):
        arr = np.zeros((50, 80, 3), dtype=np.uint8)
        result = numpy_to_pil(arr)
        self.assertEqual(result.size, (80, 50))   # PIL size is (width, height)

    def test_pixel_values_preserved(self):
        arr = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = numpy_to_pil(arr)
        self.assertEqual(result.getpixel((0, 0)), (128, 128, 128))


if __name__ == '__main__':
    unittest.main()