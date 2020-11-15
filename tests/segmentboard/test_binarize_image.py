import unittest
import cv2
import os

from tests.testbase_class import TestBaseClass
from segmentboard.segmentboard import binarize_image


class TestBinarizeImage(TestBaseClass):
    def setUp(self):
        self.img = cv2.imread(os.path.join(self.RESOURCE_DIR, 'lenas_128.png'))
        self.binarized_img = binarize_image(self.img)

    def test_binary(self):
        """ Only 0 and 1 values in image"""
        distinct_values = set(self.binarized_img.flatten())
        self.assertEqual(distinct_values, set([0, 1]))

    def test_dimensions(self):
        """ If img.shape is (w, h, 3) for RGB format then binarized.shape should be (w, h)."""
        self.assertEqual(self.img.shape[:2], self.binarized_img.shape)


if __name__ == '__main__':
    unittest.main()
