import unittest
import numpy as np
from tests.test_base_class import TestBaseClass
from segmentboard.segmentboard import img_ratio


class TestImgRatio(TestBaseClass):
    def test_img_ratio(self):
        img1 = np.arange(18).reshape((3, 2, 3))
        img2 = np.arange(18).reshape((2, 3, 3))
        img3 = np.arange(12).reshape((2, 2, 3))
        self.assertEqual(img_ratio(img1), 3 / 2)
        self.assertEqual(img_ratio(img2), 3 / 2)
        self.assertEqual(img_ratio(img3), 1.)


if __name__ == '__main__':
    unittest.main()
