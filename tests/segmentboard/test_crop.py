import unittest
import numpy as np
from tests.testbase_class import TestBaseClass
from setgame.segmentboard.segmentboard import crop


class TestCrop(TestBaseClass):
    def test_crop(self):
        img = np.arange(20).reshape((4, 5))
        cropped = crop(img, 2, 5, 0, 2)
        expected = np.array([[2, 3, 4], [7, 8, 9]])
        np.testing.assert_array_equal(cropped, expected)


if __name__ == '__main__':
    unittest.main()
