import unittest
import numpy as np
from tests.testbase_class import TestBaseClass
from setgame.segmentboard.segmentboard import lower_cut_points


class TestLowerCutPoints(TestBaseClass):
    def test_lower_cut_points(self):
        arr = np.array([0., 0.1, 0.2, 0.5, 0.5, 0.3, 0.2, 0.1, 0.])
        result = lower_cut_points(arr, threshold=0.25)
        expected = np.array([2, 5])
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
