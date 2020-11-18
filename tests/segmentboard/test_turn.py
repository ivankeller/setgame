import unittest
import numpy as np
from tests.testbase_class import TestBaseClass
from segmentboard.segmentboard import turn


class TestTurn(TestBaseClass):
    def test_turn_portrait_3channels(self):
        img_portrait = np.arange(18).reshape((3, 2, 3))
        turned = turn(img_portrait, orient='landscape')
        not_turned = turn(img_portrait, orient='portrait')
        expected_turned = np.array(
            [[[3,  4,  5],
              [9, 10, 11],
              [15, 16, 17]],
             [[0,  1,  2],
              [6,  7,  8],
              [12, 13, 14]]])
        self.assertEqual(turned.shape, (2, 3, 3))
        np.testing.assert_array_equal(turned, expected_turned)
        self.assertEqual(not_turned.shape, (3, 2, 3))
        np.testing.assert_array_equal(not_turned, img_portrait)

    def test_turn_landscape_3channels(self):
        img_landscape = np.arange(18).reshape((2, 3, 3))
        turned = turn(img_landscape, orient='portrait')
        not_turned = turn(img_landscape, orient='landscape')
        expected_turned = np.array(
            [[[6, 7, 8],
             [15, 16, 17]],
            [[3, 4, 5],
             [12, 13, 14]],
            [[0, 1, 2],
             [9, 10, 11]]])
        self.assertEqual(turned.shape, (3, 2, 3))
        np.testing.assert_array_equal(turned, expected_turned)
        self.assertEqual(not_turned.shape, (2, 3, 3))
        np.testing.assert_array_equal(not_turned, img_landscape)

    def test_turn_2channels(self):
        img_landscape = np.arange(6).reshape((3, 2))
        turned = turn(img_landscape, orient='portrait')
        expected = np.array([[0, 1], [2, 3], [4, 5]])
        np.testing.assert_array_equal(turned, img_landscape)


if __name__ == '__main__':
    unittest.main()
