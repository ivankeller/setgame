import unittest
import numpy as np
from tests.testbase_class import TestBaseClass
from setgame.segmentboard.segmentboard import add_frame


class TestAddFrame(TestBaseClass):
    def test_add_frame(self):
        img = np.arange(20).reshape((4, 5))
        framed = add_frame(img, width=1, value=0)
        expected = np.array([[0, 0, 0, 0, 0],
                            [0, 6, 7, 8, 0],
                            [0, 11, 12, 13, 0],
                            [0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(framed, expected)


if __name__ == '__main__':
    unittest.main()
