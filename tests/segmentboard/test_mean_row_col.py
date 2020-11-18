import unittest
import numpy as np
from tests.testbase_class import TestBaseClass
from setgame.segmentboard.segmentboard import mean_row_col


class TestMeanRowCol(TestBaseClass):
    def test_dimensions(self):
        img = np.random.randint(0, 256, size=(10, 6))
        mean_row, mean_col = mean_row_col(img)
        self.assertEqual(mean_row.ndim, 1)
        self.assertEqual(mean_col.ndim, 1)
        self.assertEqual(img.shape, (mean_col.size, mean_row.size))


if __name__ == '__main__':
    unittest.main()
