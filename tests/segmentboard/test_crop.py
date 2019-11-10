import numpy as np
from segmentboard.segmentboard import crop


def test_crop():
    img_test = np.arange(20).reshape((4, 5))
    assert (crop(img_test, 2, 5, 0, 2) == np.array([[2, 3, 4], [7, 8, 9]])).all()