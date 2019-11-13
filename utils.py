import cv2


def bgr2rgb(img):
    """Return RGB image from BGR image.

    Parameters
    ----------
    img : 3d numpy.ndarray
        BGR image as returned by cv2.imread(filepath)

    Returns
    -------
    3d numpy.ndarray
        RGB image

    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_image(path):
    """Read RGB image.

    Parameters
    ----------
    path : str
        image file path

    Returns
    -------
    numpy.ndarray
        RGB representation of the image, shape (w, h, 3)

    """
    return bgr2rgb(cv2.imread(path))


