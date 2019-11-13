import cv2


def bgr2rgb(img):
    """Return RGB image from BGR image.

    Parameters
    ----------
    img : 3d numpy.ndarray
        BGR image as returned by cv2.imread(file)

    Returns
    -------
    numpy array 
        RGB image

    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
