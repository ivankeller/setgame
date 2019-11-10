"""
Module to segment a board of cards into individual cards.
Assumption: 
    - cards are aligned horizontally and vertically to the edges of the image
    - the background is darker enough (test and try to quantify the min contrast)
"""

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


def binarize_image(img):
    """Return a binary image (0-black and 1-white) using Otsu thresholding algorithm.

    Parameters
    ----------
    img : numpy array
        RBG image

    Returns
    -------
    bin_image : numpy array
        binarized image, 0 and 1 values

    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin_image = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_image


def add_frame(img, width=10, value=0):
    """Change edge pixels to a specific value.

    The pixels at a distance less than 'witdh' from the four image edges are set to 'value' (default: 0, black).

    Parameters
    ----------
    img : 2d numpy array
        input 1-channel image
    width : int (optional)
        width of the frame
    value : int (optional)

    Returns
    -------
    framed : 2d numpy array

    """
    framed = img.copy()
    framed[:width, :] = value
    framed[-width:, :] = value
    framed[:, :width] = value
    framed[:, -width:] = value
    return framed


def mean_row_col(img, sigma=5):
    """Return mean row and column of a 1-channel image, gaussian filtered.

    Parameters
    ----------
    img : 2d numpy array 
        single channel image  
    sigma : scalar (optional)
        standard deviation for Gaussian kernel

    Returns
    -------
    mean_row, mean_col : tuple of 1d numpy arrays
        The mean row and mean column of the input image, filtered by Gaussian kernel for smoothing purposes.

    """
    mean_row = gaussian_filter1d(img.mean(axis=0), sigma)
    mean_col = gaussian_filter1d(img.mean(axis=1), sigma)
    return mean_row, mean_col


def lower_cut_points(data, threshold):
    """Return the indices of the nearest values below a given threshold (+/- 1 index value).

    Ex: 
        > arr = np.array([0., 0.1, 0.2, 0.5, 0.5, 0.3, 0.2, 0.1, 0.])
        > segb.lower_cut_points(arr, threshold=0.25)
        array([2, 5])
    The lack of accuracy of the index +/- 1 is not an issue for the present use case.

    Parameters
    ----------
    data : 1d numpy array 
        input vector
    threshold : scalar
        lower cut threshold

    Returns
    -------
    numpy array (int)
        the indices corresponding to the first or last values below the threshold (+/- 1 index value)

    """
    binary = data < threshold
    flips = np.argwhere(np.logical_xor(binary[1:], binary[:-1]))
    return flips.flatten()


def crop(img, xmin, xmax, ymin, ymax):
    """Return a cropped image given x (vertical) and y (horizontal) boundaries.

    Parameters
    ----------
    img : numpy array
        input image of shape (Y, X, n_channels) or (Y, X)
    xmin, xmax, ymin, ymax : int
        indices defining the rectangle to crop.

    Returns
    -------
    cropped image : numpy array
        cropped image of shape (ymax - ymin, xmax - xmin, n_channels) or (ymax - ymin, xmax - xmin)

    """
    return img[ymin:ymax, xmin:xmax]


def extract_cards(board_img, background_thres=0.25, verb=True):
    """Return a list of individual cards.bkup from an image of the board.

    Parameters
    ----------
    board_img : numpy array
        image of the board
        Assumption: 
            - cards.bkup are aligned horizontally and vertically to the edges of the image
            - background is darker enough (test and try to quantify the min contrast)
    background_thres : float in [0, 1]
        value to discriminate the background from the mean of pattern inside the cards.bkup

    Returns
    -------
    cards.bkup : list of numpy arrays
        the individual cards.bkup

    """

    framed_bin_img = add_frame(binarize_image(board_img))
    mean_row, mean_col = mean_row_col(framed_bin_img)
    xs_cuts = lower_cut_points(mean_row, background_thres).reshape((-1, 2))
    ys_cuts = lower_cut_points(mean_col, background_thres).reshape((-1, 2))
    cards = []
    for xcut in xs_cuts:
        for ycut in ys_cuts:
            cropped = crop(board_img, *(xcut.tolist() + ycut.tolist()))
            cards.append(cropped)
    if verb:
        print("{0} cards.bkup segmented.".format(len(cards)))
    return cards
