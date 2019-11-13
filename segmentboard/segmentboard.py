"""
Module to segment a board of cards into individual cards.
Assumption: 
    - cards are aligned horizontally and vertically to the edges of the image
    - the background is darker enough
"""

import cv2
import imageio
import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from loguru import logger
from utils import read_image


def binarize_image(img):
    """Return a binary image (0-black and 1-white) using Otsu thresholding algorithm.

    Parameters
    ----------
    img : numpy.ndarray
        RBG image

    Returns
    -------
    bin_image : numpy.ndarray
        binarized image, only 0 and 1 values

    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin_image = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_image


def add_frame(img, width=10, value=0):
    """Change values of the edge pixels to a given values.

    Parameters
    ----------
    img : 2d numpy.ndarray
        input 1-channel image, shape (w, h)
    width : int, optional
        width of the frame in pixels
    value : int, optional

    Returns
    -------
    framed : 2d numpy.ndarray

    """
    framed = img.copy()
    framed[:width, :] = value
    framed[-width:, :] = value
    framed[:, :width] = value
    framed[:, -width:] = value
    return framed


def mean_row_col(img, sigma=5):
    """Average pixel values of rows and columns of a 1-channel image, gaussian filtered.

    Parameters
    ----------
    img : 2d numpy.ndarray
        single channel image of shape (nb_of_rows, nb_of_columns)
    sigma : scalar, optional
        standard deviation for Gaussian kernel

    Returns
    -------
    mean_row, mean_col : tuple of 1d numpy arrays
        The mean row and mean column of the input image, filtered by a Gaussian kernel for smoothing.
        mean_row.size is nb_of_columns, and mean_col.size is nb_of_rows

    """
    mean_row = gaussian_filter1d(img.mean(axis=0), sigma)
    mean_col = gaussian_filter1d(img.mean(axis=1), sigma)
    return mean_row, mean_col


def lower_cut_points(data, threshold):
    """Return the indices of a 1d array for the nearest values below a given threshold (+/- 1 index value).

    Ex: 
        >>> arr = np.array([0., 0.1, 0.2, 0.5, 0.5, 0.3, 0.2, 0.1, 0.])
        >>> lower_cut_points(arr, threshold=0.25)
        array([2, 5])
    The lack of accuracy of the index +/- 1 is not an issue for the present use case.

    Parameters
    ----------
    data : 1d numpy.ndarray
        input vector
    threshold : scalar
        lower cut threshold

    Returns
    -------
    numpy.ndarray (int)
        the indices +/- 1 corresponding to the first or last values below the threshold.

    """
    binary = data < threshold
    flips = np.argwhere(np.logical_xor(binary[1:], binary[:-1]))
    return flips.flatten()


def crop(img, xmin, xmax, ymin, ymax):
    """Return a cropped image given x (vertical) and y (horizontal) boundaries.

    Parameters
    ----------
    img : 3d or 2d numpy.ndarray
        input image of shape (Y, X, n_channels) or (Y, X)
    xmin, xmax, ymin, ymax : int
        indices defining the rectangle to crop.

    Returns
    -------
    cropped image : numpy.ndarray
        cropped image of shape (ymax - ymin, xmax - xmin, n_channels) or (ymax - ymin, xmax - xmin)
        (xmin, ymin) corresponds to the upper-left crop point,
        (xmax-1, ymax-1) corresponds to the lower-right crop point.

    """
    return img[ymin:ymax, xmin:xmax]


def extract_cards(board_img, background_thres=0.25):
    """Return a list of individual segmented cards from an image of a board of cards.

    Parameters
    ----------
    board_img : 3d numpy.ndarray
        color image of the board
        Assumption: 
            - cards are aligned horizontally and vertically to the edges of the image
            - background is darker enough (TODO: test and try to quantify the min contrast)
    background_thres : float in [0, 1]
        value to discriminate the background from the mean of pattern inside the cards

    Returns
    -------
    cards : list of numpy.ndarray
        the individual cards

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
    logger.info(f"{len(cards)} cards segmented from board image.")
    return cards


def segment_board(board_path, output_dir, background_thres=0.25, format='jpg'):
    """Segment cards from a image of a board of cards and save them to a directory.

    The file name of the board is used to create the individual card file names.

    Parameters
    ----------
    board_path : str
        input image path of the board of cards
    output_dir : str
        output directory to save the segmented card images
    background_thres : float in [0, 1], optional
        parameter for discriminating the image background from the mean of pattern inside the cards
    format : {'jpg', 'png'}, optional
        output format, '.png' format is slow and takes disk space

    Returns
    -------
    output_dir : str

    """
    basename = os.path.split(board_path)[1].split('.')[0]
    board_img = read_image(board_path)
    logger.info(f'Read board image at {os.path.abspath(board_path)}')
    cards = extract_cards(board_img, background_thres)
    for i, card in enumerate(cards):
        card_path = os.path.join(output_dir, f'{basename}_{i}.{format}')
        imageio.imwrite(card_path, card)
        logger.debug(f"Saved {os.path.abspath(card_path)}")
    logger.info(f"Saved {len(cards)} cards to directory {os.path.abspath(output_dir)}")
