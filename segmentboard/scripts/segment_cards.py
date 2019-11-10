#!/usr/bin/env python3

"""
Segment cards in a given board image and save to a given directory.
type ./segment_cards.py -h for more information
"""

import getopt
import os
import sys

import cv2
from scipy import misc

from segmentboard.segmentboard import extract_cards
from utils.format import bgr2rgb


def usage(command, more):
    print(more)
    print("Usage:")
    print(command, "-i input_image -o output_dir")
    print("Segmen board of cards into individual cards and save to given path.")
    print("-i path \t path to a jpeg or png image of a board of cards.")
    print("-o path \t path to a valid output directory.")
    print("-h (optional)     \t display this usage message.")
    sys.exit(0)


def parse_args(argv):
    try:
        opts, args = getopt.getopt(argv[1:], "h:i:o:")
    except getopt.GetoptError:
        usage(argv[0], "\nBad arguments")
        sys.exit(2)
    inputpath = None
    outputdir = None
    for o, a in opts:
        if o == "-h":
            usage(argv[0], "")
            sys.exit(2)
        elif o == "-i":
            inputpath = a
        elif o == "-o":
            outputdir = a
        else:
            assert False, "Unhandled option"
    if None in [inputpath, outputdir]:
        usage(argv[0], "\n>>> Some argument is missing")
        sys.exit(2)
    if not(os.path.exists(inputpath)):
        sys.exit(' '.join(['Input path error:', inputpath, 'does not exist.']))
    if not(os.path.exists(outputdir)):
        sys.exit(' '.join(['Output path error:', outputdir, 'does not exist.']))
    return (inputpath, outputdir)


def main(argv):
    input_board_image_path, outputdir = parse_args(argv)
    basename = os.path.split(input_board_image_path)[1].split('.')[0]
    board_img = bgr2rgb(cv2.imread(input_board_image_path))
    cards = extract_cards(board_img, verb=False)
    for i, card in enumerate(cards):
        card_path = os.path.join(outputdir, "{0}_{1}.png".format(basename, i))
        misc.imsave(card_path, card)
    print("Saved {0} cards.bkup to directory {1}".format(len(cards), outputdir))


if __name__ == "__main__":
    main(sys.argv)