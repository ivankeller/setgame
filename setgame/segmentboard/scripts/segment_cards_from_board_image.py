#!/usr/bin/env python3
"""
Script for segmenting cards from a given board image and save the card images to a given directory.
Usage:
$ ./segment_cards_from_board_image.py -i <input_image> -o <output_dir>
For help do:
$ ./segment_cards_from_board_image.py -h
"""

import getopt
import os
import sys
from setgame.segmentboard.segmentboard import segment_board


def usage(command, more):
    print(more)
    print("Usage:")
    print(command, "-i input_image -o output_dir")
    print("Segment board of cards into individual cards and save to output directory.")
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
        os.mkdir(outputdir)
    return inputpath, outputdir


def main(argv):
    input_board_image_path, outputdir = parse_args(argv)
    segment_board(input_board_image_path, outputdir, img_format='jpg')


if __name__ == "__main__":
    main(sys.argv)
