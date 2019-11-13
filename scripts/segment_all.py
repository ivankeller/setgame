#!/usr/bin/env python3
"""
Segment all the images of boards in given directory INPUT_DIR
and save the segmented card images to directory OUTPUT_DIR
"""

import os
import segment_cards_from_board_image

INPUT_DIR = "/Users/ivankeller/Projects/setgame/datatest/raw/jpg"
OUTPUT_DIR = "/Users/ivankeller/Projects/setgame/datatest/output_segmented"

for board_img_file in os.listdir(INPUT_DIR):
    if os.path.splitext(board_img_file)[1] in ['.png', '.jpeg', '.jpg']:
        print("Processing board image", board_img_file)
        segment_cards_from_board_image.main(["dummy_command",
                                             '-i', os.path.join(INPUT_DIR, board_img_file),
                                             '-o', OUTPUT_DIR])



