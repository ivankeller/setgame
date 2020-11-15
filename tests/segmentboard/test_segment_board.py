import unittest
import tempfile
import cv2
import shutil
import os

from tests.testbase_class import TestBaseClass
from segmentboard.segmentboard import segment_board
from loguru import logger


class TestSegmentBoard(TestBaseClass):

    def test_segment_board(self):
        # arrange
        jpg_board_img_path = os.path.join(self.RESOURCE_DIR, 'test_board.jpg')
        output_dir = tempfile.mkdtemp()

        # act
        segment_board(jpg_board_img_path, output_dir)
        saved_imgs = os.listdir(output_dir)

        # assert
        # nb of segmented card files
        self.assertEqual(len(saved_imgs), 12, msg='wrong number of saved images')
        # individual file names
        self.assertEqual(saved_imgs[0].startswith('test_board'), True)

        # clean
        shutil.rmtree(output_dir)
        logger.debug(f'temp dir {output_dir} removed')


if __name__ == '__main__':
    unittest.main()

