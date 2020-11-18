import unittest
import cv2
import os

from tests.testbase_class import TestBaseClass
from setgame.segmentboard.segmentboard import extract_cards


class TestExtractCards(TestBaseClass):
    def setUp(self):
        self.jpg_img_path = os.path.join(self.RESOURCE_DIR, 'test_board.jpg')
        self.png_img_path = os.path.join(self.RESOURCE_DIR, 'test_board.png')

    def _test_extract_cards(self, img_path):
        """Test on a test board image, nb of segmented cards and shapes."""
        board_img = cv2.imread(img_path)
        cards = extract_cards(board_img)
        cards_shapes = [card.shape for card in cards]
        # number of cards
        self.assertEqual(len(cards), 12)
        # each card is a 3d array with third dimension equal to 3
        self.assertEqual(set([card_shape[2] for card_shape in cards_shapes]), {3})

    def test_extract_cards_jpg(self):
        self._test_extract_cards(self.jpg_img_path)

    def test_extract_cards_png(self):
        self._test_extract_cards(self.png_img_path)


if __name__ == '__main__':
    unittest.main()

