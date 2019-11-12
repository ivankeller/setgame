import unittest
import cv2
import os

from tests.test_base_class import TestBaseClass
from segmentboard.segmentboard import extract_cards


class TestExtractCards(TestBaseClass):

    def test_extract_cards_simple(self):
        """Test on a test board image, nb of segmented cards and shapes."""
        board_img = cv2.imread(os.path.join(self.FIXTURES_DIR, 'test_board.jpg'))
        cards = extract_cards(board_img)
        cards_shapes = [card.shape for card in cards]
        # number of cards
        self.assertEqual(len(cards), 12)
        # each card is a 3d array with third dimension equal to 3
        self.assertEqual(set([card_shape[2] for card_shape in cards_shapes]), {3})


if __name__ == '__main__':
    unittest.main()

