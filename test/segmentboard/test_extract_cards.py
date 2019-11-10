from segmentboard.segmentboard import extract_cards
from utils.format import bgr2rgb
import cv2

TEST_IMG = "./test/test_data/test_board.jpg"


def test_extract_cards():
    """Simple test. Todo: add more test"""
    board_img = bgr2rgb(cv2.imread(TEST_IMG))
    cards = extract_cards(board_img)
    assert len(cards) == 12

