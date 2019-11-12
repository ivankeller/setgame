import os
import unittest


class TestBaseClass(unittest.TestCase):
    DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    FIXTURES_DIR = os.path.join(DIR_PATH, 'fixtures')

    def __init__(self, *args, **kwargs):
        super(TestBaseClass, self).__init__(*args, **kwargs)
