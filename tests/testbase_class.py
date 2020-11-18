import os
import unittest


class TestBaseClass(unittest.TestCase):
    DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    RESOURCE_DIR = os.path.join(DIR_PATH, 'resource')

    def __init__(self, *args, **kwargs):
        super(TestBaseClass, self).__init__(*args, **kwargs)
