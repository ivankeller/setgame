import unittest
from tests.testbase_class import TestBaseClass
import utils


class TestUtils(TestBaseClass):
    def test_list_files_with_extension(self):
        directory = f'{self.RESOURCE_DIR}/utils'
        extension = '.PNG'
        result = utils.list_files_with_extension(directory, extension)
        expected = [f'{directory}/{file}' for file in ['image2.PNG', 'image3.PNG']]
        self.assertListEqual(result, expected)

    def test_list_files_with_extension_return_empty(self):
        directory = f'{self.RESOURCE_DIR}/utils'
        extension = '.py'
        result = utils.list_files_with_extension(directory, extension)
        expected = []
        self.assertListEqual(result, expected)

    def test_list_images_in_directory(self):
        directory = f'{self.RESOURCE_DIR}/utils'
        result = utils.list_images_in_directory(directory)
        expected = [f'{directory}/{file}' for file in ['image1.jpg', 'image2.PNG', 'image3.PNG']]
        self.assertListEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
