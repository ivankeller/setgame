import unittest
from tests.testbase_class import TestBaseClass
from base_classes import label


class TestLabel(TestBaseClass):
    def test_as_dict(self):
        # arrange
        one_label = label.Label(
            number=label.Number.ONE,
            color=label.Color.RED,
            shape=label.Shape.OVAL,
            shading=label.Shading.OPEN
        )
        # act
        result = one_label.as_dict()
        # assert
        expected = {
            'number': '1',
            'color': 'red',
            'shape': 'oval',
            'shading': 'open'
        }
        self.assertDictEqual(result, expected)

    def test_as_json(self):
        # arrange
        one_label = label.Label(
            number=label.Number.ONE,
            color=label.Color.RED,
            shape=label.Shape.OVAL,
            shading=label.Shading.OPEN
        )
        # act
        result = one_label.as_json()
        # assert
        expected = '{"number": "1", "color": "red", "shape": "oval", "shading": "open"}'
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
