import unittest

import torch

from setgame.classify_card.dataset import ATTRIBUTES, LABEL_MAPS
from setgame.classify_card.model import SetCardClassifier


class TestSetCardClassifier(unittest.TestCase):

    def setUp(self):
        # pretrained=False avoids network download during tests
        self.model = SetCardClassifier(pretrained=False)
        self.model.eval()

    def test_forward_output_has_all_attribute_keys(self):
        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(set(out.keys()), set(ATTRIBUTES))

    def test_forward_output_shapes(self):
        batch_size = 4
        x = torch.zeros(batch_size, 3, 224, 224)
        with torch.no_grad():
            out = self.model(x)
        for attr in ATTRIBUTES:
            expected_classes = len(LABEL_MAPS[attr])
            self.assertEqual(out[attr].shape, (batch_size, expected_classes),
                             msg=f"Wrong shape for attribute '{attr}'")

    def test_forward_output_is_float(self):
        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = self.model(x)
        for attr in ATTRIBUTES:
            self.assertEqual(out[attr].dtype, torch.float32)

    def test_different_backbones(self):
        # Verify the model can be instantiated with alternative backbones
        model_b0  = SetCardClassifier(backbone='efficientnet_b0', pretrained=False)
        model_r18 = SetCardClassifier(backbone='resnet18',        pretrained=False)
        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            for model in (model_b0, model_r18):
                out = model(x)
                self.assertEqual(set(out.keys()), set(ATTRIBUTES))

    def test_heads_count(self):
        self.assertEqual(len(self.model.heads), len(ATTRIBUTES))

    def test_head_output_dimensions(self):
        for attr in ATTRIBUTES:
            head = self.model.heads[attr]
            self.assertEqual(head.out_features, len(LABEL_MAPS[attr]))


if __name__ == '__main__':
    unittest.main()