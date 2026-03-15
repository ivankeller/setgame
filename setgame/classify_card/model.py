"""
Multi-task EfficientNet-B0 classifier for Set card attributes.

The model shares a single EfficientNet-B0 backbone and attaches four
independent linear classification heads — one per attribute:
    number  : 3 classes  (1 / 2 / 3)
    shape   : 3 classes  (diamond / squiggle / oval)
    shading : 3 classes  (solid / striped / open)
    color   : 3 classes  (red / green / purple)
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required. Install it with: pip install timm"
    ) from e

from setgame.classify_card.dataset import ATTRIBUTES, LABEL_MAPS


class SetCardClassifier(nn.Module):
    """EfficientNet-B0 backbone with one linear head per card attribute.

    Parameters
    ----------
    backbone : str
        Any timm model name. Defaults to 'efficientnet_b0'.
    pretrained : bool
        Load ImageNet pretrained weights for the backbone.
    dropout : float
        Dropout probability applied before each classification head.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.dropout = nn.Dropout(p=dropout)
        self.heads = nn.ModuleDict({
            attr: nn.Linear(feat_dim, len(LABEL_MAPS[attr]))
            for attr in ATTRIBUTES
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a dict of raw logits, one tensor per attribute.

        Parameters
        ----------
        x : torch.Tensor
            Batch of images, shape (N, 3, H, W).

        Returns
        -------
        dict mapping attribute name → logit tensor of shape (N, 3)
        """
        features = self.dropout(self.backbone(x))
        return {attr: head(features) for attr, head in self.heads.items()}