import torch.nn as nn

from .blocks import AngleSimpleLinear

class LiveSpoofHead(nn.Module):
    """Binary classification head for live/spoof."""

    def __init__(self, in_features, dropout=0.15):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(in_features),
            nn.Hardswish(inplace=True),
            AngleSimpleLinear(in_features, 2),
        )

    def forward(self, x):
        return self.fc(x)[0]


class DepthHead(nn.Module):
    """Depth map regression head."""

    def __init__(self, in_channels, output_size=(14, 14)):
        super().__init__()
        self.depth = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(output_size, mode='bilinear'),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.depth(x)
