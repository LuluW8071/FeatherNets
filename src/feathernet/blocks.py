import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# -------------------------------
# Basic Blocks
# -------------------------------

class AngleSimpleLinear(nn.Module):
    """AM-Softmax style linear layer: computes cosine of angles."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7),


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer for channel attention."""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.Hardswish(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Conv2d_cd(nn.Module):
    """Conv2d with central difference mechanism."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.isclose(self.theta, 0.0):
            return out_normal
        else:
            C_out, C_in, k, _ = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2).view(C_out, C_in, 1, 1)
            out_diff = F.conv2d(x, weight=kernel_diff, bias=self.conv.bias,
                                stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)
            return out_normal - self.theta * out_diff


class InvertedResidual(nn.Module):
    """MobileNetV2-style Inverted Residual Block with optional CD Conv."""

    def __init__(self, inp, oup, stride, expand_ratio, downsample=None):
        super().__init__()
        self.stride = stride
        self.downsample = downsample
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = (self.stride == 1 and inp == oup)

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                Conv2d_cd(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(inplace=True),
                Conv2d_cd(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        elif self.downsample is not None:
            return self.downsample(x) + self.conv(x)
        else:
            return self.conv(x)
