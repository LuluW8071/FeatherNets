import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# -------------------------------
# Basic Blocks
# -------------------------------

class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weight vectors (AM-Softmax style)."""

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
        super(SELayer, self).__init__()
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
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
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
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
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


# -------------------------------
# FeatherNet Model
# -------------------------------

class FeatherNet(nn.Module):
    def __init__(self, input_channel=32, width_mult=1.,
                 interverted_residual_setting=None, use_depth=True):
        """
        FeatherNet for binary spoof detection.

        Args:
            input_channel (int): initial channel size
            width_mult (float): width multiplier
            interverted_residual_setting (list): IR block configs
            use_depth (bool): whether to output depth map
        """
        super(FeatherNet, self).__init__()

        if interverted_residual_setting is None:
            interverted_residual_setting = [
                # t (expand), c (channels), n (repeats), s (stride)
                [1, 16, 1, 2],   # 112x112
                [6, 32, 3, 2],   # 56x56
                # [6, 48, 6, 2],   # 28x28
                [6, 64, 6, 2],   # 14x14
                [6, 96, 4, 2],   # 7x7
            ]

        self.input_channel = int(input_channel * width_mult)
        self.interverted_residual_setting = interverted_residual_setting
        self.use_depth = use_depth

        # Stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.input_channel),
            nn.Hardswish(inplace=True),
        )

        # Body (IR blocks)
        self.layer1 = self._make_layer(InvertedResidual, 1)
        self.layer2 = self._make_layer(InvertedResidual, 2)
        self.layer3 = self._make_layer(InvertedResidual, 3)
        self.layer4 = self._make_layer(InvertedResidual, 4)

        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Facial marks for semandtic informaton 
        # self.fc_face = nn.Sequential(
        #     nn.Dropout(p=0.15),
        #     nn.BatchNorm1d(self.interverted_residual_setting[-1][1]),
        #     nn.Hardswish(inplace=True),
        #     AngleSimpleLinear(self.interverted_residual_setting[-1][1], 40),
        # )

        # Binary live/spoof head
        self.fc_live = nn.Sequential(
            nn.Dropout(p=0.15),
            nn.BatchNorm1d(self.interverted_residual_setting[-1][1]),
            nn.Hardswish(inplace=True),
            AngleSimpleLinear(self.interverted_residual_setting[-1][1], 2),
        )

        # Depth head (optional)
        if self.use_depth:
            self.depth = nn.Sequential(
                nn.Conv2d(self.interverted_residual_setting[-1][1],
                          1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Upsample((14, 14), mode='bilinear'),
                nn.Sigmoid(),
            )

        # Initialization
        self._initialize_weights()

    def _make_layer(self, block, layer_no):
        t, c, n, s = self.interverted_residual_setting[layer_no - 1]
        output_channel = int(c)
        layer = []
        for i in range(n):
            downsample = None
            if i == 0:
                downsample = nn.Sequential(
                    nn.AvgPool2d(2, stride=2),
                    nn.BatchNorm2d(self.input_channel),
                    nn.Conv2d(self.input_channel, output_channel, 1, bias=False)
                )
                layer.append(block(self.input_channel, output_channel, s,
                                   expand_ratio=t, downsample=downsample))
            else:
                layer.append(block(self.input_channel, output_channel, 1,
                                   expand_ratio=t))
            self.input_channel = output_channel
        layer.append(SELayer(self.input_channel))
        return nn.Sequential(*layer)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, Conv2d_cd):
                nn.init.kaiming_normal_(m.conv.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        """Default forward: return binary live/spoof logits."""
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x).view(x.size(0), -1)
        x_live = self.fc_live(x)[0]  # binary logits
        return x_live

    def _forward_train(self, x):
        """
        Training forward: return both logits and auxiliary outputs.
        Returns:
            [x_live, x_depth] if depth is enabled
            [x_live] otherwise
        """
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_feat = self.avgpool(x).view(x.size(0), -1)
        x_live = self.fc_live(x_feat)[0]

        if self.use_depth:
            x_depth = self.depth(x)
            return [x_live, x_depth]
        else:
            return [x_live]


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    model = FeatherNet()
    sample = torch.rand((8, 3, 224, 224))

    # Training mode for multi task criterion
    model.train()
    x_live, x_depth = model.forward_train(sample)
    print("Train outputs:", x_live.shape, x_depth.shape)
    print(x_live, x_depth)

    # Inference mode
    model.eval()
    logits = model.forward(sample)
    print("Inference logits:", logits.shape)
    probs = F.softmax(logits, dim=-1)
    print("Inference probs:", probs.shape)
    print(probs)
