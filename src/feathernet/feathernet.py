import torch.nn as nn

from .blocks import InvertedResidual, SELayer
from .heads import LiveSpoofHead, DepthHead

class FeatherNet(nn.Module):
    def __init__(self, input_channel=32, width_mult=1.,
                 interverted_residual_setting=None, use_depth=True):
        super().__init__()
        if interverted_residual_setting is None:
            interverted_residual_setting = [
                # t (expand), c (channels), n (repeats), s (stride) 
                [1, 16, 1, 2],  # 112x112
                [6, 32, 3, 2],  # 56x56
                [6, 64, 6, 2],  # 14x14
                [6, 96, 4, 2],  # 7x7
            ]

        self.input_channel = int(input_channel * width_mult)
        self.use_depth = use_depth
        self.interverted_residual_setting = interverted_residual_setting

        # Stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.input_channel),
            nn.Hardswish(inplace=True),
        )

        # Body
        self.layer1 = self._make_layer(1)
        self.layer2 = self._make_layer(2)
        self.layer3 = self._make_layer(3)
        self.layer4 = self._make_layer(4)

        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Heads
        last_channel = self.interverted_residual_setting[-1][1]
        self.fc_live = LiveSpoofHead(last_channel)
        if self.use_depth:
            self.depth_head = DepthHead(last_channel)

        self._initialize_weights()

    def _make_layer(self, layer_no):
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
            layer.append(InvertedResidual(self.input_channel, output_channel,
                                          s if i == 0 else 1, expand_ratio=t,
                                          downsample=downsample))
            self.input_channel = output_channel
        layer.append(SELayer(self.input_channel))
        return nn.Sequential(*layer)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Inference forward: binary logits only"""
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return self.fc_live(x)

    def _forward_train(self, x):
        """Training forward: [logits, depth] if depth used"""
        x_feat = self.conv1(x)
        x_feat = self.layer1(x_feat)
        x_feat = self.layer2(x_feat)
        x_feat = self.layer3(x_feat)
        x_feat = self.layer4(x_feat)

        x_pool = self.avgpool(x_feat).view(x.size(0), -1)
        x_live = self.fc_live(x_pool)
        if self.use_depth:
            x_depth = self.depth_head(x_feat)
            return x_live, x_depth
        else:
            return x_live
