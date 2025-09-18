# ===========================================================================
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/

#    Copyright [2019] [Peng(SoftwareGift)]

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ===========================================================================

# Code base on https://github.com/tonylins/pytorch-mobilenet-v2

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.Hardswish(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.Hardswish(inplace=True)
    )


class SELayer(nn.Module):
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


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, se=False, downsample=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and inp == oup
        hidden_dim = round(inp * expand_ratio)
        self.se = se
        self.downsample = downsample

        layers = []
        if expand_ratio != 1:
            # pointwise
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.Hardswish(inplace=True))

        # depthwise
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.Hardswish(inplace=True))

        # SE layer
        if se:
            layers.append(SELayer(hidden_dim))

        # pointwise linear
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        elif self.downsample is not None:
            return self.downsample(x) + self.conv(x)
        else:
            return self.conv(x)


class FeatherNet(pl.LightningModule):
    def __init__(self, n_class=2, input_size=224, se=False, avgdown=False, width_mult=1.0):
        super(FeatherNet, self).__init__()
        block = InvertedResidual
        self.se = se
        self.avgdown = avgdown
        self.width_mult = width_mult

        input_channel = int(32 * width_mult)
        last_channel = 1024
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        self.features = [conv_bn(3, input_channel, 2)]

        # inverted residual settings: t, c, n, s
        settings = [
            [1, 16, 1, 2],
            [6, 32, 2, 2],
            [6, 48, 6, 2],
            [6, 64, 3, 2],
        ]

        for t, c, n, s in settings:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                downsample = None
                if self.avgdown and i == 0 and stride != 1:
                    downsample = nn.Sequential(
                        nn.AvgPool2d(stride, stride),
                        nn.Conv2d(input_channel, output_channel, 1, bias=False),
                        nn.BatchNorm2d(output_channel)
                    )
                self.features.append(block(input_channel, output_channel, stride, t, se=self.se, downsample=downsample))
                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        self.final_dw = nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False)
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.final_dw(x)
        x = x.mean([2, 3])  # global average pool
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# Two models FeatherNetA and FeatherNetB
# def FeatherNetA():
#     model = FeatherNet(se = True)
#     return model

if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    # model = FeatherNet(se = True)                   # FeatherNetA
    model = FeatherNet(se = True, avgdown=True)     # FeatherNetB
    sample = torch.rand((16, 3, 224, 224))
    logits = model(sample)                   # raw scores (logits)
    probs = F.softmax(logits, dim=1)         # probabilities

    print("Logits:\n", logits)
    print("\nSoftmax probabilities:\n", probs)
    print("\nPredicted class indices:\n", torch.argmax(probs, dim=1))

