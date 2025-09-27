import torch.nn as nn
import torch.nn.functional as F

from loss.amSoftmax_loss import AMSoftmaxLoss
from loss.contrast_depth_loss import Contrast_depth_loss


class MultiTaskCriterion(nn.Module):
    """
    Multi-task loss for spoof classification + depth supervision.
    
    - output -> (x_live, x_depth)
        x_live: [batch, 2]  (logits for live vs spoof)
        x_depth: [batch, 1, H, W]  (predicted depth map)
    
    - target -> (spoof_labels, depth_maps)
        spoof_labels: [batch] (0 = real, 1 = spoof)
        depth_maps:  [batch, 1, H, W] (normalized depth ground truth)
    """
    def __init__(self, C: float = 1.0, Cd: float = 0.2, alpha: float | list[float] = 0.65,
                 gamma: float = 2.0, device: str = 'cpu'):
        super().__init__()
        self.C = C
        self.Cd = Cd
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

        self.spoof_loss_fn = AMSoftmaxLoss(
            margin_type='cross_entropy',
            label_smooth=False,
            smoothing=0.1,
            ratio=[1, 1],
            m=0.5,
            s=1,
            alpha=alpha,
            gamma=gamma,
            device=device
        )
        self.depth_loss_fn = Contrast_depth_loss(device=device)

    def forward(self, output: tuple, target: tuple, eval: bool = False):
        spoof_labels, depth_maps = target
        # print(target)

        # Classification loss
        spoof_target = F.one_hot(spoof_labels.long(), num_classes=2).to(self.device)
        spoof_loss = self.spoof_loss_fn(output[0], spoof_target)

        if eval:
            return spoof_loss

        # Depth supervision loss
        depth_loss = self.depth_loss_fn(output[1].squeeze(1), depth_maps.squeeze(1).to(self.device))

        # Final combined loss
        loss = self.C * spoof_loss + self.Cd * depth_loss

        return spoof_loss, depth_loss, loss