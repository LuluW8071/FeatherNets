import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal_loss import FocalLoss  # import the class

class AMSoftmaxLoss(nn.Module):
    """Computes the AM-Softmax loss with cos or arc margin"""
    margin_types = ['cos', 'arc', 'cross_entropy']

    def __init__(
            self, 
            margin_type='cos', 
            device='cpu', 
            num_classes=2,
            label_smooth=False, 
            smoothing=0.1, 
            ratio=(1, 1), 
            alpha=0.75,
            gamma=2.0,
            m=0.5, 
            s=30, 
            t=1.
            ):
        super().__init__()
        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        self.classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.m = torch.Tensor([m/i for i in ratio]).to(device)
        self.s = s if margin_type in ('arc', 'cos') else 1
        self.t = t
        self.label_smooth = label_smooth
        self.smoothing = smoothing
        self.device = device

    def forward(self, cos_theta, target):
        """target: one-hot vector"""
        if isinstance(cos_theta, tuple):
            cos_theta = cos_theta[0]

        if self.label_smooth:
            target = self._label_smoothing(classes=self.classes, y_hot=target, smoothing=self.smoothing)

        # Convert one-hot to class indices
        fold_target = target.argmax(dim=1)
        one_hot_target = F.one_hot(fold_target, num_classes=self.classes)

        # Margin application
        m = self.m * one_hot_target
        if self.margin_type == 'cos':
            phi_theta = cos_theta - m
            output = phi_theta
        elif self.margin_type == 'arc':
            theta = torch.acos(cos_theta)
            phi_theta = torch.cos(theta + self.m)
            output = phi_theta
        else:  # cross_entropy
            output = cos_theta

        if self.gamma > 0:
            focal = FocalLoss(class_num=self.classes, alpha=self.alpha, gamma=self.gamma, device=self.device)
            loss = focal(inputs=self.s*output, targets=fold_target)
        else:
            pred = F.log_softmax(self.s*output, dim=-1)
            loss = torch.mean(torch.sum(-target * pred, dim=-1))

        return loss

    def _label_smoothing(self, classes, y_hot, smoothing=0.1):
        lam = 1 - smoothing
        return torch.where(y_hot.bool(), lam, smoothing/(classes-1))