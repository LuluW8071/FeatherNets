import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Original FocalLoss implementation from FeatherNet paper
    """
    def __init__(self, class_num, alpha=None, gamma=2.0, size_average=True, device='cpu'):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.gamma = gamma
        self.size_average = size_average
        self.device = device

        if alpha is None:
            self.alpha = torch.ones(class_num, device=device)
        elif isinstance(alpha, (float, int)):
            # If alpha is a scalar, use it for all classes
            self.alpha = torch.full((class_num,), alpha, device=device, dtype=torch.float32)
        else:
            # If alpha is a list or tensor
            self.alpha = torch.tensor(alpha, device=device, dtype=torch.float32)

    def forward(self, inputs, targets):
        """
        Original FeatherNet FocalLoss forward implementation
        inputs: logits tensor of shape [N, C] 
        targets: labels tensor of shape [N] (0..C-1)
        """
        # N = inputs.size(0)
        # C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        
        class_mask = torch.zeros_like(inputs)
        ids = targets.view(-1,1)
        class_mask.scatter_(1, ids, 1.)
        
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(self.device)
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss