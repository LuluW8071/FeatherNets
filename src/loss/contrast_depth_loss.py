import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

def contrast_depth_conv(input, device='cpu'):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''

    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], 
        [[0, 1, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]], 
        [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], 
        [[0, 0, 0], [0, -1, 0], [0, 1, 0]], 
        [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]

    kernel_filter = np.array(kernel_filter_list, np.float32)

    kernel_filter = torch.from_numpy(
        kernel_filter.astype(np.float64)).float().to(device)
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)

    input = input.unsqueeze(dim=1).expand(
        input.shape[0], 8, input.shape[1], input.shape[2])

    contrast_depth = F.conv2d(
        input, weight=kernel_filter, groups=8)  # depthwise conv

    return contrast_depth


# Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
class Contrast_depth_loss(nn.Module):
    def __init__(self, device='cpu'):
        super(Contrast_depth_loss, self).__init__()
        self.device = device
        return

    def forward(self, out, label):
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out, device=self.device)
        contrast_label = contrast_depth_conv(label, device=self.device)

        criterion_MSE = nn.MSELoss().to(self.device)

        loss = criterion_MSE(contrast_out, contrast_label)
        # loss = torch.pow(contrast_out - contrast_label, 2)
        # loss = torch.mean(loss)

        return loss
