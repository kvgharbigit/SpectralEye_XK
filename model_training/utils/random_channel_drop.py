import torch
from torch import nn


class RandomChannelDrop(nn.Module):
    def __init__(self, drop_prob=0.1):
        super(RandomChannelDrop, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x

        batch_size, num_channels, height, width = x.size()
        drop_mask = torch.rand(batch_size, num_channels, 1, 1, device=x.device) > self.drop_prob
        return x * drop_mask
