import torch
import torch.nn as nn
from .cross_entropy import CrossEntropyLoss


class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2, eps=1e-7, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = CrossEntropyLoss(num_classes, **kwargs)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
