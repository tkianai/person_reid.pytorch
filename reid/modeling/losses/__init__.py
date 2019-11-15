

import torch
import torch.nn as nn
import torch.nn.functional as F

from .center import CenterLoss
from .triplet import TripletLoss
from .cross_entropy import CrossEntropyLoss


__all__ = ['ReIDLoss']

class ReIDLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, params):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.ce, self.triplet, self.center = None, None, None
        self.ce_weight, self.triplet_weight, self.center_weight = params.softmax.loss_weight, params.triplet.loss_weight, params.center.loss_weight
        if 'softmax' in params.name:
            self.ce = CrossEntropyLoss(
                num_classes=num_classes, label_smooth=params.softmax.label_smooth)

        if 'triplet' in params.name:
            self.triplet = TripletLoss(params.triplet.margin)

        if 'center' in params.name:
            self.center = CenterLoss(
                num_classes=num_classes, feat_dim=feat_dim)

    def forward(self, score, feat, target):

        loss_items = {k: v for k, v in zip(
            ['softmax', 'triplet', 'center'], [None, None, None])}
        if self.ce is not None:
            #print('target: {}'.format(target.shape))
            #print('score: {}'.format(score.shape))
            loss_items['softmax'] = self.ce(score, target) * self.ce_weight
        if self.triplet is not None:
            loss_items['triplet'] = self.triplet(feat, target) * self.triplet_weight
        if self.center is not None:
            loss_items['center'] = self.center(feat, target) * self.center_weight

        total_loss = None
        for k, v in loss_items.items():
            if total_loss is None:
                if v is not None:
                    total_loss = v
            else:
                if v is not None:
                    total_loss += v

        return total_loss, loss_items
