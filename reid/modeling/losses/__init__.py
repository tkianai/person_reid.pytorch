

import torch
import torch.nn as nn
import torch.nn.functional as F

from .center import CenterLoss
from .triplet import TripletLoss
from .cross_entropy import CrossEntropyLoss
from .focal import FocalLoss
from .ranked import RankedLoss


__all__ = ['ReIDLoss']

class ReIDLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, params):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.ce, self.triplet, self.center, self.focal, self.ranked = None, None, None, None, None
        self.ce_weight, self.triplet_weight, self.center_weight, self.focal_weight, self.ranked_weight = params.crossentropy.weight, params.triplet.weight, params.center.weight, params.focal.weight, params.ranked.weight

        if 'crossentropy' in params.name:
            self.ce = CrossEntropyLoss(
                num_classes=num_classes, label_smooth=params.crossentropy.label_smooth)

        if 'triplet' in params.name:
            self.triplet = TripletLoss(params.triplet.margin)

        if 'center' in params.name:
            self.center = CenterLoss(
                num_classes=num_classes, feat_dim=feat_dim)

        # NOTE `crossentropy` and `focal` should be mutually exclusive
        if 'focal' in params.name:
            self.focal = FocalLoss(
                num_classes=num_classes, label_smooth=params.crossentropy.label_smooth)

        if 'ranked' in params.name:
            self.ranked = RankedLoss()


    def forward(self, score, feat, target):

        loss_items = {k: v for k, v in zip(
            ['crossentropy', 'triplet', 'center', 'focal', 'ranked'], [None, None, None, None, None])}
        if self.ce is not None:
            #print('target: {}'.format(target.shape))
            #print('score: {}'.format(score.shape))
            loss_items['crossentropy'] = self.ce(
                score, target) * self.ce_weight
        if self.triplet is not None:
            loss_items['triplet'] = self.triplet(feat, target) * self.triplet_weight
        if self.center is not None:
            loss_items['center'] = self.center(feat, target) * self.center_weight
        if self.focal is not None:
            loss_items['focal'] = self.focal(score, target) * self.focal_weight
        if self.ranked is not None:
            loss_items['ranked'] = self.ranked(feat, target) * self.ranked_weight

        total_loss = None
        for k, v in loss_items.items():
            if total_loss is None:
                if v is not None:
                    total_loss = v
            else:
                if v is not None:
                    total_loss += v

        return total_loss, loss_items
