

import torch
from torch import nn

from .backbones import build_backbone
from .losses import ReIDLoss
from .midnecks import build_midneck
from .heads import build_head


class ReIDNet(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, backbone_params, midneck_params, head_params, loss_params, feat_after_neck=False):
        super().__init__()

        self.num_classes = num_classes
        self.feat_after_neck = feat_after_neck

        self.backbone, self.in_planes = build_backbone(backbone_params)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.midneck = build_midneck(midneck_params, self.in_planes)
        #self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=(
        #    True if self.midneck is None else False))
        #self.classifier.apply(weights_init_classifier)
        # corresponding to the origin classifier
        self.head = build_head(head_params, self.in_planes, self.num_classes)

        self.loss = ReIDLoss(self.num_classes, self.in_planes, loss_params)

    def forward(self, x, label=None):
        bone_feat = self.gap(self.backbone(x))  # (b, 2048, 1, 1)
        bone_feat = bone_feat.view(
            bone_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.midneck is None:
            feat = bone_feat
        else:
            # normalize for angular softmax
            feat = self.midneck(bone_feat)

        if self.training:
            cls_score = self.head(feat, label)
            # compute acc
            acc = (cls_score.max(1)[1] == label).float().mean()
            total_loss, loss_items = self.loss(cls_score, bone_feat, label)
            return total_loss, acc, loss_items
        else:
            if self.feat_after_neck:
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return bone_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            # if 'classifier' in i:
            #     continue
            self.state_dict()[i].copy_(param_dict[i])


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
