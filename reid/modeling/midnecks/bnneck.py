
import torch
import torch.nn as nn


class BnNeck(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.neck = nn.BatchNorm1d(in_channels)
        self.neck.bias.requires_grad_(False)
        self.neck.apply(_weights_init)

    def forward(self, x):
        return self.neck(x)


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
