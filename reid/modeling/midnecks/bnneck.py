
import torch
import torch.nn as nn


class SingleBnNeck(nn.Module):

    def __init__(self, in_channels, loss_names, **kwargs):
        super().__init__()
        self.loss_names = loss_names
        self.neck = nn.BatchNorm1d(in_channels)
        self.neck.bias.requires_grad_(False)
        self.neck.apply(_weights_init)

    def forward(self, x):
        output = self.neck(x)
        outputs = {}
        for name in self.loss_names:
            outputs[name] = x
        
        # merged output feature to ID loss
        outputs['merged'] = output
        return outputs


class MultiBnNeck(nn.Module):

    def __init__(self, in_channels, loss_names, **kwargs):
        super().__init__()
        self.loss_names = loss_names
        for name in self.loss_names:
            neck = nn.BatchNorm1d(in_channels)
            neck.bias.requires_grad_(False)
            neck.apply(_weights_init)
            setattr(self, "neck_{}".format(name), neck)

        self.merge_neck = nn.BatchNorm1d(in_channels)
        self.merge_neck.bias.requires_grad_(False)
        self.merge_neck.apply(_weights_init)

    def forward(self, x):
        outputs = {}
        merged = None
        for name in self.loss_names:
            outputs[name] = getattr(self, "neck_{}".format(name))(x)
            if merged is None:
                merged = outputs[name]
            else:
                merged += outputs[name]

        # merge
        merged = merged / float(len(self.loss_names))
        outputs['merged'] = self.merge_neck(merged)
        return outputs


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
