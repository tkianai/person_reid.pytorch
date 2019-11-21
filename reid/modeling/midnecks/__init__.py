
import torch
import torch.nn as nn
from .bnneck import SingleBnNeck, MultiBnNeck


__all__ = ['build_midneck']

class NoneNeck(nn.Module):
    def __init__(self, in_channels, loss_names, **kwargs):
        super().__init__()
        self.loss_names = loss_names

    def forward(self, x):
        outputs = {}
        for name in self.loss_names:
            outputs[name] = x
        
        outputs['merged'] = x
        return outputs


_midneck_factory = {
    'none': NoneNeck,
    'single_bnneck': SingleBnNeck,
    'multi_bnneck': MultiBnNeck,
}


def build_midneck(params, in_channels, loss_names):

    midneck = _midneck_factory[params.name]
    midneck = midneck(in_channels, loss_names)
    return midneck
