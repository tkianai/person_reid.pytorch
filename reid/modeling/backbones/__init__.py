
from .resnet import *

__all__ = ['build_backbone']


_backbone_factory = {
    # name: [func, out_channels]
    'resnet50': [resnet50, 2048]
}


def build_backbone(params):
    if params.name not in _backbone_factory:
        raise NotImplementedError("unknow backbone name [{}]".format(params.name))

    func, out_channels = _backbone_factory[params.name]
    backbone = func(params.last_stride)
    if params.pretrained:
        backbone.load_param(params.pretrained)
    return backbone, out_channels