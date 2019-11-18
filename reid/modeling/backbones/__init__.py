
from .resnet import *
from .resnet_ibn_a import *
from .resnet_ibn_b import *
from .senet import *
from .densenet import *
from .mudeep import *
from .mlfn import *
from .hacnn import *
from .osnet import *
from .osnet_ain import *


__all__ = ['build_backbone']


_backbone_factory = {
    # name: [func, out_channels]
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
    'resnet152': [resnet152, 2048],
    'resnext50_32x4d': [resnext50_32x4d, 2048],
    'resnext101_32x8d': [resnext101_32x8d, 2048],
    'resnet50_ibn_a': [resnet50_ibn_a, 2048],
    'resnet101_ibn_a': [resnet101_ibn_a, 2048],
    'resnet50_ibn_b': [resnet50_ibn_b, 2048],
    'resnet101_ibn_b': [resnet101_ibn_b, 2048],
    'senet154': [senet154, 2048],
    'se_resnet50': [se_resnet50, 2048],
    'se_resnet101': [se_resnet101, 2048],
    'se_resnet152': [se_resnet152, 2048],
    'se_resnext50_32x4d': [se_resnext50_32x4d, 2048],
    'se_resnext101_32x4d': [se_resnext101_32x4d, 2048],
    'densenet121': [densenet121, 1024],
    'mudeep': [mudeep, 256],
    'mlfn': [mlfn, 1024],
    'hacnn': [hacnn, 1024],
    'osnet_x1_0': [osnet_x1_0, 512],
    'osnet_ibn_x1_0': [osnet_ibn_x1_0, 512],
    'osnet_ain_x1_0': [osnet_ain_x1_0, 512]
}


def build_backbone(params):
    if params.name not in _backbone_factory:
        raise NotImplementedError("unknow backbone name [{}]".format(params.name))

    func, out_channels = _backbone_factory[params.name]
    backbone = func(params.last_stride)
    if params.pretrained:
        backbone.load_param(params.pretrained)
        print("Load self provided pretrained backbone: {}".format(params.pretrained))
    return backbone, out_channels
