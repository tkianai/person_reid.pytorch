

from .bnneck import BnNeck


__all__ = ['build_midneck']

_midneck_factory = {
    'bnneck': BnNeck,

}


def build_midneck(params, in_channels):

    midneck = _midneck_factory.get(params.name, None)
    if midneck is not None:
        midneck = midneck(in_channels)

    return midneck
