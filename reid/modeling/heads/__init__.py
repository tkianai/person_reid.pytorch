


from .softmax import SoftmaxHead, AmSoftmaxHead
from .arc import ArcHead
from .cos import CosHead
from .sphere import SphereHead


__all__ = ['build_head']

_head_factory = {
    'softmax': SoftmaxHead,
    'am_softmax': AmSoftmaxHead,
    'arc': ArcHead,
    'cos': CosHead,
    'sphere': SphereHead,
}


def build_head(params, in_features, out_features, **kwargs):

    head = _head_factory.get(params.name, None)
    if head is None:
        print("Using default head: softmax")
        head = SoftmaxHead

    head = head(in_features, out_features, **kwargs)
    return head
