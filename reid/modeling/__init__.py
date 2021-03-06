

from .reid_net import ReIDNet


def build_model(cfg, num_classes):
    backbone_params = cfg.model.backbone
    midneck_params = cfg.model.midneck
    head_params = cfg.head
    loss_params = cfg.loss
    model = ReIDNet(
        num_classes,
        backbone_params,
        midneck_params,
        head_params,
        loss_params,
        feat_after_neck=cfg.test.feat_after_neck
    )
    return model
