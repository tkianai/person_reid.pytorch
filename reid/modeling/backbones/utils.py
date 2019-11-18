
import torch
import torch.utils.model_zoo as model_zoo


def init_pretrained_weights(model, url_or_path):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    if url_or_path.startswith('http'):
        pretrain_dict = model_zoo.load_url(url_or_path)
    else:
        pretrain_dict = torch.load(url_or_path)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items(
    ) if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
