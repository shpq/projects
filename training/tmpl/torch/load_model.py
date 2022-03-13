import torch
from utils import load_obj


class Model(torch.nn.Module):
    """
    Model class
    """

    def __init__(self, cfg):
        super(Model, self).__init__()
        model_cfg = cfg.model
        kwargs = {
            **model_cfg.params,
            "num_classes": cfg.project.dataset.num_classes,
        }
        self.model = load_obj(model_cfg.class_name)(model_cfg.name, **kwargs)

    def forward(self, x):
        return self.model(x)
