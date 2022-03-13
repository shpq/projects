from proj.torch.magface.code_src.losses import MagLinear
from utils import load_obj
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, cfg):
        """
        Model class for Magface task
        """
        super(Model, self).__init__()
        self.features = Backbone(cfg)
        self.fc = MagLinear(
            cfg.project.model.embedding_size,
            cfg.project.training.last_fc_size,
            scale=cfg.project.training.loss.scale,
            easy_margin=cfg.project.training.easy_margin,
        )

        self.l_margin = cfg.project.training.loss.l_margin
        self.u_margin = cfg.project.training.loss.u_margin
        self.l_a = cfg.project.training.loss.l_a
        self.u_a = cfg.project.training.loss.u_a

    def _margin(self, x):
        """
        Generating adaptive margin
        """
        margin = (self.u_margin - self.l_margin) / (self.u_a - self.l_a) * (
            x - self.l_a
        ) + self.l_margin
        return margin

    def forward(self, x, target):
        x = self.features(x)
        logits, x_norm = self.fc(x, self._margin, self.l_a, self.u_a)
        return logits, x_norm


class Backbone(torch.nn.Module):
    """
    Backbone class for timm models
    """

    def __init__(self, cfg):
        super(Backbone, self).__init__()
        img_size = cfg.project.training.size
        embedding_size = cfg.project.model.embedding_size
        model_cfg = cfg.model
        kwargs = {**model_cfg.params}
        model = load_obj(model_cfg.class_name)(model_cfg.name, **kwargs)
        # delete different kind of heads
        if hasattr(model, "classifier"):
            model.classifier = torch.nn.Identity()
            if hasattr(model, "global_pool"):
                model.global_pool = torch.nn.Identity()
            if hasattr(model, "act2"):
                model.global_pool = torch.nn.Identity()
        elif hasattr(model, "head"):
            if hasattr(model.head, "global_pool"):
                model.head.global_pool = torch.nn.Identity()
            if hasattr(model.head, "fc"):
                model.head.fc = torch.nn.Identity()
        self.model = model
        self.model.eval()

        # calculate output size to create model head properly
        size_output = model(torch.rand(1, 3, *img_size)).size()
        self.conv = nn.Conv2d(
            size_output[1],
            cfg.project.model.out_conv,
            kernel_size=3,
            bias=False,
        )
        size = self.conv(model(torch.rand(1, 3, *img_size))).size()
        in_features = torch.prod(torch.tensor(size[1:]))
        self.bn2 = nn.BatchNorm2d(size[1], eps=2e-05, momentum=0.9)
        self.dropout = nn.Dropout2d(p=cfg.project.model.dropout, inplace=True)
        self.fc = nn.Linear(in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.model.train()

    def forward(self, x):
        x = self.model(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x
