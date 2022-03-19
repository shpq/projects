from utils import load_obj
import torch.nn as nn
import torch


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
            "drop_rate": cfg.project.model.dropout
        }
        self.back_model = load_obj(model_cfg.class_name)(
            model_cfg.name, **kwargs)
        if hasattr(self.back_model, "classifier"):
            self.fc = self.back_model.classifier
            self.back_model.classifier = nn.Dropout(
                p=float(cfg.project.model.dropout))
        elif hasattr(self.back_model, "head"):
            self.fc = self.back_model.head.fc
            self.back_model.head.fc = nn.Dropout(
                p=float(cfg.project.model.dropout))
        else:
            raise NotImplementedError

        self.linear_1_bias = nn.Parameter(
            torch.zeros(cfg.project.dataset.num_classes).float()
        )

    def forward(self, x):

        x = self.back_model(x)

        x = x.view(x.size(0), -1)

        logits = self.fc(x)

        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)

        return logits, probas
