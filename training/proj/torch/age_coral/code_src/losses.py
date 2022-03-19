import torch.nn.functional as F
import torch


class Loss(torch.nn.Module):
    """
    Loss class
    """

    def __init__(
        self,
    ):
        super(Loss, self).__init__()

    def forward(self, logits, levels):
        val = -torch.sum(
            (
                F.logsigmoid(logits) * levels
                + (F.logsigmoid(logits) - logits) * (1 - levels)
            ),
            dim=1,
        )
        return torch.mean(val)
