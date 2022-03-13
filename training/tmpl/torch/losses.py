import torch


class Loss(torch.nn.Module):
    """
    Loss class
    """

    def __init__(
        self,
    ):
        super(Loss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)
