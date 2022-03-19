import torch


def calculate_log1p(probas, targets):
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    predicted_log1p = torch.log1p(predicted_labels.float())
    targets_log1p = torch.log1p(targets.float())
    return torch.abs(predicted_log1p - targets_log1p).mean()
