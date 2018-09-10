import torch


def log_sum_exp(x):
    max_x, _ = torch.max(x, dim=0)
    return torch.log(torch.sum(torch.exp(x - max_x), dim=0)) + max_x
