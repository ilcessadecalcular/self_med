import torch


def dice_func(output, target):
    smooth = 0.001
    logits = torch.sigmod(output)
    intersection = logits*target
    intersection_sum = torch.sum(intersection)
    logits_sum = logits.sum()
    target_sum = target.sum()
    f = 2*intersection_sum / (logits_sum + target_sum +smooth)

    return f
