"""metric"""
import torch
import numpy as np


def calc_acc(output: torch.Tensor, target: torch.Tensor,
             threshold: float) -> float:
    """Calculate accuraccy for a sequence

    Args:
        output (torch.Tensor): output
        target (torch.Tensor): target
        threshold (float): threshold
    """
    pred = output.ge(threshold)

    return pred.eq(target).sum().item()/(target.numel())


def confusion_matrix(output: torch.Tensor, target: torch.Tensor,
                     threshold: float):
    """calculate TP, FP, TP, TN of a sequence

    Args:
        output (torch.Tensor): _description_
        target (torch.Tensor): _description_
        threshold (float): _description_
    """
    pred = output.ge(threshold).type_as(target)
    TP = ((pred == 1) & (target == 1)).float().sum().item()
    TN = ((pred == 0) & (target == 0)).float().sum().item()
    FP = ((pred == 1) & (target == 0)).float().sum().item()
    FN = ((pred == 0) & (target == 1)).float().sum().item()
    return TP, TN, FP, FN


def f1_score(output: torch.Tensor, target: torch.Tensor, threshold: float,
             eps:  float = 1e-5):
    """calculate f1 score

    Args:
        output (torch.Tensor): output
        target (torch.Tensor): target
        threshold (float): threshold
    """
    pred = output.ge(threshold).type_as(target)
    TP = (pred*target).float().sum().item()
    TP_FP = pred.sum().item() + eps
    TP_FN = target.sum().item() + eps
    # case with no positive class
    if TP_FN == eps:
        return -1000
    precision = TP/TP_FP
    recall = TP/TP_FN
    if precision == 0:
        return 0
    return 2./(1./precision + 1./recall)


def metric_thresholds(output: torch.Tensor, target: torch.Tensor,
                      thresholds: list):
    """caculate best f1 score and acc for a list of thresholds

    Args:
        output (torch.Tensor): output
        target (torch.Tensor): target
        thresholds (list): list of threshold
    """
    acc = []
    f1 = []
    for threshold in thresholds:
        acc.append(calc_acc(output, target, threshold))
        f1.append(f1_score(output, target, threshold))
    f1 = np.array(f1)
    return max(acc),  max(f1), thresholds[np.argmax(f1)]
