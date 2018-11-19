# @author sli

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(scores, labels):
    _labels = torch.zeros_like(scores)
    _labels.scatter_(dim=-1, index=labels[..., None], value=1)
    _labels.requires_grad = False
    return _labels


def focal_loss_sigmoid(scores, scores_targets, n, masks_enable=None, alpha=0.25, gamma=2.0):
    """

    :param scores: Tensor n-dim
    :param scores_targets: Tensor[labels] (n-1)-dim
    :param n: int, norm scale
    :param masks_enable: None/Tensor, shape same as scores_targets [1-enable, 0-disable]
    :param alpha: float
    :param gamma: float
    :return:
        focal_loss
    """
    o = one_hot(scores, scores_targets)
    p = torch.sigmoid(scores)
    w = alpha * o * (1 - p) ** gamma + (1 - alpha) * (1 - o) * p ** gamma
    if masks_enable is not None:
        w = masks_enable[..., None] * w
    bce = F.binary_cross_entropy_with_logits(scores, o, reduction='none')
    return (w * bce).sum() / max(n, 1)


class FocalLossSigmoid(nn.Module):
    
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        :param alpha: float
        :param gamma: float
        
        """
        super(FocalLossSigmoid, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, scores, scores_targets, n, masks_enable=None):
        """
        :param scores: Tensor n-dim
        :param scores_targets: Tensor[labels] (n-1)-dim
        :param n: int, norm scale
        :param masks_enable: None/Tensor, shape same as scores_targets [1-enable, 0-disable]
        :return:
            focal_loss
        """
        return focal_loss_sigmoid(scores, scores_targets, n, masks_enable, self.alpha, self.gamma)
