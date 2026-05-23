import torch
import torch.nn as nn

from .utils import overlapping_caculation


def iou(pred, target, num_classes, thresh=None, reduce_zero_label=True):
    """Intersection and Union calculation."""
    return overlapping_caculation(pred, target, 'IoU', num_classes, thresh,
                                  reduce_zero_label)


def miou(pred, target, num_classes, thresh=None, reduce_zero_label=True):
    """Intersection and Union calculation."""
    return torch.mean(
        overlapping_caculation(pred, target, 'IoU', num_classes, thresh,
                               reduce_zero_label))


def tiou(pred, target, num_classes, thresh=None, reduce_zero_label=True):
    """total Dice coefficient calculation."""
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    pred_value, pred_label = pred.topk(1, dim=1)
    label = target
    # transpose to shape (maxk, N, ...)
    pred_label = pred_label.transpose(0, 1)[0]
    pred_value = pred_value.transpose(0, 1)[0]
    # fuse all foreground class
    intersect_mask = pred_label == label
    if thresh is not None:
        intersect_mask = intersect_mask & pred_value > thresh
    intersect = pred_label[intersect_mask]
    area_intersect = torch.histc(
        intersect.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_pred = torch.histc(
        pred_label.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=num_classes, min=0, max=num_classes - 1)

    # zero label usually stands for background, so we remove it.
    if reduce_zero_label:
        area_intersect = area_intersect[1:]
        area_pred = area_pred[1:]
        area_label = area_label[1:]

    area_union = area_pred + area_label - area_intersect

    res = 100 * area_intersect.sum() / (area_union.sum())

    # TODO: Make this to arg
    res[torch.isnan(res)] = 0

    return res


class IoU(nn.Module):
    """Intersection and Union calculation module."""

    def __init__(self, thresh=None):
        """Module to calculate the iou.

        Args:
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.thresh = thresh

    def forward(self, pred, target):
        """Forward function to calculate iou.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return iou(pred, target, self.thresh)
