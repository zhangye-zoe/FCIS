import torch
import torch.nn as nn
import torch.nn.functional as F


def _convert_to_one_hot(tensor, bins, on_value=1, off_value=0):
    """Convert NxHxW shape tensor to NxCxHxW one-hot tensor.

    Args:
        tensor (torch.Tensor): The tensor to convert.
        bins (int): The number of one-hot channels.
            (`bins` is usually `num_classes + 1`)
        on_value (int): The one-hot activation value. Default: 1
        off_value (int): The one-hot deactivation value. Default: 0
    """
    assert tensor.ndim == 3
    assert on_value != off_value
    tensor_one_hot = F.one_hot(tensor, bins)
    tensor_one_hot[tensor_one_hot == 1] = on_value
    tensor_one_hot[tensor_one_hot == 0] = off_value

    return tensor_one_hot


class TopologicalLoss(nn.Module):

    def __init__(self, use_regression=True, weight=False, num_angles=None, use_dice=False):
        super().__init__()
        self.use_regression = use_regression
        self.weight = weight
        self.num_angles = num_angles
        self.use_dice = use_dice

    def forward(self, pred, target, pred_contour, target_contour):
        if self.use_regression:
            dir_mse_loss_calculator = nn.MSELoss(reduction='none')
            dir_mse_loss = dir_mse_loss_calculator(pred, target)
            all_contour = (pred_contour + target_contour) > 0
            loss = torch.sum(dir_mse_loss * all_contour) / torch.sum(all_contour)
        else:
            all_contour = (pred_contour + target_contour) > 0
            loss = 0
            if self.use_dice:
                assert target.ndim == 3
                # one-hot encoding for target
                target = target * all_contour
                target_one_hot = _convert_to_one_hot(target, self.num_angles+1).permute(0, 3, 1, 2).contiguous()
                smooth = 1e-4
                 # softmax for logit
                logit = F.softmax(pred, dim=1)
                N, C, _, _ = target_one_hot.shape
                
                for i in range(1, C):
                    logit_per_class = logit[:, i] * all_contour
                    target_per_class = target_one_hot[:, i]

                    intersection = logit_per_class * target_per_class
                    # calculate per class dice loss
                    if self.weight:
                        pred_dir = torch.argmax(pred, dim=1)
                        diff = torch.abs(pred_dir - target)
                        weight = diff.min(self.num_angles - diff) + 1
                        background = (pred_dir == 0) + (target == 0)
                        weight[background > 0] = 2
                        dice_loss_per_class = (2 * (intersection * weight).sum((0, -2, -1))  + smooth) / (
                            (logit_per_class * weight).sum((0, -2, -1))  + (target_per_class * weight).sum((0, -2, -1)) + smooth)
                    else:
                        dice_loss_per_class = (2 * intersection.sum((0, -2, -1)) + smooth) / (
                            logit_per_class.sum((0, -2, -1)) + target_per_class.sum((0, -2, -1)) + smooth)

                    dice_loss_per_class = 1 - dice_loss_per_class
                    loss += dice_loss_per_class
            
            dir_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
            dir_ce_loss = dir_ce_loss_calculator(pred, target)
            if self.weight:
                pred_dir = torch.argmax(pred, dim=1)
                diff = torch.abs(pred_dir - target)
                weight = diff.min(self.num_angles - diff) + 1
                background = (pred_dir == 0) + (target == 0)
                weight[background > 0] = 2
                dir_ce_loss = dir_ce_loss * weight
            loss += torch.sum(dir_ce_loss * all_contour) / torch.sum(all_contour)
        return loss
