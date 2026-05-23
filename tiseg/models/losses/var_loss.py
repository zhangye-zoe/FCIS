"""
Modified from huiqu code at https://github.com/huiqu18/FullNet-varCE/blob/master/loss.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VarianceLoss(nn.Module):
    """ The instances in target should be labeled
    """

    def __init__(self):
        super(VarianceLoss, self).__init__()

    def forward(self, logit, inst_gt):
        logit = F.softmax(logit, dim=1)
        B = logit.shape[0]

        loss = 0
        for k in range(B):
            inst_ids = inst_gt[k].unique()
            inst_ids = inst_ids[inst_ids != 0]

            sum_var = 0
            for inst_id in inst_ids:
                inst_map = inst_gt[k] == inst_id
                inst_logit = logit[k][:, inst_map]
                # The number of pixels included in instance must be larger than 1.
                if torch.sum(inst_map) > 1:
                    sum_var += inst_logit.var(dim=1).sum()

            loss += sum_var / (len(inst_ids) + 1e-8)
        loss /= B
        return loss
