"""
Modified from vqdang code at https://github.com/vqdang/hover_net/blob/tensorflow-final/src/model/dcan.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import morphology, measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from mmcv.cnn import ConvModule

from tiseg.utils import resize
from ..builder import SEGMENTORS
from ..losses import BatchMultiClassDiceLoss
from .base import BaseSegmentor


def conv1x1(in_dims, out_dims, norm_cfg=None, act_cfg=None):
    return ConvModule(in_dims, out_dims, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)


def conv3x3(in_dims, out_dims, norm_cfg=None, act_cfg=None):
    return ConvModule(in_dims, out_dims, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)


def conv(in_dims, out_dims, kernel, norm_cfg=None, act_cfg=None):
    return ConvModule(in_dims, out_dims, kernel, 1, (kernel - 1) // 2, norm_cfg=norm_cfg, act_cfg=act_cfg)


def up_convs(in_dims, out_dims, up_nums, norm_cfg=None, act_cfg=None):
    conv_list = []
    for idx in range(up_nums):
        if idx == 0:
            conv_list.extend([nn.Upsample(scale_factor=2), conv3x3(in_dims, out_dims, norm_cfg, act_cfg)])
        else:
            conv_list.extend([nn.Upsample(scale_factor=2), conv3x3(out_dims, out_dims, norm_cfg, act_cfg)])

    return nn.Sequential(*conv_list)


class BottleNeck(nn.Module):

    def __init__(self, in_dims):
        super().__init__()
        self.res_conv = nn.Sequential(*[
            conv1x1(in_dims // 4, in_dims // 4, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
            conv3x3(in_dims // 4, in_dims // 4, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
            conv1x1(in_dims // 4, in_dims, norm_cfg=None, act_cfg=None),
        ])
        self.ide_conv = conv1x1(in_dims, in_dims, norm_cfg=dict(type='BN'), act_cfg=None)

        self.act = nn.ReLU()

    def forward(self, x):
        res = self.res_conv(x)
        ide = self.ide_conv(x)

        return self.act(res + ide)


@SEGMENTORS.register_module()
class DCAN(BaseSegmentor):
    """Implementation of `DCAN: deep contour-aware networks for accurate gland segmentation`.
    """

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(DCAN, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes

        self.stage1 = nn.Sequential(
            conv3x3(3, 64, None, dict(type='ReLU')),
            conv3x3(64, 64, None, dict(type='ReLU')),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.stage2 = nn.Sequential(
            conv3x3(64, 128, None, dict(type='ReLU')),
            conv3x3(128, 128, None, dict(type='ReLU')),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.stage3 = nn.Sequential(
            conv3x3(128, 256, None, dict(type='ReLU')),
            conv3x3(256, 256, None, dict(type='ReLU')),
            conv3x3(256, 256, None, dict(type='ReLU')),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.stage4 = nn.Sequential(
            conv3x3(256, 512, None, dict(type='ReLU')),
            conv3x3(512, 512, None, dict(type='ReLU')),
            conv3x3(512, 512, None, dict(type='ReLU')),
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.stage5 = nn.Sequential(
            conv3x3(512, 512, None, dict(type='ReLU')),
            conv3x3(512, 512, None, dict(type='ReLU')),
            conv3x3(512, 512, None, dict(type='ReLU')),
        )
        self.pool5 = nn.MaxPool2d(2, 2)

        self.stage6 = nn.Sequential(
            conv(512, 1024, 7, None, dict(type='ReLU')),
            nn.Dropout(p=0.5),
            conv1x1(1024, 1024, None, dict(type='ReLU')),
        )

        self.up_conv_4_cell = conv1x1(512, num_classes, None, None)
        self.up_conv_4_cont = conv1x1(512, 2, None, None)

        self.up_conv_5_cell = conv1x1(512, num_classes, None, None)
        self.up_conv_5_cont = conv1x1(512, 2, None, None)

        self.up_conv_6_cell = conv1x1(1024, num_classes, None, None)
        self.up_conv_6_cont = conv1x1(1024, 2, None, None)

    def calculate(self, img):
        B, _, H, W = img.shape
        x1 = self.stage1(img)
        p1 = self.pool1(x1)

        x2 = self.stage2(p1)
        p2 = self.pool2(x2)

        x3 = self.stage3(p2)
        p3 = self.pool3(x3)

        x4 = self.stage4(p3)
        p4 = self.pool4(x4)

        x5 = self.stage5(p4)
        p5 = self.pool5(x5)

        x6 = self.stage6(p5)

        out4 = F.interpolate(x4, (H, W), mode='bilinear', align_corners=False)
        out5 = F.interpolate(x5, (H, W), mode='bilinear', align_corners=False)
        out6 = F.interpolate(x6, (H, W), mode='bilinear', align_corners=False)

        cell_4 = self.up_conv_4_cell(out4)
        cont_4 = self.up_conv_4_cont(out4)

        cell_5 = self.up_conv_5_cell(out5)
        cont_5 = self.up_conv_5_cont(out5)

        cell_6 = self.up_conv_6_cell(out6)
        cont_6 = self.up_conv_6_cont(out6)

        cell_logit = cell_4 + cell_5 + cell_6
        cont_logit = cont_4 + cont_5 + cont_6

        return cell_logit, cont_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            cell_logit, cont_logit = self.calculate(data['img'])
            assert label is not None
            sem_gt = label['sem_gt']
            sem_gt_wb = label['sem_gt_w_bound']
            # NOTE: 0 ~ (num_classes - 1) are regular semantic classes.
            # num_classes is the id of edge class.
            cont_gt = sem_gt_wb == self.num_classes
            loss = dict()
            sem_gt = sem_gt.squeeze(1)
            cont_gt = cont_gt.squeeze(1)
            sem_loss = self._sem_loss(cell_logit, cont_logit, sem_gt, cont_gt)
            loss.update(sem_loss)
            # calculate training metric
            training_metric_dict = self._training_metric(cell_logit, sem_gt)
            loss.update(training_metric_dict)
            return loss
        else:
            assert metas is not None
            # NOTE: only support batch size = 1 now.
            cell_logit, cont_logit = self.inference(data['img'], metas[0], True)
            np.save("z_contour.npy", cont_logit.detach().cpu().numpy())
            cell_pred = cell_logit.argmax(dim=1)
            cont_pred = cont_logit.argmax(dim=1)
            cell_pred = cell_pred.cpu().numpy()[0]
            cont_pred = cont_pred.cpu().numpy()[0]
            sem_pred, inst_pred = self.postprocess(cell_pred, cont_pred)
            # unravel batch dim
            ret_list = []
            ret_list.append({'sem_pred': sem_pred, 'inst_pred': inst_pred})
            return ret_list

    def postprocess(self, cell_pred, cont_pred):
        """model free post-process for both instance-level & semantic-level."""
        # Use boundary prediction to split cells.
        cell_pred[cont_pred > 0] = 0
        sem_id_list = list(np.unique(cell_pred))
        inst_pred = np.zeros_like(cell_pred).astype(np.int32)
        sem_pred = np.zeros_like(cell_pred).astype(np.uint8)
        cur = 0
        for sem_id in sem_id_list:
            # 0 is background semantic class.
            if sem_id == 0:
                continue
            sem_id_mask = cell_pred == sem_id
            # fill instance holes
            sem_id_mask = binary_fill_holes(sem_id_mask)
            sem_id_mask = remove_small_objects(sem_id_mask, 5)
            inst_sem_mask = measure.label(sem_id_mask)
            inst_sem_mask = morphology.dilation(inst_sem_mask, selem=morphology.disk(self.test_cfg.get('radius', 3)))
            inst_sem_mask[inst_sem_mask > 0] += cur
            inst_pred[inst_sem_mask > 0] = 0
            inst_pred += inst_sem_mask
            cur += len(np.unique(inst_sem_mask))
            sem_pred[inst_sem_mask > 0] = sem_id

        return sem_pred, inst_pred

    def _sem_loss(self, cell_logit, cont_logit, sem_gt, cont_gt):
        """calculate mask branch loss."""
        sem_loss = {}
        sem_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
        cont_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=2)
        # Assign weight map for each pixel position
        # sem_loss *= weight_map
        cell_ce_loss = torch.mean(sem_ce_loss_calculator(cell_logit, sem_gt.long()))
        cell_dice_loss = sem_dice_loss_calculator(cell_logit, sem_gt.long())
        cont_ce_loss = torch.mean(sem_ce_loss_calculator(cont_logit, cont_gt.long()))
        cont_dice_loss = cont_dice_loss_calculator(cont_logit, cont_gt.long())
        # loss weight
        alpha = 5
        beta = 0.5
        sem_loss['cell_ce_loss'] = alpha * cell_ce_loss
        sem_loss['cont_ce_loss'] = alpha * cont_ce_loss
        sem_loss['cell_dice_loss'] = beta * cell_dice_loss
        sem_loss['cont_dice_loss'] = beta * cont_dice_loss

        return sem_loss

    def inference(self, img, meta, rescale):
        """Inference with split/whole style.
        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            meta (dict): Image info dict.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['split', 'whole']

        self.rotate_degrees = self.test_cfg.get('rotate_degrees', [0])
        self.flip_directions = self.test_cfg.get('flip_directions', ['none'])
        cell_logit_list = []
        cont_logit_list = []
        img_ = img
        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                img = self.tta_transform(img_, rotate_degree, flip_direction)

                # inference
                if self.test_cfg.mode == 'split':
                    cell_logit, cont_logit = self.split_inference(img, meta, rescale)
                else:
                    cell_logit, cont_logit = self.whole_inference(img, meta, rescale)

                cell_logit = self.reverse_tta_transform(cell_logit, rotate_degree, flip_direction)
                cont_logit = self.reverse_tta_transform(cont_logit, rotate_degree, flip_direction)

                cell_logit = F.softmax(cell_logit, dim=1)
                cont_logit = F.softmax(cont_logit, dim=1)

                cell_logit_list.append(cell_logit)
                cont_logit_list.append(cont_logit)

        cell_logit = sum(cell_logit_list) / len(cell_logit_list)
        cont_logit = sum(cont_logit_list) / len(cont_logit_list)

        if rescale:
            cell_logit = resize(cell_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            cont_logit = resize(cont_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)

        return cell_logit, cont_logit

    def split_inference(self, img, meta, rescale):
        """using half-and-half strategy to slide inference."""
        window_size = self.test_cfg.crop_size[0]
        overlap_size = self.test_cfg.overlap_size[0]

        B, C, H, W = img.shape

        # zero pad for border patches
        pad_h = 0
        if H - window_size > 0:
            pad_h = (window_size - overlap_size) - (H - window_size) % (window_size - overlap_size)
        else:
            pad_h = window_size - H

        if W - window_size > 0:
            pad_w = (window_size - overlap_size) - (W - window_size) % (window_size - overlap_size)
        else:
            pad_w = window_size - W

        H1, W1 = pad_h + H, pad_w + W
        img_canvas = torch.zeros((B, C, H1, W1), dtype=img.dtype, device=img.device)
        img_canvas[:, :, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W] = img

        cell_logit = torch.zeros((B, self.num_classes, H1, W1), dtype=img.dtype, device=img.device)
        cont_logit = torch.zeros((B, 2, H1, W1), dtype=img.dtype, device=img.device)
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (i + window_size - overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                img_patch = img_canvas[:, :, i:r_end, j:c_end]
                cell_patch, cont_patch = self.calculate(img_patch)

                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (j + window_size - overlap_size // 2 if j + window_size < W1 else W1)
                cell_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = cell_patch[:, :, ind1_s - i:ind1_e - i,
                                                                            ind2_s - j:ind2_e - j]
                cont_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = cont_patch[:, :, ind1_s - i:ind1_e - i,
                                                                            ind2_s - j:ind2_e - j]

        cell_logit = cell_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        cont_logit = cont_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]

        return cell_logit, cont_logit

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        cell_logit, cont_logit = self.calculate(img)

        return cell_logit, cont_logit
