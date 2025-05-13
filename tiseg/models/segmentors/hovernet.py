"""
Modified from vqdang code at https://github.com/vqdang/hover_net/blob/conic/models/hovernet/net_desc.py
"""

import math
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from tiseg.utils import resize
from .base import BaseSegmentor
from ..builder import SEGMENTORS
from ..losses import GradientMSELoss, BatchMultiClassDiceLoss, mdice, tdice


class ResNetExt(ResNet):

    def _forward_impl(self, x, freeze):
        # See note [TorchScript super()]
        if self.training:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            with torch.set_grad_enabled(not freeze):
                x1 = x = self.layer1(x)
                x2 = x = self.layer2(x)
                x3 = x = self.layer3(x)
                x4 = x = self.layer4(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x1 = x = self.layer1(x)
            x2 = x = self.layer2(x)
            x3 = x = self.layer3(x)
            x4 = x = self.layer4(x)
        return x1, x2, x3, x4

    def forward(self, x: torch.Tensor, freeze: bool = False) -> torch.Tensor:
        return self._forward_impl(x, freeze)

    @staticmethod
    def resnet50(num_input_channels, pretrained=None):
        model = ResNetExt(ResNetBottleneck, [3, 4, 6, 3])
        model.conv1 = nn.Conv2d(num_input_channels, 64, 7, stride=1, padding=3)
        if pretrained is not None:
            pretrained = torch.load(pretrained)
            (missing_keys, unexpected_keys) = model.load_state_dict(pretrained, strict=False)
        return model


class DenseBlock(nn.Module):
    """Dense Block as defined in:
    Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger.
    "Densely connected convolutional networks." In Proceedings of the IEEE conference
    on computer vision and pattern recognition, pp. 4700-4708. 2017.
    Only performs `valid` convolution.
    """

    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, split=1):
        super().__init__()
        assert len(unit_ksize) == len(unit_ch), "Unbalance Unit Info"

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch

        # ! For inference only so init values for batchnorm may not match tensorflow
        unit_in_ch = in_ch
        pad_vals = [v // 2 for v in unit_ksize]
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            self.units.append(
                nn.Sequential(
                    nn.BatchNorm2d(unit_in_ch, eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        unit_in_ch,
                        unit_ch[0],
                        unit_ksize[0],
                        stride=1,
                        padding=pad_vals[0],
                        bias=False,
                    ),
                    nn.BatchNorm2d(unit_ch[0], eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        unit_ch[0],
                        unit_ch[1],
                        unit_ksize[1],
                        stride=1,
                        padding=pad_vals[1],
                        bias=False,
                        groups=split,
                    ),
                ))
            unit_in_ch += unit_ch[1]

        self.blk_bna = nn.Sequential(nn.BatchNorm2d(unit_in_ch, eps=1e-5), nn.ReLU(inplace=True))

    def out_ch(self):
        return self.in_ch + self.nr_unit * self.unit_ch[-1]

    def forward(self, prev_feat):
        for idx in range(self.nr_unit):
            new_feat = self.units[idx](prev_feat)
            prev_feat = torch.cat([prev_feat, new_feat], dim=1)
        prev_feat = self.blk_bna(prev_feat)

        return prev_feat


class UpSample2x(nn.Module):
    """A layer to scale input by a factor of 2.
    This layer uses Kronecker product underneath rather than the default
    pytorch interpolation.
    """

    def __init__(self):
        super().__init__()
        # correct way to create constant within module
        self.register_buffer("unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32")))
        self.unpool_mat.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        """Logic for using layers defined in init.
        Args:
            x (torch.Tensor): Input images, the tensor is in the shape of NCHW.
        Returns:
            ret (torch.Tensor): Input images upsampled by a factor of 2
                via nearest neighbour interpolation. The tensor is the shape
                as NCHW.
        """
        input_shape = list(x.shape)
        # un-squeeze is the same as expand_dims
        # permute is the same as transpose
        # view is the same as reshape
        x = x.unsqueeze(-1)  # bchwx1
        mat = self.unpool_mat.unsqueeze(0)  # 1xshxsw
        ret = torch.tensordot(x, mat, dims=1)  # bxcxhxwxshxsw
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))
        return ret


@SEGMENTORS.register_module()
class HoverNet(BaseSegmentor):
    """Initialise HoVer-Net."""

    def __init__(self, num_classes, train_cfg, test_cfg):
        super().__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes

        self.backbone = ResNetExt.resnet50(3, pretrained=None)
        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        ksize = 3
        self.decoder = nn.ModuleDict(
            OrderedDict([
                ("tp", self.create_decoder_branch(ksize=ksize, out_ch=num_classes)),
                ("np", self.create_decoder_branch(ksize=ksize, out_ch=2)),
                ("hv", self.create_decoder_branch(ksize=ksize, out_ch=2)),
            ]))

        self.upsample2x = UpSample2x()

    def create_decoder_branch(self, out_ch=2, ksize=5):
        pad = ksize // 2
        module_list = [
            nn.Conv2d(1024, 256, ksize, stride=1, padding=pad, bias=False),
            DenseBlock(256, [1, ksize], [128, 32], 8, split=4),
            nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
        ]
        u3 = nn.Sequential(*module_list)

        module_list = [
            nn.Conv2d(512, 128, ksize, stride=1, padding=pad, bias=False),
            DenseBlock(128, [1, ksize], [128, 32], 4, split=4),
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
        ]
        u2 = nn.Sequential(*module_list)

        module_list = [
            nn.Conv2d(256, 64, ksize, stride=1, padding=pad, bias=False),
        ]
        u1 = nn.Sequential(*module_list)

        module_list = [
            nn.BatchNorm2d(64, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),
        ]
        u0 = nn.Sequential(*module_list)

        decoder = nn.Sequential(OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)]))
        return decoder

    def decoder_forward(self, x, decoder):
        u3 = self.upsample2x(x[-1]) + x[-2]
        u3 = decoder[0](u3)

        u2 = self.upsample2x(u3) + x[-3]
        u2 = decoder[1](u2)

        u1 = self.upsample2x(u2) + x[-4]
        u1 = decoder[2](u1)

        u0 = decoder[3](u1)

        return u0

    def calculate(self, x):
        d0, d1, d2, d3 = self.backbone(x)
        d3 = self.conv_bot(d3)
        d = [d0, d1, d2, d3]

        sem_logit = self.decoder_forward(d, self.decoder['tp'])
        hv_logit = self.decoder_forward(d, self.decoder['hv'])
        fore_logit = self.decoder_forward(d, self.decoder['np'])

        return sem_logit, hv_logit, fore_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            sem_logit, hv_logit, fore_logit = self.calculate(data['img'])

            assert label is not None
            sem_gt = label['sem_gt']
            hv_gt = label['hv_gt']
            fore_gt = sem_gt.clone()
            fore_gt = (sem_gt > 0).long()

            loss = dict()

            sem_gt = sem_gt.squeeze(1)
            fore_gt = fore_gt.squeeze(1)

            # TODO: Conside to remove some edge loss value.
            # mask branch loss calculation
            sem_loss = self._sem_loss(sem_logit, sem_gt)
            loss.update(sem_loss)
            # direction branch loss calculation
            hv_loss = self._hv_loss(hv_logit, hv_gt, fore_gt)

            print("hv logit", hv_logit.shape)
            print("hv gt", hv_gt.shape)
            print("fore gt", fore_gt.shape)
            loss.update(hv_loss)
            # point branch loss calculation
            fore_loss = self._fore_loss(fore_logit, fore_gt)
            loss.update(fore_loss)

            # calculate training metric
            training_metric_dict = self._training_metric(sem_logit, fore_logit, sem_gt, fore_gt)
            loss.update(training_metric_dict)
            return loss
        else:
            assert metas is not None
            # NOTE: only support batch size = 1 now.
            sem_logit, hv_logit, fore_logit = self.inference(data['img'], metas[0], True)
            sem_pred = sem_logit.argmax(dim=1)
            # Extract inside class
            sem_pred = sem_pred.cpu().numpy()[0]
            # NHW -> HWN
            hv_pred = hv_logit.permute(0, 2, 3, 1).cpu().numpy()[0]
            fore_logit = fore_logit.cpu().numpy()[0][1]
            # unravel batch dim
            inst_pred = self.hover_post_proc(fore_logit, hv_pred, scale_factor=self.test_cfg.get('scale_factor', 1))
            ret_list = []
            ret_list.append({'sem_pred': sem_pred, 'inst_pred': inst_pred})
            return ret_list

    def hover_post_proc(self, fore_map, hv_map, fx=1, scale_factor=1):
        raw_h, raw_w = hv_map.shape[:2]

        fore_map = cv2.resize(fore_map, (0, 0), fx=scale_factor, fy=scale_factor)
        hv_map = cv2.resize(hv_map, (0, 0), fx=scale_factor, fy=scale_factor)

        blb_raw = fore_map
        h_dir_raw = hv_map[:, :, 0]
        v_dir_raw = hv_map[:, :, 1]

        # processing
        blb = np.array(blb_raw >= 0.5, dtype=np.int32)

        blb = measurements.label(blb)[0]
        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1  # background is 0 already

        h_dir = cv2.normalize(
            h_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        ksize = int((20 * fx) + 1)
        obj_size = math.ceil(10 * (fx**2))
        # Get resolution specific filters etc.

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        sobelh = 1 - (
            cv2.normalize(
                sobelh,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            ))
        sobelv = 1 - (
            cv2.normalize(
                sobelv,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            ))

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        dist = (1.0 - overall) * blb
        # * nuclei values form mountains so inverse to get basins
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        overall = np.array(overall >= 0.4, dtype=np.int32)

        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=obj_size)

        proced_pred = watershed(dist, markers=marker, mask=blb)

        proced_pred = cv2.resize(proced_pred, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)

        return proced_pred

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
        sem_logit_list = []
        hv_logit_list = []
        fore_logit_list = []
        img_ = img
        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                img = self.tta_transform(img_, rotate_degree, flip_direction)

                # inference
                if self.test_cfg.mode == 'split':
                    sem_logit, hv_logit, fore_logit = self.split_inference(img, meta, rescale)
                else:
                    sem_logit, hv_logit, fore_logit = self.whole_inference(img, meta, rescale)

                sem_logit = self.reverse_tta_transform(sem_logit, rotate_degree, flip_direction)
                hv_logit = self.reverse_tta_transform(hv_logit, rotate_degree, flip_direction)
                fore_logit = self.reverse_tta_transform(fore_logit, rotate_degree, flip_direction)

                sem_logit = F.softmax(sem_logit, dim=1)
                fore_logit = F.softmax(fore_logit, dim=1)

                sem_logit_list.append(sem_logit)
                hv_logit_list.append(hv_logit)
                fore_logit_list.append(fore_logit)

        sem_logit = sum(sem_logit_list) / len(sem_logit_list)
        hv_logit = hv_logit_list[0]
        fore_logit = sum(fore_logit_list) / len(fore_logit_list)

        if rescale:
            sem_logit = resize(sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            hv_logit = resize(hv_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            fore_logit = resize(fore_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)

        return sem_logit, hv_logit, fore_logit

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

        sem_logit = torch.zeros((B, self.num_classes, H1, W1), dtype=img.dtype, device=img.device)
        hv_logit = torch.zeros((B, 2, H1, W1), dtype=img.dtype, device=img.device)
        fore_logit = torch.zeros((B, 2, H1, W1), dtype=img.dtype, device=img.device)
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (i + window_size - overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                img_patch = img_canvas[:, :, i:r_end, j:c_end]
                sem_patch, hv_patch, fore_patch = self.calculate(img_patch)

                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (j + window_size - overlap_size // 2 if j + window_size < W1 else W1)
                sem_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = sem_patch[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]
                hv_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = hv_patch[:, :, ind1_s - i:ind1_e - i,
                                                                        ind2_s - j:ind2_e - j]
                fore_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = fore_patch[:, :, ind1_s - i:ind1_e - i,
                                                                            ind2_s - j:ind2_e - j]

        sem_logit = sem_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        hv_logit = hv_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        fore_logit = fore_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]

        return sem_logit, hv_logit, fore_logit

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        sem_logit, hv_logit, fore_logit = self.calculate(img)

        return sem_logit, hv_logit, fore_logit

    def _sem_loss(self, sem_logit, sem_gt, weight_map=None):
        sem_loss = {}
        sem_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
        # Assign weight map for each pixel position
        sem_ce_loss = sem_ce_loss_calculator(sem_logit, sem_gt)
        if weight_map is not None:
            sem_ce_loss *= weight_map[:, 0]
        sem_ce_loss = torch.mean(sem_ce_loss)
        sem_dice_loss = sem_dice_loss_calculator(sem_logit, sem_gt)
        # loss weight
        alpha = 5
        beta = 0.5
        sem_loss['sem_ce_loss'] = alpha * sem_ce_loss
        sem_loss['sem_dice_loss'] = beta * sem_dice_loss

        return sem_loss

    def _hv_loss(self, hv_logit, hv_gt, fore_gt):
        hv_loss = {}
        hv_mse_loss_calculator = nn.MSELoss()
        hv_msge_loss_calculator = GradientMSELoss()
        hv_mse_loss = hv_mse_loss_calculator(hv_logit, hv_gt)
        hv_msge_loss = hv_msge_loss_calculator(hv_logit, hv_gt, fore_gt)
        # loss weight
        alpha = 1
        beta = 1
        hv_loss['hv_mse_loss'] = alpha * hv_mse_loss
        hv_loss['hv_msge_loss'] = beta * hv_msge_loss

        return hv_loss

    def _fore_loss(self, fore_logit, fore_gt):
        fore_loss = {}
        fore_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        fore_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=2)
        fore_ce_loss = fore_ce_loss_calculator(fore_logit, fore_gt)
        fore_ce_loss = torch.mean(fore_ce_loss)
        fore_dice_loss = fore_dice_loss_calculator(fore_logit, fore_gt)
        # loss weight
        alpha = 1
        beta = 1
        fore_loss['fore_ce_loss'] = alpha * fore_ce_loss
        fore_loss['fore_dice_loss'] = beta * fore_dice_loss

        return fore_loss

    def _training_metric(self, sem_logit, fore_logit, sem_gt, fore_gt):
        wrap_dict = {}
        # detach these training variable to avoid gradient noise.
        clean_sem_logit = sem_logit.clone().detach()
        clean_sem_gt = sem_gt.clone().detach()
        clean_fore_logit = fore_logit.clone().detach()
        clean_fore_gt = fore_gt.clone().detach()

        wrap_dict['sem_mdice'] = mdice(clean_sem_logit, clean_sem_gt, self.num_classes)
        wrap_dict['fore_mdice'] = mdice(clean_fore_logit, clean_fore_gt, 2)

        wrap_dict['sem_tdice'] = tdice(clean_sem_logit, clean_sem_gt, self.num_classes)
        wrap_dict['fore_tdice'] = tdice(clean_fore_logit, clean_fore_gt, 2)

        # NOTE: training aji calculation metric calculate (This will be deprecated.)
        # sem_pred = torch.argmax(sem_logit, dim=1).cpu().numpy().astype(np.uint8)
        # sem_pred[sem_pred == (self.num_classes - 1)] = 0
        # sem_target = sem_gt.cpu().numpy().astype(np.uint8)
        # sem_target[sem_target == (self.num_classes - 1)] = 0

        # N = sem_pred.shape[0]
        # wrap_dict['aji'] = 0.
        # for i in range(N):
        #     aji_single_image = aggregated_jaccard_index(sem_pred[i], sem_target[i])
        #     wrap_dict['aji'] += 100.0 * torch.tensor(aji_single_image)
        # # distributed environment requires cuda tensor
        # wrap_dict['aji'] /= N
        # wrap_dict['aji'] = wrap_dict['aji'].cuda()

        return wrap_dict
