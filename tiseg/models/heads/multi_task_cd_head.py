import torch.nn as nn
from mmcv.cnn import build_activation_layer

from .unet_head import UNetHead, conv1x1, conv3x3


class RU(nn.Module):
    """Residual Unit.

    Residual Unit comprises of:
    (Conv3x3 + BN + ReLU + Conv3x3 + BN) + Identity + ReLU
    ( . ) stands for residual inside block

    Args:
        in_dims (int): The input channels of Residual Unit.
        out_dims (int): The output channels of Residual Unit.
        norm_cfg (dict): The normalize layer config. Default: dict(type='BN')
        act_cfg (dict): The activation layer config. Default: dict(type='ReLU')
    """

    def __init__(self, in_dims, out_dims, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super().__init__()

        # NOTE: inplace wise relu can largely save gpu memory cost.
        real_act_cfg = dict()
        real_act_cfg['inplace'] = True
        real_act_cfg.update(act_cfg)

        self.act_layer = build_activation_layer(real_act_cfg)
        self.residual_ops = nn.Sequential(
            conv3x3(in_dims, out_dims, norm_cfg), self.act_layer, conv3x3(out_dims, out_dims, norm_cfg))
        self.identity_ops = nn.Sequential(conv1x1(in_dims, out_dims))

    def forward(self, x):
        ide_value = self.identity_ops(x)
        res_value = self.residual_ops(x)
        out = ide_value + res_value
        return self.act_layer(out)


class AU(nn.Module):
    """Attention Unit.

    This module use (conv1x1 + sigmoid) to generate 0-1 (float) attention mask.

    Args:
        in_dims (int): The input channels of Attention Unit.
        num_masks (int): The number of masks to generate. Default: 1
    """

    def __init__(self, in_dims, num_masks=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dims, num_masks, kernel_size=1, bias=False), nn.Sigmoid())

    def forward(self, signal, gate):
        """Using gate to generate attention map and assign the attention map to
        signal."""
        attn_map = self.conv(gate)
        return signal * (1 + attn_map)


class Identity(nn.Module):

    def forward(self, *args):
        return args[0]


class DGM(nn.Module):
    """Direction-Guided Refinement Module (DGM)

    This module will accept prediction of regular segmentation output. This
    module has three branches:
    (1) Mask Branch;
    (2) Direction Map Branch;
    (3) Point Map Branch;

    When training phrase, these three branches provide mask, direction, point
    supervision, respectively. When testing phrase, direction map and point map
    provide refinement operations.

    Args:
        in_dims (int): The input channels of DGM.
        feed_dims (int): The feedforward channels of DGM.
        num_classes (int): The number of mask semantic classes.
        num_angles (int): The number of angle types. Default: 8
        norm_cfg (dict): The normalize layer config. Default: dict(type='BN')
        act_cfg (dict): The activation layer config. Default: dict(type='ReLU')
    """

    def __init__(self,
                 in_dims,
                 feed_dims,
                 num_classes,
                 num_angles=8,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 noau=False,
                 use_regression=False,
                 parallel=False):
        super().__init__()
        self.in_dims = in_dims
        self.feed_dims = feed_dims
        self.num_classes = num_classes
        self.num_angles = num_angles
        self.parallel = parallel

        if self.parallel:
            self.tc_mask_feats = RU(self.feed_dims, self.feed_dims, norm_cfg, act_cfg)
            self.mask_feats = RU(self.in_dims, self.feed_dims, norm_cfg, act_cfg)
            self.dir_feats = RU(self.in_dims, self.feed_dims, norm_cfg, act_cfg)
            self.point_feats = RU(self.in_dims, self.feed_dims, norm_cfg, act_cfg)
        else:
            self.tc_mask_feats = RU(self.in_dims, self.feed_dims, norm_cfg, act_cfg)
            self.mask_feats = RU(self.in_dims, self.feed_dims, norm_cfg, act_cfg)
            self.dir_feats = RU(self.feed_dims, self.feed_dims, norm_cfg, act_cfg)
            self.point_feats = RU(self.feed_dims, self.feed_dims, norm_cfg, act_cfg)

        # Cross Branch Attention
        if noau:
            self.point_to_dir_attn = Identity()
            self.dir_to_tc_mask_attn = Identity()
        else:
            self.point_to_dir_attn = AU(1)
            self.dir_to_tc_mask_attn = AU(self.num_angles + 1)

        # Prediction Operations
        self.point_conv = nn.Conv2d(self.feed_dims, 1, kernel_size=1)
        if use_regression:
            self.dir_conv = nn.Conv2d(self.feed_dims, 1, kernel_size=1)
        else:
            self.dir_conv = nn.Conv2d(self.feed_dims, self.num_angles + 1, kernel_size=1)
        self.tc_mask_conv = nn.Conv2d(self.feed_dims, 3, kernel_size=1)
        self.mask_conv = nn.Conv2d(self.feed_dims, self.num_classes, kernel_size=1)

    def forward(self, x):

        if self.parallel:
            dir_feature = self.dir_feats(x)
            mask_feature = self.mask_feats(x)
            point_feature = self.point_feats(x)
            tc_mask_feature = self.tc_mask_feats(mask_feature)
        else:
            tc_mask_feature = self.tc_mask_feats(x)
            dir_feature = self.dir_feats(tc_mask_feature)
            point_feature = self.point_feats(dir_feature)
            mask_feature = self.mask_feats(x)

        # point branch
        point_logit = self.point_conv(point_feature)

        # direction branch
        dir_feature_with_point_logit = self.point_to_dir_attn(dir_feature, point_logit)
        dir_logit = self.dir_conv(dir_feature_with_point_logit)

        # mask branch
        tc_mask_feature_with_dir_logit = self.dir_to_tc_mask_attn(tc_mask_feature, dir_logit)
        tc_mask_logit = self.tc_mask_conv(tc_mask_feature_with_dir_logit)

        # semantic mask branch
        mask_logit = self.mask_conv(mask_feature)

        return tc_mask_logit, mask_logit, dir_logit, point_logit


class MultiTaskCDHead(UNetHead):

    def __init__(self,
                 num_classes,
                 num_angles=8,
                 dgm_dims=64,
                 noau=False,
                 use_regression=False,
                 parallel=False,
                 **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.num_classes = num_classes
        self.num_angles = num_angles

        self.postprocess = DGM(
            self.stage_dims[0],
            dgm_dims,
            num_classes=self.num_classes,
            num_angles=self.num_angles,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            noau=noau,
            use_regression=use_regression,
            parallel=parallel)
