import torch.nn as nn

from mmcv.cnn import build_activation_layer
from .unet_head import conv3x3, conv1x1, UNetLayer


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


class MultiTaskBranches(nn.Module):

    def __init__(self, in_dims, feed_dims, num_classes, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_dims = in_dims
        self.feed_dims = feed_dims
        self.norm_cfg = norm_cfg

        # NOTE: inplace wise relu can largely save gpu memory cost.
        real_act_cfg = dict()
        real_act_cfg['inplace'] = True
        real_act_cfg.update(act_cfg)
        self.act_cfg = real_act_cfg

        self.mask_feats = RU(self.in_dims, self.feed_dims, norm_cfg, act_cfg)
        self.aux_mask_feats = RU(self.feed_dims, self.feed_dims, norm_cfg, act_cfg)

        self.aux_mask_conv = nn.Conv2d(self.feed_dims, num_classes[0], kernel_size=1)
        self.mask_conv = nn.Conv2d(self.feed_dims, num_classes[1], kernel_size=1)

    def forward(self, x):
        mask_feature = self.mask_feats(x)
        aux_mask_feature = self.aux_mask_feats(mask_feature)

        mask_logit = self.mask_conv(mask_feature)
        aux_mask_logit = self.aux_mask_conv(aux_mask_feature)

        return aux_mask_logit, mask_logit


class MultiTaskUNetHead(nn.Module):
    """UNet for nulcie segmentation task.

    Args:
        stage_convs (list[int]): The conv number of each stage.
            Default: [3, 3, 3, 3]
        stage_channels (list[int]): The feedforward channel number of each
            stage. Default: [16, 32, 64, 128]
    """

    def __init__(self,
                 num_classes,
                 mt_dims=64,
                 bottom_in_dim=512,
                 skip_in_dims=[64, 128, 256, 512, 512],
                 stage_dims=[16, 32, 64, 128, 256],
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()

        assert isinstance(num_classes, list)

        self.num_classes = num_classes
        self.bottom_in_dim = bottom_in_dim
        self.skip_in_dims = skip_in_dims
        self.stage_dims = stage_dims
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        num_layers = len(self.skip_in_dims)

        self.decode_layers = nn.ModuleList()
        for idx in range(num_layers - 1, -1, -1):
            if idx == num_layers - 1:
                # bottom, initial layer
                self.decode_layers.append(
                    UNetLayer(self.bottom_in_dim, self.skip_in_dims[idx], self.stage_dims[idx], 2, norm_cfg, act_cfg))
            else:
                self.decode_layers.append(
                    UNetLayer(self.stage_dims[idx + 1], self.skip_in_dims[idx], self.stage_dims[idx], 2, norm_cfg,
                              act_cfg))

        self.postprocess = MultiTaskBranches(self.stage_dims[0], mt_dims, num_classes)

    def forward(self, bottom_input, skip_inputs):
        # decode stage feed forward
        x = bottom_input
        skips = skip_inputs[::-1]

        decode_layers = self.decode_layers
        for skip, decode_stage in zip(skips, decode_layers):
            x = decode_stage(x, skip)

        out = self.postprocess(x)

        return out
