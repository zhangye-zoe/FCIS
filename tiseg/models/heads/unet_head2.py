import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


def conv1x1(in_dims, out_dims, norm_cfg=None, act_cfg=None):
    return ConvModule(in_dims, out_dims, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)


def conv3x3(in_dims, out_dims, norm_cfg=None, act_cfg=None):
    return ConvModule(in_dims, out_dims, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)


def transconv4x4(in_dims, out_dims, bn):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_dims, out_channels=out_dims, kernel_size=(4, 4), stride=2, padding=1, bias=not bn),
        nn.BatchNorm2d(out_dims),
        nn.ReLU(),
    )


class UNetLayer(nn.Module):

    def __init__(self, in_dims, skip_dims, feed_dims, num_convs=2, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_dims = in_dims
        self.skip_dims = skip_dims
        self.feed_dims = feed_dims

        self.up_conv = transconv4x4(in_dims, feed_dims, norm_cfg is not None)

        convs = [conv3x3(skip_dims + feed_dims, feed_dims, norm_cfg, act_cfg)]
        for _ in range(num_convs - 2):
            convs.append(conv3x3(feed_dims, feed_dims, norm_cfg, act_cfg))
        self.convs = nn.Sequential(*convs)

    def forward(self, x, skip):
        x = self.up_conv(x)

        if x.shape != skip.shape:
            diff_h = skip.shape[-2] - x.shape[-2]
            diff_w = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (diff_h // 2, diff_h - diff_h // 2, diff_w // 2, diff_w - diff_w // 2))

        x = torch.cat([x, skip], dim=1)
        out = self.convs(x)
        return out


class UNetHead_FCIS(nn.Module):
    """UNet for nulcie segmentation task.

    Args:
        stage_convs (list[int]): The conv number of each stage.
            Default: [3, 3, 3, 3]
        stage_channels (list[int]): The feedforward channel number of each
            stage. Default: [16, 32, 64, 128]
    """

    def __init__(self,
                 num_classes=None,
                 bottom_in_dim=512,
                 skip_in_dims=[64, 128, 256, 512, 512],
                 stage_dims=[16, 32, 64, 128, 256],
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
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

        if self.num_classes is not None:
            # self.cls_layer = nn.Conv2d(self.stage_dims[0]+16, self.num_classes, kernel_size=1, stride=1)
            self.cls_layer = nn.Conv2d(self.stage_dims[0], self.num_classes, kernel_size=1, stride=1)
            self.sem_layer = nn.Conv2d(self.num_classes-1, 1, kernel_size=1, stride=1)

    def forward(self, bottom_input, skip_inputs, encoding=None):
        # decode stage feed forward
        x = bottom_input
        skips = skip_inputs[::-1]

        decode_layers = self.decode_layers
        for skip, decode_stage in zip(skips, decode_layers):
            x = decode_stage(x, skip)

        out = x
        
        
        if encoding is not None:
            device = out.device
            print(" Using image encoding ......")
            pos_encode = encoding.repeat(out.shape[0], 1, 1, 1).to(device)
            out = torch.cat((out, pos_encode), dim=1)

        if self.num_classes is not None:
            inter_pred = self.cls_layer(out)
            back_pred = inter_pred[:, :1, ...]
            cls_pred = inter_pred[:, 1:, ...]

            fore_pred = self.sem_layer(cls_pred)
            sem_pred = torch.cat([back_pred, fore_pred], dim=1)

        return sem_pred, inter_pred
