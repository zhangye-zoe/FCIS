from functools import partial

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torchvision import models

from ..builder import BACKBONES

MODEL_DICT = {
    'resnet18-d32':
    models.resnet18,
    'resnet34-d32':
    models.resnet34,
    'resnet50-d32':
    models.resnet50,
    'resnet101-d32':
    models.resnet101,
    'resnet50-d8':
    partial(models.resnet50, replace_stride_with_dilation=(False, 2, 4)),
    'resnet101-d8':
    partial(models.resnet101, replace_stride_with_dilation=(False, 2, 4)),
}
OUTPUT_NAMES = {
    'resnet18-d32': ('relu', 'layer1', 'layer2', 'layer3', 'layer4'),
    'resnet34-d32': ('relu', 'layer1', 'layer2', 'layer3', 'layer4'),
    'resnet50-d32': ('relu', 'layer1', 'layer2', 'layer3', 'layer4'),
    'resnet101-d32': ('relu', 'layer1', 'layer2', 'layer3', 'layer4'),
    'resnet50-d8': ('relu', 'layer1', 'layer2', 'layer3', 'layer4'),
    'resnet101-d8': ('relu', 'layer1', 'layer2', 'layer3', 'layer4'),
}


@BACKBONES.register_module()
class TorchResNet(BaseModule):

    output_names = None

    def __init__(self,
                 model_name,
                 in_channels=3,
                 out_indices=(0, 1, 2, 3, 4),
                 group_base_channels=64,
                 pretrained=True,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.model_name = model_name
        self.in_channels = in_channels
        self.out_indices = out_indices
        if self.output_names is None:
            self.output_names = OUTPUT_NAMES[model_name]

        assert len(self.out_indices) <= len(self.output_names)

        self.stages = self.get_stages(
            MODEL_DICT[model_name](
                pretrained=pretrained, width_per_group=group_base_channels),
            max(out_indices))

        if self.in_channels != 3:
            self.input_stem = ConvModule(
                self.in_channels, 3, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def get_stages(self, model, depth):
        stages = nn.ModuleList()
        stage_modules = []

        cur = 0
        for module_name, module in model.named_children():
            stage_modules.append(module)

            if cur <= depth:
                if module_name == self.output_names[cur]:
                    stages.append(nn.Sequential(*stage_modules))
                    stage_modules = []
                    cur += 1
            else:
                break

        return stages

    def forward(self, x):
        if self.in_channels != 3:
            x = self.input_stem(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)

        return outs


@BACKBONES.register_module()
class TorchResNet18(TorchResNet):

    output_names = ('relu', 'layer1', 'layer2', 'layer3', 'layer4')

    def __init__(self, **kwargs):
        super().__init__(
            model_name='resnet18-d32',
            grou_base_channels=64,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            **kwargs)


@BACKBONES.register_module()
class TorchResNet34(TorchResNet):

    output_names = ('relu', 'layer1', 'layer2', 'layer3', 'layer4')

    def __init__(self, **kwargs):
        super().__init__(
            model_name='resnet34-d32',
            grou_base_channels=64,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            **kwargs)


@BACKBONES.register_module()
class TorchResNet50(TorchResNet):

    output_names = ('relu', 'layer1', 'layer2', 'layer3', 'layer4')

    def __init__(self, **kwargs):
        super().__init__(
            model_name='resnet50-d32',
            grou_base_channels=64,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            **kwargs)


@BACKBONES.register_module()
class TorchResNet101(TorchResNet):

    output_names = ('relu', 'layer1', 'layer2', 'layer3', 'layer4')

    def __init__(self, **kwargs):
        super().__init__(
            model_name='resnet101-d32',
            grou_base_channels=64,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            **kwargs)


@BACKBONES.register_module()
class TorchDeeplabResNet50(TorchResNet):

    output_names = ('relu', 'layer1', 'layer2', 'layer3', 'layer4')

    def __init__(self, **kwargs):
        super().__init__(
            model_name='resnet50-d8',
            grou_base_channels=64,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            **kwargs)


@BACKBONES.register_module()
class TorchDeeplabResNet101(TorchResNet):

    output_names = ('relu', 'layer1', 'layer2', 'layer3', 'layer4')

    def __init__(self, **kwargs):
        super().__init__(
            model_name='resnet101-d8',
            grou_base_channels=64,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            **kwargs)
