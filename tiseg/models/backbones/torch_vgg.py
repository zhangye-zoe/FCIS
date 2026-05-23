import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torchvision import models

from ..builder import BACKBONES

MODEL_DICT = {
    'vgg16_bn': models.vgg16_bn,
    'vgg19_bn': models.vgg19_bn,
}
OUTPUT_NAMES = {
    'vgg16_bn': ('5', '12', '22', '32', '42', '43'),
    'vgg19_bn': ('5', '12', '25', '38', '52', '53'),
}


@BACKBONES.register_module()
class TorchVGG(BaseModule):

    output_names = None

    def __init__(self,
                 model_name,
                 in_channels=3,
                 out_indices=(0, 1, 2, 3, 4, 5),
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

        self.stages = self.get_stages(MODEL_DICT[model_name](pretrained=pretrained).features, len(out_indices))

        if self.in_channels != 3:
            self.input_stem = ConvModule(self.in_channels, 3, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def get_stages(self, model, depth):
        stages = nn.ModuleList()
        stage_modules = []

        cur = 0
        for module_name, module in model.named_children():
            stage_modules.append(module)

            if cur < depth:
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
class TorchVGG16BN(TorchVGG):

    output_names = ('5', '12', '22', '32', '42', '43')

    def __init__(self, **kwargs):
        super().__init__(model_name='vgg16_bn', norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), **kwargs)


class TorchVGG19BN(TorchVGG):

    output_names = ('5', '12', '25', '38', '52', '53')

    def __init__(self, **kwargs):
        super().__init__(model_name='vgg19_bn', norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), **kwargs)
