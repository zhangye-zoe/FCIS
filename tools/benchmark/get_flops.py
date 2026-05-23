import argparse

import torch
from mmcv import Config
from thop import profile

from tiseg.models import build_segmentor


def test_flops(model, cfg):
    if hasattr(model, 'forward'):
        model.forward = model.forward
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    image_input = torch.randn((1,1, 3, 256,256)).cuda()
    text_input = torch.randint(0, 1, (1, 1, 1))

    if torch.cuda.is_available():
        model.cuda()
        image_input = image_input.cuda()
        text_input = text_input.cuda()

    MACs, Params = profile(model, inputs=(image_input))
    # We use twice MACs to roughly represent FLOPs
    FLOPs = 2 * MACs

    print(f'params: {(Params / 1e6):.4f}M')
    print(f'flops: {(FLOPs / 1e9):.4f}G')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate model params & model flops.')
    parser.add_argument('config', help='test config file path.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    # cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # model prepare
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    # load_checkpoint(model, args.checkpoint, map_location='cpu')

    test_flops(model, cfg)


if __name__ == '__main__':
    main()
