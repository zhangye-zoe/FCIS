import argparse
import os.path as osp

import torch
import cv2
import numpy as np
import mmcv
import matplotlib.pyplot as plt
from mmcv.runner import load_checkpoint
from PIL import Image

from tiseg.datasets.ops import TorchFormatting, Normalize
from tiseg.models import build_segmentor


def read_image(path):
    _, suffix = osp.splitext(osp.basename(path))
    if suffix == '.tif':
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif suffix == '.npy':
        img = np.load(path)
    else:
        img = Image.open(path)
        img = np.array(img)

    return img


def pillow_save(array, save_path=None, palette=None):
    """save array to a image by using pillow package.

    Args:
        array (np.ndarry): The numpy array which is need to save as an image.
        save_path (str, optional): The save path of numpy array image.
            Default: None
        palette (np.ndarry, optional): The palette for save image.
    """
    image = Image.fromarray(array.astype(np.uint8))

    if palette is not None:
        image = image.convert('P')
        image.putpalette(palette)

    if save_path is not None:
        image.save(save_path)

    return image


def pillow_load(img_path):
    return np.array(Image.open(img_path))


def parse_args():
    parser = argparse.ArgumentParser(description='test (and eval) a model')
    parser.add_argument('config', help='test config file path.')
    parser.add_argument('checkpoint', help='checkpoint file.')
    parser.add_argument('img_path', help='The inference image path.')
    parser.add_argument('--show', action='store_true', help='Whether to illustrate evaluation results.')
    parser.add_argument(
        '--show-folder', default='.nuclei_show', type=str, help='The storage folder of illustration results.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # build the model and load checkpoint
    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = model.cuda().eval()
    img = read_image(args.img_path)

    data = {}
    data['img'] = img
    data['data_info'] = [{}]
    data['seg_fields'] = []
    data['data_info'][0]['ori_hw'] = img.shape[:2]

    trans_pre = [Normalize(), TorchFormatting(data_keys=['img'], label_keys=[])]
    for p in trans_pre:
        data = p(data)
    data['data']['img'] = data['data']['img'][None, ...].cuda()
    res = model(**data)[0]

    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(res['inst_pred'])
    plt.subplot(133)
    plt.imshow(res['sem_pred'])
    plt.savefig('2.png')


if __name__ == '__main__':
    main()
