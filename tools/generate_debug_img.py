import argparse
import os
import os.path as osp

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def min_max_norm(feat):
    min_v = np.min(feat)
    max_v = np.max(feat)
    return (feat - min_v) / (max_v - min_v)


def color_map(mask, compressed=True):
    if compressed:
        mask = min_max_norm(mask)
    tmp = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    return tmp


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
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='The data storage path.')
    parser.add_argument('-s', '--save_folder', default='show/', help='The visual results storage folder.')

    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data_path
    save_folder = args.save_folder

    if not osp.exists(save_folder):
        os.makedirs(save_folder, 0o775)

    tmp = np.load(data_path)
    img = tmp[:, :, :3].astype(np.uint8)
    sem_gt = tmp[:, :, 3].astype(np.uint8)
    inst_gt = tmp[:, :, 4].astype(np.int32)
    dir_gt = tmp[:, :, 5].astype(np.uint8)
    point_gt = tmp[:, :, 6]
    tc_gt = tmp[:, :, 7].astype(np.uint8)
    raw_loss = tmp[:, :, 8]
    weight_loss = tmp[:, :, 9]

    # raw_loss[raw_loss < 1] = 0
    # raw_loss[raw_loss > 1] = np.log2(raw_loss[raw_loss > 1])
    weight_loss[weight_loss < 1] = 0
    weight_loss[weight_loss > 1] = np.log2(weight_loss[weight_loss > 1])

    sem_gt_color = color_map(sem_gt)
    inst_gt_color = color_map(inst_gt)
    dir_gt_color = color_map(dir_gt)
    point_gt_color = color_map(point_gt)
    tc_gt_color = color_map(tc_gt)
    raw_loss_color = color_map(raw_loss)
    weight_loss_color = color_map(weight_loss)

    pillow_save(img, osp.join(save_folder, 'img.png'))
    pillow_save(sem_gt_color, osp.join(save_folder, 'sem.png'))
    pillow_save(inst_gt_color, osp.join(save_folder, 'inst.png'))
    pillow_save(dir_gt_color, osp.join(save_folder, 'dir.png'))
    pillow_save(point_gt_color, osp.join(save_folder, 'point.png'))
    pillow_save(tc_gt_color, osp.join(save_folder, 'tc.png'))
    pillow_save(raw_loss_color, osp.join(save_folder, 'raw_dir_loss.png'))
    pillow_save(weight_loss_color, osp.join(save_folder, 'weighted_dir_loss.png'))

    plt.figure(dpi=300)
    plt.subplot(241)
    plt.imshow(img)
    plt.subplot(242)
    plt.imshow(sem_gt_color)
    plt.subplot(243)
    plt.imshow(inst_gt_color)
    plt.subplot(244)
    plt.imshow(dir_gt_color)
    plt.subplot(245)
    plt.imshow(point_gt_color)
    plt.subplot(246)
    plt.imshow(tc_gt_color)
    plt.subplot(247)
    plt.imshow(raw_loss_color)
    plt.subplot(248)
    plt.imshow(weight_loss_color)
    plt.savefig('2.png')


if __name__ == '__main__':
    main()
