import argparse
import math
import os
import os.path as osp
from functools import partial

import mmcv
import numpy as np
from PIL import Image
from scipy.io import loadmat

from tiseg.datasets.utils import colorize_seg_map


def convert_mat_to_array(mat, array_key='inst_map', save_path=None):
    """Convert matlab format array file to numpy array file."""
    if isinstance(mat, str):
        mat = loadmat(mat)

    mat = mat[array_key]

    if save_path is not None:
        pass

    return mat


def pillow_save(save_path, array, palette=None):
    """storage image array by using pillow."""
    image = Image.fromarray(array.astype(np.uint8))
    if palette is not None:
        image = image.convert('P')
        image.putpalette(palette)
    image.save(save_path)


# NOTE: Old style patch crop
# def crop_patches(image, crop_size, crop_stride):
#     """crop image into several patches according to the crop size & slide
#     stride."""
#     h_crop = w_crop = crop_size
#     h_stride = w_stride = crop_stride

#     assert image.ndim >= 2

#     h_img, w_img = image.shape[:2]

#     image_patch_list = []

#     h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
#     w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

#     for h_idx in range(h_grids):
#         for w_idx in range(w_grids):
#             y1 = h_idx * h_stride
#             x1 = w_idx * w_stride
#             y2 = min(y1 + h_crop, h_img)
#             x2 = min(x1 + w_crop, w_img)
#             y1 = max(y2 - h_crop, 0)
#             x1 = max(x2 - w_crop, 0)
#             crop_img = image[y1:y2, x1:x2]

#             image_patch_list.append(crop_img)

#     return image_patch_list


# NOTE: new style patch crop.
def crop_patches(image, c_size):
    h, w = image.shape[:2]
    patches = []

    if h % c_size == 0:
        h_overlap = 0
    else:
        div = h // c_size
        h_overlap = math.ceil(((div + 1) * c_size - h) / div)

    if w % c_size == 0:
        w_overlap = 0
    else:
        div = w // c_size
        w_overlap = math.ceil(((div + 1) * c_size - w) / div)

    for i in range(0, h - c_size + 1, c_size - h_overlap):
        for j in range(0, w - c_size + 1, c_size - w_overlap):
            patch = image[i:i + c_size, j:j + c_size]
            patches.append(patch)

    return patches


def parse_single_item(item, raw_image_folder, raw_label_folder, new_path, crop_size):
    """meta process of single item data."""

    image_path = osp.join(raw_image_folder, item + '.bmp')
    label_path = osp.join(raw_label_folder, item + '_anno.bmp')

    # image & label extraction
    image = np.array(Image.open(image_path))
    instance_label = np.array(Image.open(label_path))
    semantic_label = (instance_label > 0).astype(np.uint8)

    # split map into patches
    if crop_size != 0:
        image_patches = crop_patches(image, crop_size)
        instance_patches = crop_patches(instance_label, crop_size)
        semantic_patches = crop_patches(semantic_label, crop_size)

        assert len(image_patches) == len(instance_patches) == len(semantic_patches)

        item_len = len(image_patches)
        # record patch item name
        sub_item_list = [f'{item}_{i}' for i in range(item_len)]
    else:
        image_patches = [image]
        instance_patches = [instance_label]
        semantic_patches = [semantic_label]
        # record patch item name
        sub_item_list = [item]

    # patch storage
    patch_batches = zip(image_patches, instance_patches, semantic_patches)
    for patch, sub_item in zip(patch_batches, sub_item_list):
        # jump when exists
        if osp.exists(osp.join(new_path, sub_item + '.png')):
            continue
        # save image
        pillow_save(osp.join(new_path, sub_item + '.png'), patch[0])
        # save instance level label
        np.save(osp.join(new_path, sub_item + '_instance.npy'), patch[1])
        pillow_save(osp.join(new_path, sub_item + '_instance_colorized.png'), colorize_seg_map(patch[1]))
        # save semantic level label
        palette = np.zeros((2, 3), dtype=np.uint8)
        palette[0, :] = (0, 0, 0)
        palette[1, :] = (255, 255, 2)
        pillow_save(osp.join(new_path, sub_item + '_semantic.png'), patch[2], palette)

    return {item: sub_item_list}


def convert_cohort(img_folder, lbl_folder, new_folder, item_list, c_size=0):
    if not osp.exists(new_folder):
        os.makedirs(new_folder, 0o775)

    fix_kwargs = {
        'raw_image_folder': img_folder,
        'raw_label_folder': lbl_folder,
        'new_path': new_folder,
        'crop_size': c_size,
    }

    meta_process = partial(parse_single_item, **fix_kwargs)

    real_item_dict = {}
    results = mmcv.track_parallel_progress(meta_process, item_list, 4)
    [real_item_dict.update(result) for result in results]

    return real_item_dict


def parse_args():
    parser = argparse.ArgumentParser('Convert cpm17 dataset.')
    parser.add_argument('root_path', help='dataset root path.')
    parser.add_argument(
        '-c', '--crop-size', type=int, default=0, help='the crop size of fix crop in dataset convertion operation')

    return parser.parse_args()


def main():
    args = parse_args()
    root_path = args.root_path
    crop_size = args.crop_size

    for split in ['train', 'testA', 'testB']:
        raw_root = osp.join(root_path, 'glas')

        item_list = [x.rstrip('_anno.bmp') for x in os.listdir(raw_root) if '_anno.bmp' in x and split in x]

        # temp = np.array(Image.open(osp.join(raw_root, item_list[23] + '_anno.bmp')))

        # print(np.unique(temp))

        # import cv2
        # temp = cv2.resize(temp, (512, 512))

        # import matplotlib.pyplot as plt
        # plt.imshow(temp)
        # plt.savefig('2.png')

        # exit(0)

        raw_img_folder = raw_root
        raw_lbl_folder = raw_root

        if split == 'testA' or split == 'testB':
            new_root = osp.join(root_path, split, 'c0')
            convert_cohort(raw_img_folder, raw_lbl_folder, new_root, item_list, 0)
        else:
            new_root = osp.join(root_path, split, f'c{crop_size}')
            convert_cohort(raw_img_folder, raw_lbl_folder, new_root, item_list, c_size=crop_size)

        item_list = [x.rstrip('_instance.npy') for x in os.listdir(new_root) if '_instance.npy' in x]

        if split == 'train':
            name = f'train_c{crop_size}.txt'
        else:
            name = f'{split}_c0.txt'
        with open(osp.join(root_path, name), 'w') as fp:
            [fp.write(item + '\n') for item in item_list]


if __name__ == '__main__':
    main()
