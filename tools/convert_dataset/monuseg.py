import argparse
import math
import os
import os.path as osp
import random
from functools import partial

import cv2
import mmcv
import numpy as np
from lxml import etree
from PIL import Image

# dataset split
split_dict = {
    'train': [
        'TCGA-A7-A13E-01Z-00-DX1', 'TCGA-A7-A13F-01Z-00-DX1', 'TCGA-AR-A1AK-01Z-00-DX1', 'TCGA-B0-5711-01Z-00-DX1',
        'TCGA-HE-7128-01Z-00-DX1', 'TCGA-HE-7129-01Z-00-DX1', 'TCGA-18-5592-01Z-00-DX1', 'TCGA-38-6178-01Z-00-DX1',
        'TCGA-49-4488-01Z-00-DX1', 'TCGA-G9-6336-01Z-00-DX1', 'TCGA-G9-6348-01Z-00-DX1', 'TCGA-G9-6356-01Z-00-DX1'
    ],
    'val': ['TCGA-AR-A1AS-01Z-00-DX1', 'TCGA-HE-7130-01Z-00-DX1', 'TCGA-50-5931-01Z-00-DX1', 'TCGA-G9-6363-01Z-00-DX1'],
    'test1': [
        'TCGA-E2-A1B5-01Z-00-DX1', 'TCGA-E2-A14V-01Z-00-DX1', 'TCGA-B0-5710-01Z-00-DX1', 'TCGA-B0-5698-01Z-00-DX1',
        'TCGA-21-5784-01Z-00-DX1', 'TCGA-21-5786-01Z-00-DX1', 'TCGA-CH-5767-01Z-00-DX1', 'TCGA-G9-6362-01Z-00-DX1'
    ],
    'test2': [
        'TCGA-DK-A2I6-01A-01-TS1', 'TCGA-G2-A2EK-01A-02-TSB', 'TCGA-AY-A8YK-01A-01-TS1', 'TCGA-NH-A8F7-01A-01-TS1',
        'TCGA-KB-A93J-01A-01-TS1', 'TCGA-RD-A8N9-01A-01-TS1'
    ]
}


def extract_contours(path):
    """Extract contours (multiple vertexs) from xml file.

    Args:
        path (pathlib.PosixPath): path to the annotation file.
    Returns:
        list: list of contours with each annotation encoded
           as numpy.ndarray values.
    """
    tree = etree.parse(path)
    regions = tree.xpath('/Annotations/Annotation/Regions/Region')
    contours = []
    for region in regions:
        points = []
        for point in region.xpath('Vertices/Vertex'):
            points.append([math.floor(float(point.attrib['X'])), math.floor(float(point.attrib['Y']))])

        contours.append(np.array(points, dtype=np.int32))
    return contours


def convert_contour_to_instance(contours, height, width):
    """Make the mask image from the contour annotations and image sizes.

    Args:
        contours (list): list of contours with each annotation encoded
           as numpy.ndarray values.
        height (int): The height of mask.
        width (int): The width of mask.

    Returns:
        mask (numpy.ndarray): mask contains multiple instances with different
            label value.
    """

    mask = np.zeros([height, width], dtype=np.float32)
    red, green, blue = 0, 0, 0
    index_value = 0
    for contour in contours:
        # Compress value to avoid overflow
        red = 1 + index_value / 10
        red = float(f'{red:.2f}')
        mask = cv2.drawContours(mask, [contour], 0, (red, green, blue), thickness=cv2.FILLED)
        index_value = index_value + 1

    return mask


def pillow_save(save_path, array, palette=None):
    """storage image array by using pillow."""
    image = Image.fromarray(array.astype(np.uint8))
    if palette is not None:
        image = image.convert('P')
        image.putpalette(palette)
    image.save(save_path)


def colorize_seg_map(seg_map):
    """using random rgb color to colorize segmentation map."""
    colorful_seg_map = np.zeros((*seg_map.shape, ), dtype=np.float32)
    id_list = list(np.unique(seg_map))

    for id_ in id_list:
        if id_ == 0:
            continue
        colorful_seg_map[seg_map == id_] = random.random()

    colorful_seg_map = cv2.applyColorMap((colorful_seg_map * 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    colorful_seg_map[seg_map == 0, :] = (0, 0, 0)
    colorful_seg_map = cv2.cvtColor(colorful_seg_map, cv2.COLOR_BGR2RGB)

    return colorful_seg_map


# NOTE: new style patch crop.
def crop_patches(image, w_size, s_size):
    """Modifed from https://github.com/vqdang/hover_net/blob/master/misc/patch_extractor.py"""
    patches = []

    diff = w_size - s_size
    pad1 = diff // 2
    pad2 = diff - pad1

    if len(image.shape) == 2:
        image = image[:, :, None]
        image = np.lib.pad(image, ((pad1, pad2), (pad1, pad2), (0, 0)), 'reflect')
        image = image[:, :, 0]
    elif len(image.shape) == 3:
        image = np.lib.pad(image, ((pad1, pad2), (pad1, pad2), (0, 0)), 'reflect')

    pad_h, pad_w = image.shape[:2]
    h_last_step = math.floor((pad_h - w_size) / s_size)
    h_last = (h_last_step + 1) * s_size
    w_last_step = math.floor((pad_w - w_size) / s_size)
    w_last = (w_last_step + 1) * s_size

    for i in range(0, h_last, s_size):
        for j in range(0, w_last, s_size):
            patch = image[i:(i + w_size), j:(j + w_size)]
            patches.append(patch)

    if h_last_step > ((pad_h - w_size + s_size) // s_size):
        i = pad_h - w_size
        for j in range(0, w_last, s_size):
            patch = image[i:(i + w_size), j:(j + w_size)]
            patches.append(patch)

    if w_last_step > ((pad_w - w_size + s_size) // s_size):
        j = pad_w - w_size
        for i in range(0, h_last, s_size):
            patch = image[i:(i + w_size), j:(j + w_size)]
            patches.append(patch)

    if h_last_step > ((pad_h - w_size + s_size) // s_size) and w_last_step > ((pad_w - w_size + s_size) // s_size):
        i = pad_h - w_size
        j = pad_w - w_size
        patches.append(image[i:(i + w_size), j:(j + w_size)])

    return patches


def parse_single_item(item, raw_image_folder, raw_label_folder, new_path, w_size, s_size):
    """meta process of single item data."""

    image_path = osp.join(raw_image_folder, item + '.tif')
    label_path = osp.join(raw_label_folder, item + '.xml')

    # image & label extraction
    image = cv2.imread(image_path)
    H, W = image.shape[:2]
    contours = extract_contours(label_path)
    instance_label = convert_contour_to_instance(contours, H, W)
    semantic_label = (instance_label > 0).astype(np.uint8)

    # split map into patches
    if w_size != 0:
        image_patches = crop_patches(image, w_size, s_size)
        instance_patches = crop_patches(instance_label, w_size, s_size)
        semantic_patches = crop_patches(semantic_label, w_size, s_size)

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
        if osp.exists(osp.join(new_path, sub_item + '.tif')):
            continue
        # save image
        cv2.imwrite(osp.join(new_path, sub_item + '.tif'), patch[0])
        # save instance level label
        np.save(osp.join(new_path, sub_item + '_inst.npy'), patch[1])
        # save colorized instance level label
        pillow_save(osp.join(new_path, sub_item + '_inst_color.png'), colorize_seg_map(patch[1]))
        # save semantic level label
        palette = np.zeros((2, 3), dtype=np.uint8)
        palette[0, :] = (0, 0, 0)
        palette[1, :] = (255, 255, 2)
        pillow_save(osp.join(new_path, sub_item + '_sem.png'), patch[2], palette=palette)

    return {item: sub_item_list}


def convert_cohort(raw_image_folder, raw_label_folder, new_path, item_list, w_size=0, s_size=0):
    if not osp.exists(new_path):
        os.makedirs(new_path, 0o775)

    fix_kwargs = {
        'raw_image_folder': raw_image_folder,
        'raw_label_folder': raw_label_folder,
        'new_path': new_path,
        'w_size': w_size,
        's_size': s_size,
    }
    meta_process = partial(parse_single_item, **fix_kwargs)

    real_item_dict = {}
    results = mmcv.track_parallel_progress(meta_process, item_list, 4)
    [real_item_dict.update(result) for result in results]

    return real_item_dict


def parse_args():
    parser = argparse.ArgumentParser('Convert monuseg dataset.')
    parser.add_argument('root_path', help='dataset root path.')
    parser.add_argument('split', help='split mode selection.')
    parser.add_argument(
        '-w',
        '--window-size',
        type=int,
        default=0,
        help='the window size of step crop in dataset convertion operation.')
    parser.add_argument('-s', '--step-size', type=int, default=0, help='patch croping step.')

    return parser.parse_args()


def main():
    args = parse_args()
    root_path = args.root_path
    total_split = args.split
    # only support images with square shape now.
    w_size = args.window_size
    s_size = args.step_size

    assert w_size > s_size

    assert total_split in ['official', 'only-train_t16', 'only-train_t12_v4']

    for split, name in [('train', 'MoNuSeg 2018 Training Data'), ('test', 'MoNuSegTestData')]:
        raw_root = osp.join(root_path, 'monuseg', name)

        if split == 'train':
            raw_img_folder = osp.join(raw_root, 'Tissue Images')
            raw_lbl_folder = osp.join(raw_root, 'Annotations')
            new_root = osp.join(root_path, split, f'w{w_size}_s{s_size}')

            item_list = [x.rstrip('.tif') for x in os.listdir(raw_img_folder) if '.tif' in x]

            convert_cohort(raw_img_folder, raw_lbl_folder, new_root, item_list, w_size, s_size)
            if w_size != 0:
                new_root = osp.join(root_path, split, 'w0_s0')
                convert_cohort(raw_img_folder, raw_lbl_folder, new_root, item_list, 0, 0)
        else:
            raw_img_folder = raw_root
            raw_lbl_folder = raw_root
            new_root = osp.join(root_path, split, 'w0_s0')

            item_list = [x.rstrip('.tif') for x in os.listdir(raw_img_folder) if '.tif' in x]

            convert_cohort(raw_img_folder, raw_lbl_folder, new_root, item_list, 0, 0)

    train_img_folder = osp.join(root_path, 'train', f'w{w_size}_s{s_size}')
    test_img_folder = osp.join(root_path, 'test', 'w0_s0')

    if total_split == 'official':
        train_item_list = [x.rstrip('.tif') for x in os.listdir(train_img_folder) if '.tif' in x]
        val_item_list = None
        test_item_list = [x.rstrip('.tif') for x in os.listdir(test_img_folder) if '.tif' in x]
    elif total_split == 'only-train_t16':
        item_list = split_dict['train'] + split_dict['val']
        train_item_list = []
        for item in item_list:
            name_list = [x.rstrip('.tif') for x in os.listdir(train_img_folder)]
            for name in name_list:
                if item in name and '_inst.npy' in name:
                    name = name.replace('_inst.npy', '')
                    train_item_list.append(name)
        val_item_list = None
        test_item_list = split_dict['test1'] + split_dict['test2']
    elif total_split == 'only-train_t12_v4':
        item_list = split_dict['train']
        train_item_list = []
        for item in item_list:
            name_list = [x.rstrip('.tif') for x in os.listdir(train_img_folder)]
            for name in name_list:
                if item in name and '_inst.npy' in name:
                    name = name.replace('_inst.npy', '')
                    train_item_list.append(name)
        val_item_list = split_dict['val']
        test_item_list = split_dict['test1'] + split_dict['test2']

    with open(osp.join(root_path, f'{total_split}_train_w{w_size}_s{s_size}.txt'), 'w') as fp:
        [fp.write(item + '\n') for item in train_item_list]
    with open(osp.join(root_path, f'{total_split}_test_w0_s0.txt'), 'w') as fp:
        [fp.write(item + '\n') for item in test_item_list]

    if val_item_list is not None:
        with open(osp.join(root_path, f'{total_split}_val_w0_s0.txt'), 'w') as fp:
            [fp.write(item + '\n') for item in val_item_list]


if __name__ == '__main__':
    main()
