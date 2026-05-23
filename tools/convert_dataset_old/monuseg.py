import argparse
import math
import os
import os.path as osp
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


def pillow_save(save_path, array):
    """storage image array by using pillow."""
    array = Image.fromarray(array)
    array.save(save_path)


# NOTE: Old style patch crop
# def crop_patches(image, c_size, c_stride):
#     """crop image into several patches according to the crop size & slide
#     stride."""
#     h_crop = w_crop = c_size
#     h_stride = w_stride = c_stride

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


def parse_single_item(item, raw_image_folder, raw_label_folder, new_path, c_size):
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
    if c_size != 0:
        image_patches = crop_patches(image, c_size)
        instance_patches = crop_patches(instance_label, c_size)
        semantic_patches = crop_patches(semantic_label, c_size)

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
        np.save(osp.join(new_path, sub_item + '_instance.npy'), patch[1])
        # save semantic level label
        pillow_save(osp.join(new_path, sub_item + '_semantic.png'), patch[2])

    return {item: sub_item_list}


def convert_cohort(raw_image_folder, raw_label_folder, new_path, item_list, c_size=0):
    if not osp.exists(new_path):
        os.makedirs(new_path, 0o775)

    fix_kwargs = {
        'raw_image_folder': raw_image_folder,
        'raw_label_folder': raw_label_folder,
        'new_path': new_path,
        'c_size': c_size,
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
        '-c', '--crop-size', type=int, default=0, help='the crop size of fix crop in dataset convertion operation')

    return parser.parse_args()


def main():
    args = parse_args()
    root_path = args.root_path
    total_split = args.split
    c_size = args.crop_size

    assert total_split in ['official', 'only-train_t16', 'only-train_t12_v4']

    for split, name in [('train', 'MoNuSeg 2018 Training Data'), ('test', 'MoNuSegTestData')]:
        raw_root = osp.join(root_path, 'monuseg', name)

        if split == 'train':
            raw_img_folder = osp.join(raw_root, 'Tissue Images')
            raw_lbl_folder = osp.join(raw_root, 'Annotations')
            new_root = osp.join(root_path, split, f'c{c_size}')

            item_list = [x.rstrip('.tif') for x in os.listdir(raw_img_folder) if '.tif' in x]

            convert_cohort(raw_img_folder, raw_lbl_folder, new_root, item_list, c_size)
            if c_size != 0:
                new_root = osp.join(root_path, split, 'c0')
                convert_cohort(raw_img_folder, raw_lbl_folder, new_root, item_list, 0)
        else:
            raw_img_folder = raw_root
            raw_lbl_folder = raw_root
            new_root = osp.join(root_path, split, 'c0')

            item_list = [x.rstrip('.tif') for x in os.listdir(raw_img_folder) if '.tif' in x]

            convert_cohort(raw_img_folder, raw_lbl_folder, new_root, item_list, 0)

    train_img_folder = osp.join(root_path, 'train', f'c{c_size}')
    test_img_folder = osp.join(root_path, 'test', 'c0')

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
                if item in name and '_instance.npy' in name:
                    name = name.replace('_instance.npy', '')
                    train_item_list.append(name)
        val_item_list = None
        test_item_list = split_dict['test1'] + split_dict['test2']
    elif total_split == 'only-train_t12_v4':
        item_list = split_dict['train']
        train_item_list = []
        for item in item_list:
            name_list = [x.rstrip('.tif') for x in os.listdir(train_img_folder)]
            for name in name_list:
                if item in name and '_instance.npy' in name:
                    name = name.replace('_instance.npy', '')
                    train_item_list.append(name)
        val_item_list = split_dict['val']
        test_item_list = split_dict['test1'] + split_dict['test2']

    with open(osp.join(root_path, f'{total_split}_train_c{c_size}.txt'), 'w') as fp:
        [fp.write(item + '\n') for item in train_item_list]
    with open(osp.join(root_path, f'{total_split}_test_c0.txt'), 'w') as fp:
        [fp.write(item + '\n') for item in test_item_list]

    if val_item_list is not None:
        with open(osp.join(root_path, f'{total_split}_val_c0.txt'), 'w') as fp:
            [fp.write(item + '\n') for item in val_item_list]


if __name__ == '__main__':
    main()
