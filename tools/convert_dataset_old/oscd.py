"""# img w/o instance annotations.

| img_id | split | img_filename |
| :--  | :-- | :-- |
| 161  | train | net (13004).jpg |
| 293  | train | net (6405).jpg  |
| 609  | train | net (665).jpg   |
| 796  | train | net (8577).jpg  |
| 1288 | train | net (4403).jpg  |
| 1449 | train | net (2006).jpg  |
| 1562 | train | net (12506).jpg |
| 1902 | train | net (4007).jpg  |
| 2278 | train | net (9182).jpg  |
| 3438 | train | net (904).jpg   |
| 3476 | train | net (4016).jpg  |
| 4264 | train | net (2003).jpg  |
| 4695 | train | net (11398).jpg |
| 5116 | train | net (12548).jpg |
| 5208 | train | net (3325).jpg  |
| 5566 | train | net (3511).jpg  |
| 6290 | train | net (11800).jpg |
| 6436 | train | net (9181).jpg  |
| 6587 | train | net (2218).jpg  |
| 6698 | train | net (3555).jpg  |
| 7102 | train | net (4012).jpg  |
| 42   | val   | net (4019).jpg  |
| 273  | val   | net (9125).jpg  |
| 356  | val   | net (8060).jpg  |
| 603  | val   | net (2001).jpg  |
| 984  | val   | net (16065).jpg |

# img has error polygon

| img_id | split | img_filename |
| :--  | :--   | :--            |
| 4858 | train | net (2941).jpg |
| 5356 | train | net (1909).jpg |

img json has error height, width

| img id | split | img_filename |
| :--  | :--   | :--             |
| 4361 | train | net (957).jpg   |
| 4781 | train | net (2954).jpg  |
| 5782 | train | net (4114).jpg  |
| 7342 | train | net (11006).jpg |
| 523  | val   | net (2445).jpg  |
| 700  | val   | net(6766).jpg   |
"""

import argparse
import os
import os.path as osp
import math
import random
from functools import partial

import mmcv
import numpy as np
from PIL import Image
from pycocotools import mask, coco

cyan_color_list = [
    (0, 0, 0),
    (41, 171, 135),
    (176, 220, 213),
    (127, 200, 255),
    (0, 127, 255),
    (50, 18, 122),
]

CLASSES = {0: 'background', 1: 'carton', 2: 'edge'}

EDGE_ID = 2


def polygon_to_mask(polygon, height, width, path=None):
    if len(polygon[0]) == 4:
        polygon = np.array(polygon, dtype=np.float64)
    rle_mask = mask.frPyObjects(polygon, height, width)
    ann_mask = mask.decode(rle_mask)
    # one mask contains only one instance, but one instance may be split into
    # several parts.
    ann_mask = np.max(ann_mask, axis=2)

    if path is not None:
        ann_mask_img = Image.fromarray(ann_mask.astype(np.uint8)).convert('P')

        # set palette
        color = cyan_color_list[random.randint(1, len(cyan_color_list) - 1)]
        palette = np.array([(0, 0, 0), color], dtype=np.uint8)
        ann_mask_img.putpalette(palette)
        ann_mask_img.save(path)

    return ann_mask


def pillow_save(img, path=None, palette=None):
    img = Image.fromarray(img)

    if palette is not None:
        img = img.convert('P')
        img.putpalette(palette)

    if path is not None:
        img.save(path)

    return img


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


def convert_single_img(task, src_img_folder, img_folder, ann_folder, c_size):
    if not osp.exists(img_folder):
        os.makedirs(img_folder, 0o775)

    if not osp.exists(ann_folder):
        os.makedirs(ann_folder, 0o775)

    imgs, anns = task

    img_filename = imgs['file_name']
    img_name = osp.splitext(img_filename)[0]
    # read real img data from img_path to correct error img height & width
    img_path = osp.join(src_img_folder, img_filename)
    img = Image.open(img_path)
    iheight, iwidth = img.height, img.width
    height, width = imgs['height'], imgs['width']

    if iheight != height or iwidth != width:
        height = iheight
        width = iwidth
    img = np.array(img)

    instance_canvas = np.zeros((height, width), dtype=np.int64)
    semantic_canvas = np.zeros((height, width), dtype=np.uint8)
    for idx, ann in enumerate(anns):
        cat_id = ann['category_id']
        # extract polygons & bounding boxes
        object_json = {}
        object_json['label'] = CLASSES[ann['category_id']]
        object_json['polygon'] = ann['segmentation']
        object_json['bbox'] = ann['bbox']
        # polygon may have error format.
        if len(object_json['polygon'][0]) == 4:
            continue

        # instance mask conversion
        instance_mask = np.array(polygon_to_mask(ann['segmentation'], height, width))

        instance_canvas[instance_mask > 0] = idx + 1
        semantic_canvas[instance_mask > 0] = cat_id

    # save instance label & semantic label & polygon
    semantic_palette = np.array([(0, 0, 0), (255, 2, 255), (2, 255, 255)], dtype=np.uint8)

    # split map into patches
    if c_size != 0:
        image_patches = crop_patches(img, c_size)
        instance_patches = crop_patches(instance_canvas, c_size)
        semantic_patches = crop_patches(semantic_canvas, c_size)

        assert len(image_patches) == len(instance_patches) == len(semantic_patches)

        item_len = len(image_patches)
        # record patch item name
        sub_name_list = [f'{img_name}_{i}' for i in range(item_len)]
    else:
        image_patches = [img]
        instance_patches = [instance_canvas]
        semantic_patches = [semantic_canvas]
        # record patch item name
        sub_name_list = [img_name]

    # patch storage
    patch_batches = zip(image_patches, instance_patches, semantic_patches)
    for patch, sub_item in zip(patch_batches, sub_name_list):
        # jump when exists
        if osp.exists(osp.join(img_folder, sub_item + '.png')):
            continue
        # save image
        pillow_save(patch[0], osp.join(img_folder, sub_item + '.png'))
        # save instance level label
        np.save(osp.join(ann_folder, sub_item + '_instance.npy'), patch[1])
        # save semantic level label
        pillow_save(patch[2], osp.join(ann_folder, sub_item + '_semantic.png'), semantic_palette)

    return sub_name_list


def parse_args():
    parser = argparse.ArgumentParser('SCD dataset conversion')
    parser.add_argument('dataset_root', help='The root path of dataset.')
    parser.add_argument(
        '-c', '--crop-size', type=int, default=0, help='the crop size of fix crop in dataset convertion operation')

    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root
    crop_size = args.crop_size

    img_base_folder = osp.join(dataset_root, 'images')
    ann_base_folder = osp.join(dataset_root, 'annotations')

    split_list = ['train', 'val']

    for split in split_list:
        # img source & annotation source
        src_img_folder = osp.join(dataset_root, f'coco_carton/oneclass_carton/images/{split}2017/')
        src_ann_json = osp.join(dataset_root, ('coco_carton/oneclass_carton/annotations/'
                                               f'instances_{split}2017.json'))

        src_ann_json = coco.COCO(src_ann_json)

        img_ids = src_ann_json.getImgIds()
        imgs = src_ann_json.loadImgs(img_ids)
        anns = [src_ann_json.loadAnns(src_ann_json.getAnnIds(imgIds=[img_id])) for img_id in img_ids]

        if split == 'train':
            img_folder = osp.join(img_base_folder, f'c{crop_size}')
            ann_folder = osp.join(ann_base_folder, f'c{crop_size}')
            # define single loop job
            loop_job = partial(
                convert_single_img,
                src_img_folder=src_img_folder,
                img_folder=img_folder,
                ann_folder=ann_folder,
                c_size=crop_size,
            )
        else:
            img_folder = osp.join(img_base_folder, 'c0')
            ann_folder = osp.join(ann_base_folder, 'c0')
            # define single loop job
            loop_job = partial(
                convert_single_img,
                src_img_folder=src_img_folder,
                img_folder=img_folder,
                ann_folder=ann_folder,
                c_size=0,
            )

        tasks = list(zip(imgs, anns))

        # only build miss imgs & labels
        records = mmcv.track_parallel_progress(loop_job, tasks, 4)

        img_names = []
        [img_names.extend(names) for names in records]
        real_crop_size = crop_size if split == 'train' else 0
        with open(f'{dataset_root}/{split}_c{real_crop_size}.txt', 'w') as fp:
            for img_name in img_names:
                fp.write(img_name + '\n')


if __name__ == '__main__':
    main()
