import os
import os.path as osp
import argparse
import random

import cv2
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit


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


def parse_args():
    parser = argparse.ArgumentParser('Convert monuseg dataset.')
    parser.add_argument('dataset_root', help='dataset root path.')

    return parser.parse_args()


def analysis_func(labels):
    cell_area = []
    cell_count = 0
    slice_cell_area = []
    slice_count = 0
    class_idx = 1
    for j in range(labels.shape[0]):
        inst = labels[j, :, :, 0]
        sem = labels[j, :, :, 1]
        # calculate cell number of single slice.
        inst[sem != class_idx] = 0
        cell_ids = np.unique(inst)
        cell_count += len(cell_ids) - 1
        # calculate single cell average area.
        for cell_id in cell_ids:
            if cell_id == 0:
                continue
            cell_area.append(np.sum(inst == cell_id))
        # calculate cell average area in a slice.
        sum_val = np.sum(sem == class_idx)
        if sum_val > 0:
            slice_count += 1
            slice_cell_area.append(sum_val)

    print('single average cell area:', sum(cell_area) / len(cell_area))
    print('single average slice cell area:', sum(slice_cell_area) / len(slice_cell_area))
    print('total number of cell:', cell_count)
    print('valid slice of cell:', slice_count)


def main(args):
    import joblib
    data_root = args.dataset_root  # Change this according to the root path where the data is located # noqa
    images_path = f'{data_root}/conic/images.npy'  # images array Nx256x256x3
    labels_path = f'{data_root}/conic/labels.npy'  # labels array Nx256x256x2

    FOLD_IDX = 0
    splits = joblib.load(f'{data_root}/splits.dat')
    val_indices = splits[FOLD_IDX]['valid']
    train_indices = splits[FOLD_IDX]['train']

    # counts_path = f"{data_root}/conic/counts.csv"  # csv of counts per nuclear type for each patch # noqa
    # info_path = f"{data_root}/conic/patch_info.csv"  # csv indicating which image from Lizard each patch comes from # noqa

    images = np.load(images_path)
    labels = np.load(labels_path)

    print('Images Shape:', images.shape)
    print('Labels Shape:', labels.shape)

    for split, indices in [('train', train_indices), ('val', val_indices)]:
        new_root = osp.join(data_root, split)

        if not osp.exists(new_root):
            os.makedirs(new_root, 0o775)

        for i in tqdm(indices):
            pillow_save(osp.join(new_root, f'{i}.png'), images[i])
            np.save(osp.join(new_root, f'{i}_inst.npy'), labels[i, :, :, 0])
            pillow_save(osp.join(new_root, f'{i}_inst_color.png'), colorize_seg_map(labels[i, :, :, 0]))
            palette = np.zeros((7, 3), dtype=np.uint8)
            palette[0, :] = (0, 0, 0)
            palette[1, :] = (255, 0, 0)
            palette[2, :] = (0, 255, 0)
            palette[3, :] = (0, 0, 255)
            palette[4, :] = (255, 255, 0)
            palette[5, :] = (255, 0, 255)
            palette[6, :] = (0, 255, 255)
            pillow_save(osp.join(new_root, f'{i}_sem.png'), labels[i, :, :, 1], palette=palette)

        # item_list = [x.rstrip('_inst.npy') for x in os.listdir(new_root) if '_inst.npy' in x]
        item_list = indices

        with open(osp.join(data_root, f'{split}.txt'), 'w') as fp:
            [fp.write(str(item) + '\n') for item in item_list]


def generate_split(args):
    SEED = 5
    info = pd.read_csv(f'{args.dataset_root}/conic/patch_info.csv')
    file_names = np.squeeze(info.to_numpy()).tolist()

    img_sources = [v.split('-')[0] for v in file_names]
    img_sources = np.unique(img_sources)

    cohort_sources = [v.split('_')[0] for v in img_sources]
    _, cohort_sources = np.unique(cohort_sources, return_inverse=True)
    num_trials = 10
    splitter = StratifiedShuffleSplit(n_splits=num_trials, train_size=0.8, test_size=0.2, random_state=SEED)

    splits = []
    split_generator = splitter.split(img_sources, cohort_sources)
    for train_indices, valid_indices in split_generator:
        train_cohorts = img_sources[train_indices]
        valid_cohorts = img_sources[valid_indices]
        assert np.intersect1d(train_cohorts, valid_cohorts).size == 0
        train_names = [
            file_name for file_name in file_names for source in train_cohorts if source == file_name.split('-')[0]
        ]
        valid_names = [
            file_name for file_name in file_names for source in valid_cohorts if source == file_name.split('-')[0]
        ]
        train_names = np.unique(train_names)
        valid_names = np.unique(valid_names)
        print(f'Train: {len(train_names):04d} - Valid: {len(valid_names):04d}')
        assert np.intersect1d(train_names, valid_names).size == 0
        train_indices = [file_names.index(v) for v in train_names]
        valid_indices = [file_names.index(v) for v in valid_names]
        splits.append({'train': train_indices, 'valid': valid_indices})
    joblib.dump(splits, f'{args.dataset_root}/splits.dat')


if __name__ == '__main__':
    args = parse_args()
    generate_split(args)
    main(args)
