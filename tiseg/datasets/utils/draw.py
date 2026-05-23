import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


def colorize_seg_map(seg_map, palette=None):
    """using random rgb color to colorize segmentation map."""
    # print("seg map", seg_map.shape, np.unique(seg_map))
    # print("palette", palette)
    colorful_seg_map = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    id_list = list(np.unique(seg_map))

    if palette is None:
        palette = {}
        for id in id_list:
            color = [random.random() * 255 for i in range(3)]
            palette[id] = color
    

    for id in id_list:
        # ignore background
        if id == 0:
            continue
        colorful_seg_map[seg_map == id, :] = palette[id]

    return colorful_seg_map


class Drawer(object):

    def __init__(self, save_folder=None, edge_id=2, sem_palette=None):
        self.save_folder = save_folder
        self.edge_id = edge_id
        self.sem_palette = sem_palette

    def draw(self, img_name, img_file_name, pred, gt, metrics):
        sem_pred = pred['sem_pred']
        inst_pred = pred['inst_pred']
        tc_sem_pred = pred['tc_sem_pred']

        sem_gt = gt['sem_gt']
        inst_gt = gt['inst_gt']
        tc_sem_gt = gt['tc_sem_gt']

        plt.figure(figsize=(5 * 4, 5 * 2 + 3))

        # image drawing
        img = cv2.imread(img_file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(241)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Image', fontsize=15, color='black')

        canvas = np.zeros((*sem_pred.shape, 3), dtype=np.uint8)
        canvas[(sem_pred > 0) * (sem_gt > 0), :] = (0, 0, 255)
        canvas[canvas == self.edge_id] = 0
        canvas[(sem_pred == 0) * (sem_gt > 0), :] = (0, 255, 0)
        canvas[(sem_pred > 0) * (sem_gt == 0), :] = (255, 0, 0)
        plt.subplot(242)
        plt.imshow(canvas)
        plt.axis('off')
        plt.title('Error Analysis: FN-FP-TP', fontsize=15, color='black')

        # get the colors of the values, according to the
        # colormap used by imshow
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        label_list = [
            'TP',
            'FN',
            'FP',
        ]
        for color, label in zip(colors, label_list):
            color = list(color)
            color = [x / 255 for x in color]
            plt.plot(0, 0, '-', color=tuple(color), label=label)
        plt.legend(loc='upper center', fontsize=9, bbox_to_anchor=(0.5, 0), ncol=3)

        plt.subplot(243)
        plt.imshow(colorize_seg_map(inst_pred))
        plt.axis('off')
        plt.title('Instance Level Prediction')

        plt.subplot(244)
        plt.imshow(colorize_seg_map(inst_gt))
        plt.axis('off')
        plt.title('Instance Level Ground Truth')

        plt.subplot(245)
        plt.imshow(colorize_seg_map(sem_pred, self.sem_palette))
        plt.axis('off')
        plt.title('Semantic Level Prediction')

        plt.subplot(246)
        plt.imshow(colorize_seg_map(sem_gt, self.sem_palette))
        plt.axis('off')
        plt.title('Semantic Level Ground Truth')

        tc_palette = [(0, 0, 0), (0, 255, 0), (255, 0, 0)]

        

        plt.subplot(247)
        plt.imshow(colorize_seg_map(tc_sem_pred, tc_palette))
        plt.axis('off')
        plt.title('Three-class Semantic Level Prediction')

        plt.subplot(248)
        plt.imshow(colorize_seg_map(tc_sem_gt, tc_palette))
        plt.axis('off')
        plt.title('Three-class Semantic Level Ground Truth')

        plt.suptitle(''.join([f'{k}:{v[1]} ' for k, v in metrics.items()]))

        # results visulization
        plt.tight_layout()
        plt.savefig(f'{self.save_folder}/{img_name}_compare.png', dpi=300)

    def draw_direction(self, img_name, img_file_name, pred, gt, metrics):
        sem_pred = pred['sem_pred']
        inst_pred = pred['inst_pred']
        tc_sem_pred = pred['tc_sem_pred']
        dir_pred = pred['dir_pred']
        ddm_pred = pred['ddm_pred']

        sem_gt = gt['sem_gt']
        inst_gt = gt['inst_gt']
        tc_sem_gt = gt['tc_sem_gt']
        dir_gt = gt['dir_gt']
        ddm_gt = gt['ddm_gt']

        plt.figure(figsize=(5 * 4, 5 * 3 + 3))

        # image drawing
        img = cv2.imread(img_file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(341)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Image', fontsize=15, color='black')

        canvas = np.zeros((*sem_pred.shape, 3), dtype=np.uint8)
        canvas[(sem_pred > 0) * (sem_gt > 0), :] = (0, 0, 255)
        canvas[canvas == self.edge_id] = 0
        canvas[(sem_pred == 0) * (sem_gt > 0), :] = (0, 255, 0)
        canvas[(sem_pred > 0) * (sem_gt == 0), :] = (255, 0, 0)
        plt.subplot(342)
        plt.imshow(canvas)
        plt.axis('off')
        plt.title('Error Analysis: FN-FP-TP', fontsize=15, color='black')

        # get the colors of the values, according to the
        # colormap used by imshow
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        label_list = [
            'TP',
            'FN',
            'FP',
        ]
        for color, label in zip(colors, label_list):
            color = list(color)
            color = [x / 255 for x in color]
            plt.plot(0, 0, '-', color=tuple(color), label=label)
        plt.legend(loc='upper center', fontsize=9, bbox_to_anchor=(0.5, 0), ncol=3)

        plt.subplot(343)
        plt.imshow(colorize_seg_map(inst_pred))
        plt.axis('off')
        plt.title('Instance Level Prediction')

        plt.subplot(344)
        plt.imshow(colorize_seg_map(inst_gt))
        plt.axis('off')
        plt.title('Instance Level Ground Truth')

        plt.subplot(345)
        plt.imshow(colorize_seg_map(sem_pred, self.sem_palette))
        plt.axis('off')
        plt.title('Semantic Level Prediction')

        plt.subplot(346)
        plt.imshow(colorize_seg_map(sem_gt, self.sem_palette))
        plt.axis('off')
        plt.title('Semantic Level Ground Truth')

        tc_palette = [(0, 0, 0), (0, 255, 0), (255, 0, 0)]

        plt.subplot(347)
        plt.imshow(colorize_seg_map(tc_sem_pred, tc_palette))
        plt.axis('off')
        plt.title('Three-class Semantic Level Prediction')

        plt.subplot(348)
        plt.imshow(colorize_seg_map(tc_sem_gt, tc_palette))
        plt.axis('off')
        plt.title('Three-class Semantic Level Ground Truth')

        dcm_palette = [[0, 0, 0], [255, 0, 0], [255, 165, 0], [255, 255, 0], [0, 255, 0], [0, 127, 255], [0, 0, 255],
                       [139, 0, 255], [255, 255, 255]]

        plt.subplot(349)
        plt.imshow(colorize_seg_map(dir_pred, palette=dcm_palette))
        plt.axis('off')
        plt.title('Direction Classification Map Prediction')

        plt.subplot(3, 4, 10)
        plt.imshow(colorize_seg_map(dir_gt, palette=dcm_palette))
        plt.axis('off')
        plt.title('Direction Classification Map Ground Truth')

        plt.subplot(3, 4, 11)
        plt.imshow(ddm_pred, cmap='gray')
        plt.axis('off')
        plt.title('Direction Differential Map Prediction')

        plt.subplot(3, 4, 12)
        plt.imshow(ddm_gt, cmap='gray')
        plt.axis('off')
        plt.title('Direction Differential Map Ground Truth')

        # results visulization
        plt.tight_layout()
        plt.savefig(f'{self.save_folder}/{img_name}_dir_compare.png', dpi=300)
