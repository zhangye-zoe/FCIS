import copy
import os.path as osp

import cv2
import numpy as np
from PIL import Image

from .ops import class_dict
import yaml


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


class DatasetMapper(object):

    def __init__(self, test_mode, *, processes):
        self.test_mode = test_mode

        self.processes = []
        for process in processes:
            class_name = process.pop('type')
            pipeline = class_dict[class_name](**process)
            self.processes.append(pipeline)

    def __call__(self, data_info):
        data_info = copy.deepcopy(data_info)

        img = read_image(data_info['file_name'])
        sem_gt = read_image(data_info['sem_file_name'])
        # print("file name", data_info['sem_file_name'])
        # print(np.unique(sem_gt))


        inst_gt = read_image(data_info['inst_file_name'])
        # import matplotlib.pyplot as plt
        # plt.imshow(inst_gt)
        # plt.show()
        # plt.savefig("z_gt1.png")
        with open(data_info['adj_file_name'], "r") as file:
            adj_gt = yaml.load(file, Loader=yaml.FullLoader)

        # dis_gt = read_image(data_info['dis_file_name'])
        # print("adj_gt", adj_gt)
        # print("=" * 200)

        data_info['ori_hw'] = img.shape[:2]

        h, w = img.shape[:2]
        assert img.shape[:2] == sem_gt.shape[:2]

        data = {
            'img': img,
            'sem_gt': sem_gt,
            'inst_gt': inst_gt,
            'adj_gt': adj_gt,
            # 'dis_gt': dis_gt,
            'seg_fields': ['sem_gt', 'inst_gt'],
            'data_info': data_info
        }
        # print("data process", self.processes)
        # print("=" * 200)
        for process in self.processes:
            data = process(data)


        return data
