from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


def format_img(img, dc=True):
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    if img.dtype is not np.float32:
        img = img.astype(np.float32)

    img = to_tensor(img)
    if dc:
        img = DC(img, stack=True)

    return img


def format_seg(seg, dc=True):
    if len(seg.shape) < 3:
        seg = seg[None, ...]
    # segmentation gt convert to long

    seg = to_tensor(seg.astype(np.int64))
    if dc:
        seg = DC(seg, stack=True)

    return seg


def format_reg(reg, dc=True):
    if len(reg.shape) < 3:
        reg = reg[None, ...]
    # regression gt convert to float
    reg = to_tensor(reg.astype(np.float32))
    if dc:
        reg = DC(reg, stack=True)

    return reg


def format_(map_, dc=True):
    map_ = to_tensor(map_)

    if dc:
        map_ = DC(map_, stack=True)

    return map_


def format_info(info, dc=True):
    if dc:
        info = DC(info, cpu_only=True)

    return info

def format_adj(adj, dc=True):
    # adj = to_tensor(adj)
    if dc:
        adj = DC(adj, cpu_only=True)
        
    return adj

class Formatting(object):

    def __init__(self, data_keys, label_keys):
        self.data_keys = data_keys
        self.label_keys = label_keys

    def __call__(self, data):
        ret = {'data': {}, 'label': {}, 'metas': {}}
        # print("data", data.keys())
        # print("=" * 100)
        data_info = data.pop('data_info')
        # print("data info", data_info["data_id"])
        # print("+" * 300)
        _ = data.pop('seg_fields')

        for data_key in self.data_keys:
            if data_key == 'img':
                h, w = data[data_key].shape[:2]
                data_info['input_hw'] = (h, w)
                ret['data'][data_key] = format_img(data[data_key])
            else:
                ret['data'][data_key] = format_(data[data_key])

        for label_key in self.label_keys:
            if label_key in ['dist_gt', 'point_gt', 'hv_gt', 'loss_weight_map', 'dis_gt']:
                ret['label'][label_key] = format_reg(data[label_key])
            
            elif label_key in ['adj_gt']:
                ret['label'][label_key] = format_adj(data['adj_gt'])

            else:
                # import matplotlib.pyplot as plt
                # print("data info", data_info["data_id"])
                # print("+" * 300)
                # if label_key == "inst_gt":
                #     plt.imshow(data[label_key])
                #     plt.show()
                #     plt.savefig("z_before.png")


                ret['label'][label_key] = format_seg(data[label_key])

                # if label_key == "inst_gt":
                #     seg2 = data[label_key]
                #     if len(seg2) < 3:
                #         seg2 = seg2[None, ...]
                #     # segmentation gt convert to long

                #     seg2 = to_tensor(seg2.astype(np.int64))
                                    

                #     plt.imshow(seg2)
                #     plt.show()
                #     plt.savefig("z_after.png")

        ret['metas'] = format_info(data_info)

        return ret


class TorchFormatting(object):

    def __init__(self, data_keys, label_keys):
        self.data_keys = data_keys
        self.label_keys = label_keys

    def __call__(self, data):
        ret = {'data': {}, 'label': {}, 'metas': {}}
        data_info = data.pop('data_info')
        _ = data.pop('seg_fields')

        for data_key in self.data_keys:
            if data_key == 'img':
                h, w = data[data_key].shape[:2]
                data_info[0]['input_hw'] = (h, w)
                ret['data'][data_key] = format_img(data[data_key], False)
            else:
                ret['data'][data_key] = format_(data[data_key], False)

        for label_key in self.label_keys:
            if label_key in ['dist_gt', 'point_gt', 'hv_gt', 'loss_weight_map']:
                ret['label'][label_key] = format_reg(data[label_key], False)
            else:
                ret['label'][label_key] = format_seg(data[label_key], False)

        ret['metas'] = format_info(data_info, False)

        return ret
