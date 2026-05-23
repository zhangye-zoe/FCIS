"""
Modified from vqdang code at https://github.com/vqdang/hover_net/blob/tensorflow-final/src/loader/augs.py
"""

import numpy as np
from skimage import measure
from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import distance_transform_cdt


def get_bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


class DistanceLabelMake(object):
    """
    Input annotation must be of original shape.

    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
        1. Obtain the standard distance map of nuclear pixels to their closest
    boundary.
        2. Can be interpreted as the inverse distance map of nuclear pixels to
    the centroid.
    """

    def __init__(self, inst_norm=True):
        self.inst_norm = inst_norm

    def _fix_inst(self, inst_gt):
        cur = 0
        new_inst_gt = np.zeros_like(inst_gt)
        inst_id_list = list(np.unique(inst_gt))
        for inst_id in inst_id_list:
            if inst_id == 0:
                continue
            inst_map = inst_gt == inst_id
            inst_map = remove_small_objects(inst_map, 5)
            inst_map = np.array(inst_map, np.uint8)
            remapped_ids = measure.label(inst_map)
            remapped_ids[remapped_ids > 0] += cur
            new_inst_gt[remapped_ids > 0] = remapped_ids[remapped_ids > 0]
            cur += len(np.unique(remapped_ids[remapped_ids > 0]))

        return new_inst_gt

    def __call__(self, data):
        sem_gt = data['sem_gt']
        inst_gt = data['inst_gt']
        inst_gt = self._fix_inst(inst_gt)
        sem_gt[inst_gt == 0] = 0
        data['sem_gt'] = sem_gt

        dist_gt = np.zeros(inst_gt.shape, dtype=np.float32)

        h, w = inst_gt.shape[:2]

        inst_id_list = list(np.unique(inst_gt))
        for inst_id in inst_id_list:
            if inst_id == 0:
                continue
            inst_map = (inst_gt == inst_id).astype(np.uint8)
            inst_box = get_bounding_box(inst_map)

            # expand the box by 2px
            inst_box[0] -= 2
            inst_box[2] -= 2
            inst_box[1] += 2
            inst_box[3] += 2

            inst_box[0] = max(inst_box[0], 0)
            inst_box[2] = max(inst_box[2], 0)
            inst_box[1] = min(inst_box[1], h)
            inst_box[3] = min(inst_box[3], w)

            inst_map = inst_map[inst_box[0]:inst_box[1], inst_box[2]:inst_box[3]]

            if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
                continue

            # chessboard distance map generation
            # normalize distance to 0-1
            inst_dst = distance_transform_cdt(inst_map)
            inst_dst = inst_dst.astype('float32')
            if self.inst_norm:
                max_value = np.amax(inst_dst)
                if max_value <= 0:
                    continue  # HACK: temporay patch for divide 0 i.e no nuclei (how?)
                inst_dst = (inst_dst / np.amax(inst_dst))

            dst_map_box = dist_gt[inst_box[0]:inst_box[1], inst_box[2]:inst_box[3]]
            dst_map_box[inst_map > 0] = inst_dst[inst_map > 0]

        data['dist_gt'] = dist_gt
        data['seg_fields'].append('dist_gt')

        return data


# class DistanceLabelMake(object):
#     """build direction label & point label for any dataset."""

#     def __init__(self, num_classes):
#         self.num_classes = num_classes

#     def dist_wo_norm(self, bin_image):
#         res = np.zeros_like(bin_image)
#         for j in range(1, bin_image.max() + 1):
#             one_cell = np.zeros_like(bin_image)
#             one_cell[bin_image == j] = 1
#             one_cell = distance_transform_cdt(one_cell)
#             res[bin_image == j] = one_cell[bin_image == j]
#         res = res.astype('uint8')
#         return res

#     def __call__(self, sem_gt, inst_gt):
#         results = {}
#         inst_ids_pre_class = assign_sem_class_to_insts(inst_gt, sem_gt, self.num_classes)
#         # reduce background dimension
#         dist_canvas = np.zeros((self.num_classes - 1, *inst_gt.shape), dtype=np.float32)
#         for sem_id, inst_ids in inst_ids_pre_class.items():
#             if sem_id == 0:
#                 continue
#             for inst_id in inst_ids:
#                 dist = self.dist_wo_norm((inst_gt == inst_id).astype(np.uint8))
#                 dist = dist / np.max(dist)
#                 dist_canvas[sem_id - 1, inst_gt == inst_id] = 0
#                 dist_canvas[sem_id - 1] += dist

#         results['dist_gt'] = dist_canvas

#         return results
