import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import morphology, measure
from skimage.morphology import remove_small_objects


class UNetLabelMake(object):
    """
    Input annotation must be of original shape.

    Perform following operation:
        1) Remove the 1px of boundary of each instance
           to create separation between touching instances
        2) Generate the weight map from the result of 1)
           according to the unet paper equation (2).

    The weight map generation follw this equation:
        `w(x) = w_{c}(x) + w_{0} \\cdot exp(- \\frac{(d_{1}(x) + d_{2}(x))^{2}}{2\\sigma^{2}})`
    where:
        w_{c} denotes the weight map to balance the class frequencies.
        d_{1} denotes the distance to the border of the nearest cell.
        d_{2} denotes the distance to the border of the second nearest cell.

    Args:
        wc (dict)        : Dictionary of weight classes.
        w0 (int/float)   : Border weight parameter.
        sigma (int/float): Border width parameter.
    """

    def __init__(self, wc=None, w0=10.0, sigma=5.0):
        super(UNetLabelMake, self).__init__()
        self.wc = wc
        self.w0 = w0
        self.sigma = sigma

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
            remapped_ids[remapped_ids > 0] = inst_id
            new_inst_gt[remapped_ids > 0] = remapped_ids[remapped_ids > 0]
            # cur += len(np.unique(remapped_ids[remapped_ids > 0]))

        return new_inst_gt
    
    def _remove_1px_boundary(self, inst_gt):
        new_gt = np.zeros(inst_gt.shape[:2], np.int32)
        inst_ids = list(np.unique(inst_gt))

        for inst_id in inst_ids:
            if inst_id == 0:
                continue
            inst_map = (inst_gt == inst_id).astype(np.uint8)
            inst_map = morphology.erosion(inst_map, morphology.selem.diamond(1))
            new_gt[inst_map > 0] = inst_id
        return new_gt

    def _get_weight_map(self, ann, inst_list):
        if len(inst_list) <= 1:  # 1 instance only
            return np.zeros(ann.shape[:2])
        stacked_inst_bgd_dst = np.zeros(ann.shape[:2] + (len(inst_list), ))

        for idx, inst_id in enumerate(inst_list):
            inst_bgd_map = np.array(ann != inst_id, np.uint8)
            inst_bgd_dst = distance_transform_edt(inst_bgd_map)
            stacked_inst_bgd_dst[..., idx] = inst_bgd_dst

        near1_dst = np.amin(stacked_inst_bgd_dst, axis=2)
        near2_dst = np.expand_dims(near1_dst, axis=2)
        near2_dst = stacked_inst_bgd_dst - near2_dst
        near2_dst[near2_dst == 0] = np.PINF  # very large
        near2_dst = np.amin(near2_dst, axis=2)
        near2_dst[ann > 0] = 0  # the instances

        near2_dst = near2_dst + near1_dst
        # to fix pixel where near1 == near2
        near2_eve = np.expand_dims(near1_dst, axis=2)
        # to avoide the warning of a / 0
        near2_eve = (1.0 + stacked_inst_bgd_dst) / (1.0 + near2_eve)
        near2_eve[near2_eve != 1] = 0
        near2_eve = np.sum(near2_eve, axis=2)
        near2_dst[near2_eve > 1] = near1_dst[near2_eve > 1]

        pix_dst = near1_dst + near2_dst
        pen_map = pix_dst / self.sigma
        pen_map = self.w0 * np.exp(-pen_map**2 / 2)
        pen_map[ann > 0] = 0  # inner instances zero
        return pen_map

    def __call__(self, data):
        # print("data", data.keys())
        # print("=" * 100)
        inst_gt = data['inst_gt']
        sem_gt = data['sem_gt']
        # adj_gt = data['adj_gt']
        # print("sem gt", np.unique(sem_gt))
        inst_gt = self._fix_inst(inst_gt)
        sem_gt[inst_gt == 0] = 0
        data['sem_gt'] = sem_gt

        # import matplotlib.pyplot as plt
        # plt.imshow(inst_gt)
        # plt.show()
        # plt.savefig("z_gt.png")

        # setting 1 boundary pix of each instance to background
        inst_gt = self._remove_1px_boundary(inst_gt)
        sem_gt_inner = sem_gt.copy()
        sem_gt_inner[inst_gt == 0] = 0

        # cant do the shortcut because near2 also needs instances
        # outside of cropped portion
        inst_ids = np.unique(inst_gt)
        inst_ids = list(inst_ids[inst_ids > 0])
        wmap = self._get_weight_map(inst_gt, inst_ids)

        if self.wc is None:
            wmap += 1  # uniform weight for all classes
        else:
            class_weights = np.zeros_like(inst_gt.shape[:2])
            for class_id, class_w in self.wc.items():
                class_weights[inst_gt == class_id] = class_w
            wmap += class_weights

        # class_weights = np.zeros_like(sem_gt)
        # # print("sem gt", sem_gt.shape)
        # # print("class w", class_weights.shape)
        # for class_id, class_w in self.wc.items():
        #     # print("class id", class_id)
        #     # print("class_w", class_w)
        #     class_weights[sem_gt == class_id] = class_w

        # print("class weight", np.unique(class_weights))
        # wmap += class_weights
        

        # data['loss_weight_map'] = class_weights
        data['loss_weight_map'] = wmap
        data['sem_gt_inner'] = sem_gt_inner
        data['inst_gt'] = inst_gt
        data['seg_fields'].append('sem_gt_inner')
        # data['adj_gt'] = adj_gt


        return data
