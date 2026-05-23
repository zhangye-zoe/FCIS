import numpy as np
from skimage import morphology, measure
from skimage.morphology import remove_small_objects


class BoundLabelMake:
    """Generate high quality boundary labels.

    The return is fixed to a three-class map (background, foreground, boundary).
    """

    def __init__(self, edge_id=2, selem_radius=3):
        self.edge_id = edge_id
        if isinstance(selem_radius, int):
            selem_radius = (selem_radius, selem_radius)
        self.radius = selem_radius

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
        """generate boundary label from instance map and pure semantic map.

        sem_map:
            0: background
            1: semantic_class 1
            2: semantic class 2
            ...

        inst_map:
            0: background
            1: instance 1
            2: instance 2
            ...

        sem_map_w_bound:
            0: background
            1: foreground
            2: boundary

        Args:
            sem_map: two-class or multi-class semantic map without edge which is
                the raw semantic map.
            inst_map: instance map with each instance id. Use inst_map = inst_id
                to extrach each instance.
        """

        sem_gt = data['sem_gt']
        inst_gt = data['inst_gt']
        inst_gt = self._fix_inst(inst_gt)
        sem_gt[inst_gt == 0] = 0
        data['sem_gt'] = sem_gt

        sem_gt_w_bound = np.zeros_like(sem_gt)
        sem_gt_w_bound += sem_gt

        # NOTE: sem_map must have same size as inst_map
        assert np.allclose(sem_gt > 0, inst_gt > 0)
        inst_id_list = list(np.unique(inst_gt))
        for inst_id in inst_id_list:
            if inst_id == 0:
                continue
            inst_id_mask = inst_gt == inst_id
            bound = morphology.dilation(
                inst_id_mask, selem=morphology.selem.diamond(self.radius[0])) & (
                    ~morphology.erosion(inst_id_mask, selem=morphology.selem.diamond(self.radius[1])))
            # bound = inst_id_mask & ~morphology.erosion(inst_id_mask, selem=morphology.selem.diamond(self.radius))
            sem_gt_w_bound[bound > 0] = self.edge_id

        # NOTE: sem_map is raw semantic map (two-class or multi-class without boundary)
        # NOTE: sem_map_w_bound has an extra semantic class (background, class1, class2, ..., bound)
        data['sem_gt_w_bound'] = sem_gt_w_bound
        data['seg_fields'].append('sem_gt_w_bound')

        return data
