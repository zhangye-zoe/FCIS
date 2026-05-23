import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage import morphology, measure
from skimage.morphology import remove_small_objects

from ..utils import (angle_to_vector, calculate_centerpoint, calculate_gradient, vector_to_label)
from ...models.utils import generate_direction_differential_map


class DirectionLabelMake(object):
    """build direction label & point label for any dataset."""

    def __init__(self, to_center=True, num_angles=8):
        self.to_center = to_center
        self.num_angles = num_angles

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
        """generate boundary label & direction from instance map and pure semantic map.

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

        # point map calculation & gradient map calculation
        point_map, gradient_map, dist_map = self.calculate_point_map(inst_gt, to_center=self.to_center)

        # direction map calculation
        dir_map = self.calculate_dir_map(inst_gt, gradient_map, self.num_angles)
        reg_dir_map = self.calculate_regression_dir_map(inst_gt, gradient_map)
        if self.num_angles == 8:
            weight_map = self.calculate_weight_map(dir_map, dist_map, self.num_angles)
        else:
            weight_map = np.zeros_like(dir_map)

        data['dist_gt'] = dist_map
        data['point_gt'] = point_map
        data['dir_gt'] = dir_map
        data['reg_dir_gt'] = reg_dir_map
        data['loss_weight_map'] = weight_map

        return data

    @classmethod
    def calculate_weight_map(self, dir_map, dist_map, num_angle_types):
        # torch style api
        dd_map = generate_direction_differential_map(dir_map, num_angle_types + 1)
        dd_map = dd_map[0].numpy()
        weight_map = dd_map * (10 - dist_map)
        weight_map = morphology.dilation(weight_map, selem=morphology.selem.disk(1))
        weight_map = (weight_map.astype(np.float32) * 2 + 1.0)

        return weight_map

    @classmethod
    def calculate_dir_map(self, instance_map, gradient_map, num_angle_types):
        # Prepare for gradient map & direction map calculation
        # continue angle calculation    instance_map.shape=(256, 256), gradient_map.shape=(256, 256, 2)
        angle_map = np.degrees(np.arctan2(gradient_map[:, :, 0], gradient_map[:, :, 1]))
        angle_map[instance_map == 0] = 0
        vector_map = angle_to_vector(angle_map, num_angle_types)
        # angle type judgement
        dir_map = vector_to_label(vector_map, num_angle_types)
        dir_map[instance_map == 0] = -1
        dir_map = dir_map + 1

        return dir_map

    @classmethod
    def calculate_regression_dir_map(self, instance_map, gradient_map):
        angle_map = np.degrees(np.arctan2(gradient_map[:, :, 0], gradient_map[:, :, 1]))
        angle_map[angle_map < 0] += 360
        angle_map[instance_map == 0] = 0

        return angle_map / 180 * np.pi

    @classmethod
    def calculate_point_map(self, instance_map, to_center=True):
        H, W = instance_map.shape[:2]
        # distance_center_map: The min distance between center and point
        distance_to_center_map = np.zeros((H, W), dtype=np.float32)
        gradient_map = np.zeros((H, W, 2), dtype=np.float32)
        point_map = np.zeros((H, W), dtype=np.float32)

        # remove background
        markers_unique = list(np.unique(instance_map))
        markers_len = len(markers_unique) - 1

        # Calculate for each instance
        for k in markers_unique:
            if k == 0:
                continue
            single_instance_map = (instance_map == k).astype(np.uint8)

            center = calculate_centerpoint(single_instance_map, H, W)
            # Count each center to judge if some instances don't get center
            assert single_instance_map[center[0], center[1]] > 0
            point_map[center[0], center[1]] = 1

            # Calculate distance from points of instance to instance center.
            if to_center:
                distance_to_center_instance = self.calculate_distance_to_center(single_instance_map, center)
                distance_to_center_map += distance_to_center_instance
            else:
                distance_to_center_instance = self.calculate_distance_to_centralridge(single_instance_map)
                distance_to_center_map += distance_to_center_instance

            # Calculate gradient of (to center) distance
            gradient_map_instance = self.calculate_gradient(single_instance_map, distance_to_center_instance)
            gradient_map[(single_instance_map != 0), :] = 0
            gradient_map += gradient_map_instance
        assert int(point_map.sum()) == markers_len

        # Use gaussian filter to process center point map
        point_map_gaussian = gaussian_filter(point_map * 255, sigma=2, order=0).astype(np.float32)
        distance_to_center_map = ((distance_to_center_map)**0.5) * 10
        return point_map_gaussian, gradient_map, distance_to_center_map

    @classmethod
    def calculate_distance_to_center(self, single_instance_map, center):
        H, W = single_instance_map.shape[:2]
        # Calculate distance (to center) map for single instance
        point_map_instance = np.zeros((H, W), dtype=np.uint8)
        point_map_instance[center[0], center[1]] = 1
        distance_to_center = distance_transform_edt(1 - point_map_instance)
        # Only calculate distance (to center) in distance region
        distance_to_center = distance_to_center * single_instance_map
        distance_to_center_instance = (1 - distance_to_center /
                                       (distance_to_center.max() + 0.0000001)) * single_instance_map

        return distance_to_center_instance

    @classmethod
    def calculate_distance_to_centralridge(self, single_instance_map):
        H, W = single_instance_map.shape[:2]
        # Calculate distance (to center) map for single instance
        distance_to_centralridge = distance_transform_edt(single_instance_map)
        # Only calculate distance (to center) in distance region
        distance_to_centralridge = distance_to_centralridge * single_instance_map
        distance_to_centralridge_instance = (distance_to_centralridge /
                                             (distance_to_centralridge.max() + 0.0000001)) * single_instance_map

        return distance_to_centralridge_instance

    @classmethod
    def calculate_gradient(self, single_instance_map, distance_to_center_instance):
        H, W = single_instance_map.shape[:2]
        gradient_map_instance = np.zeros((H, W, 2))
        gradient_map_instance = calculate_gradient(distance_to_center_instance, ksize=11)
        gradient_map_instance[(single_instance_map == 0), :] = 0
        return gradient_map_instance
