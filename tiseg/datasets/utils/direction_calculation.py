import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

from .center_calculation import calculate_centerpoint
from .gradient_calculation import calculate_gradient

label_to_vector_mapping = {
    4: [[-1, -1], [-1, 1], [1, 1], [1, -1]],
    5: [[0, 0], [-1, -1], [-1, 1], [1, 1], [1, -1]],
    8: [[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]],
    9: [[0, 0], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]],
    16: [[0, -2], [-1, -2], [-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2], [2, 1],
         [2, 0], [2, -1], [2, -2], [1, -2]],
    17: [[0, 0], [0, -2], [-1, -2], [-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2],
         [2, 1], [2, 0], [2, -1], [2, -2], [1, -2]],
    32: [
        [0, -4],
        [-1, -4],
        [-2, -4],
        [-3, -4],
        [-4, -4],
        [-4, -3],
        [-4, -2],
        [-4, -1],
        [-4, 0],
        [-4, 1],
        [-4, 2],
        [-4, 3],
        [-4, 4],
        [-3, 4],
        [-2, 4],
        [-1, 4],
        [0, 4],
        [1, 4],
        [2, 4],
        [3, 4],
        [4, 4],
        [4, 3],
        [4, 2],
        [4, 1],
        [4, 0],
        [4, -1],
        [4, -2],
        [4, -3],
        [4, -4],
        [3, -4],
        [2, -4],
        [1, -4],
    ]
}


# TODO: Add doc string and comments
def align_angle(angle_map, num_classes=8):

    assert isinstance(angle_map, np.ndarray)

    step = 360 / num_classes
    new_angle_map = np.zeros(angle_map.shape, dtype=np.float)
    angle_index_map = np.zeros(angle_map.shape, dtype=np.int)
    mask = (angle_map <= (-180 + step / 2)) | (angle_map > (180 - step / 2))
    new_angle_map[mask] = -180
    angle_index_map[mask] = 0

    for i in range(1, num_classes):
        middle = -180 + step * i
        mask = (angle_map > (middle - step / 2)) & (angle_map <= (middle + step / 2))
        new_angle_map[mask] = middle
        angle_index_map[mask] = i

    return new_angle_map, angle_index_map


def angle_to_vector(angle_map, num_classes=8):

    assert isinstance(angle_map, np.ndarray)

    vector_map = np.zeros((*angle_map.shape, 2), dtype=np.float)
    deg2rad = np.deg2rad

    if num_classes is not None:
        angle_map, _ = align_angle(angle_map, num_classes=num_classes)

    angle_map = deg2rad(angle_map)

    vector_map[..., 0] = np.sin(angle_map)
    vector_map[..., 1] = np.cos(angle_map)

    return vector_map


def angle_to_direction_label(angle_map, seg_label_map=None, num_classes=8, extra_ignore_mask=None):

    assert isinstance(angle_map, np.ndarray)
    assert isinstance(seg_label_map, np.ndarray) or seg_label_map is None

    _, label_map = align_angle(angle_map, num_classes=num_classes)

    if seg_label_map is None:
        ignore_mask = np.zeros(angle_map.shape, dtype=np.bool)
    else:
        ignore_mask = seg_label_map == -1

    if extra_ignore_mask is not None:
        ignore_mask = ignore_mask | extra_ignore_mask
    label_map[ignore_mask] = -1

    return label_map


def vector_to_label(vector_map, num_classes=8):

    assert isinstance(vector_map, np.ndarray)

    rad2deg = np.rad2deg

    angle_map = np.arctan2(vector_map[..., 0], vector_map[..., 1])
    angle_map = rad2deg(angle_map)

    return angle_to_direction_label(angle_map, num_classes=num_classes)


def label_to_vector(dir_map, num_classes=8):

    assert isinstance(dir_map, np.ndarray)

    mapping = label_to_vector_mapping[num_classes]
    offset_h = np.zeros_like(dir_map)
    offset_w = np.zeros_like(dir_map)

    for idx, (hdir, wdir) in enumerate(mapping):
        mask = dir_map == idx
        offset_h[mask] = hdir
        offset_w[mask] = wdir

    # vertical, horizontal direction concat
    vector_map = np.stack([offset_h, offset_w], axis=-1)
    # NHWC -> NCHW
    vector_map = vector_map.transpose(0, 3, 1, 2)

    return vector_map


def calculate_dir_map(instance_map, gradient_map, num_angle_types):
    # Prepare for gradient map & direction map calculation
    # continue angle calculation
    angle_map = np.degrees(np.arctan2(gradient_map[:, :, 0], gradient_map[:, :, 1]))
    angle_map[instance_map == 0] = 0
    vector_map = angle_to_vector(angle_map, num_angle_types)
    # angle type judgement
    dir_map = vector_to_label(vector_map, num_angle_types)

    dir_map[instance_map == 0] = -1
    dir_map = dir_map + 1

    return dir_map


def calculate_distance_to_center(single_instance_map, center):
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


def _calculate_gradient(single_instance_map, distance_to_center_instance):
    H, W = single_instance_map.shape[:2]
    gradient_map_instance = np.zeros((H, W, 2))
    gradient_map_instance = calculate_gradient(distance_to_center_instance, ksize=11)
    gradient_map_instance[(single_instance_map == 0), :] = 0
    return gradient_map_instance


def get_dir_from_inst(inst_map, num_angle_types):
    """Calculate direction classification map from instance map."""
    H, W = inst_map.shape[:2]

    gradient_map = np.zeros((H, W, 2), dtype=np.float32)

    # remove background
    markers_unique = list(np.unique(inst_map))

    for k in markers_unique:
        if k == 0:
            continue
        single_instance_map = (inst_map == k).astype(np.uint8)

        center = calculate_centerpoint(single_instance_map, H, W)
        # Count each center to judge if some instances don't get center
        assert single_instance_map[center[0], center[1]] > 0

        # Calculate distance from points of instance to instance center.
        distance_to_center_instance = calculate_distance_to_center(single_instance_map, center)

        # Calculate gradient of (to center) distance
        gradient_map_instance = _calculate_gradient(single_instance_map, distance_to_center_instance)
        gradient_map[(single_instance_map != 0), :] = 0
        gradient_map += gradient_map_instance

    dir_map = calculate_dir_map(inst_map, gradient_map, num_angle_types)

    return dir_map
