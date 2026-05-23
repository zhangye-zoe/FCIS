import torch

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


def label_to_vector(dir_map, num_classes=8):

    if not isinstance(dir_map, torch.Tensor):
        dir_map = torch.tensor(dir_map[None, ...])

    assert isinstance(dir_map, torch.Tensor)

    mapping = label_to_vector_mapping[num_classes]
    offset_h = torch.zeros_like(dir_map)
    offset_w = torch.zeros_like(dir_map)

    for idx, (hdir, wdir) in enumerate(mapping):
        mask = dir_map == idx
        offset_h[mask] = hdir
        offset_w[mask] = wdir

    # vertical, horizontal direction concat
    vector_map = torch.stack([offset_h, offset_w], dim=-1)
    # NHWC -> NCHW
    vector_map = vector_map.permute(0, 3, 1, 2).to(dir_map.device)

    return vector_map, dir_map


def circshift(matrix, shift_vertical, shift_horizontal):
    """Cyclic shift of matrix.

    direction:
    1. Upper left;
    2. Upper right;
    3. Lower left;
    4. Lower right;

    Args:
        matrix (torch.Tensor): The matrix to cyclic shift.
        direction (int): The direction selection argument.
        shift_vertical (int): The shift distance of vertical direction.
        shift_horizontal (int): The shift distance of horizontal direction.
    """
    # upper left
    moved_matrix = torch.roll(matrix, shifts=(shift_vertical, shift_horizontal), dims=(-2, -1))

    return moved_matrix


# TODO: regularize variable name and add doc string
def generate_direction_differential_map(dir_map, direction_classes=9, background=None, use_reg=False):
    # label_to_vector requires NxHxW (torch.Tensor) or HxW (numpy.ndarry)
    if use_reg:
        vector_map = torch.from_numpy(dir_map).cuda()
        vector_map = vector_map.permute(2, 0, 1)[None, ...]
        background = torch.from_numpy(background).cuda()[None, ...]
    else:
        vector_map, dir_map = label_to_vector(dir_map, direction_classes)
        background = dir_map == 0
    # Only support batch size == 1
    # Nx2xHxW (2: vertical and horizontal)
    vector_anchor = vector_map.float()

    N, _, H, W = vector_anchor.shape
    # Cosine Similarity Map
    cos_sim_map = torch.zeros((N, H, W), dtype=torch.float32, device=vector_map.device)

    feature_list = []
    # Only support 8 direction now
    if direction_classes - 1 == 8:
        # Lower
        lower = circshift(vector_anchor, 1, 0)
        # Lower Right
        lower_right = circshift(vector_anchor, 1, 1)
        # Right
        right = circshift(vector_anchor, 0, 1)
        # Upper Right
        upper_right = circshift(vector_anchor, -1, 1)
        # Upper
        upper = circshift(vector_anchor, -1, 0)
        # Upper Left
        upper_left = circshift(vector_anchor, -1, -1)
        # Left
        left = circshift(vector_anchor, 0, -1)
        # Lower Left
        lower_left = circshift(vector_anchor, 1, -1)

        feature_list.append(lower)
        feature_list.append(lower_right)
        feature_list.append(right)
        feature_list.append(upper_right)
        feature_list.append(upper)
        feature_list.append(upper_left)
        feature_list.append(left)
        feature_list.append(lower_left)

    cos_sim_map_single_direction = torch.zeros((N, direction_classes - 1, H, W),
                                               dtype=torch.float32,
                                               device=vector_map.device)
    for k, feature_item in enumerate(feature_list):
        numerator = (
            vector_anchor[:, 0, :, :] * feature_item[:, 0, :, :] + vector_anchor[:, 1, :, :] * feature_item[:, 1, :, :])
        denominator = (
            torch.sqrt(pow(vector_anchor[:, 0, :, :], 2) + pow(vector_anchor[:, 1, :, :], 2)) *
            torch.sqrt(pow(feature_item[:, 0, :, :], 2) + pow(feature_item[:, 1, :, :], 2)) + 0.000001)
        cos_sim_map_single_direction[:, k, :, :] = numerator / denominator

    cos_sim_map, cos_sim_indices = torch.min(cos_sim_map_single_direction, dim=1)

    cos_sim_map[background] = 1

    cos_sim_map = (1 - torch.round(cos_sim_map))
    cos_sim_map_max = torch.max(cos_sim_map)
    cos_sim_map_min = torch.min(cos_sim_map)

    # when dir_map is zero map, the direction differential map is also
    # zero map.
    if cos_sim_map_max == 0:
        return cos_sim_map

    cos_sim_map_normal = (cos_sim_map - cos_sim_map_min) / (cos_sim_map_max - cos_sim_map_min)

    return cos_sim_map_normal
