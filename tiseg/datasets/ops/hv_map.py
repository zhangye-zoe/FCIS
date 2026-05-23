import numpy as np
from scipy.ndimage import measurements


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def gen_instance_hv_map(inst_gt):
    """Input annotation must be of original shape.

    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.
    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.
    """

    x_map = np.zeros(inst_gt.shape[:2], dtype=np.float32)
    y_map = np.zeros(inst_gt.shape[:2], dtype=np.float32)

    h, w = inst_gt.shape[:2]

    inst_ids = list(np.unique(inst_gt))
    for inst_id in inst_ids:
        if inst_id == 0:
            continue
        inst_map = np.array(inst_gt == inst_id, np.uint8)
        inst_box = get_bounding_box(inst_map)

        # expand the box by 2px
        # Because we first pad the ann at line 207, the bboxes
        # will remain valid after expansion
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

        # instance center of mass, rounded to nearest pixel
        inst_com = list(measurements.center_of_mass(inst_map))

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        ####
        x_map_box = x_map[inst_box[0]:inst_box[1], inst_box[2]:inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0]:inst_box[1], inst_box[2]:inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = np.dstack([x_map, y_map])
    return hv_map


class HVLabelMake(object):
    """build direction label & point label for any dataset."""

    def __call__(self, data):
        inst_gt = data['inst_gt']

        # [H, W, 2]
        hv_gt = gen_instance_hv_map(inst_gt)
        # [2, H, W]
        hv_gt = hv_gt.transpose(2, 0, 1)

        data['hv_gt'] = hv_gt
        data['seg_fields'].append('hv_gt')

        return data
