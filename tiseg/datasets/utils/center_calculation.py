import math

from numba import jit


# TODO: Refactor
@jit(nopython=True)
def calculate_centerpoint(instance_mask, H, W):
    """Calculate Center Point by using centerness (FCOS) for single instance.

    Args:
        instance_mask (np.ndarry): The binary mask which contains a single
            instance.
        H (int): The height of mask.
        W (int): The width of mask.
    """
    max_centerness = -1
    x = -1
    y = -1
    Directions = []
    # Set eight search directions
    for i in range(8):
        Directions.append((math.sin(2 * math.pi / 8 * i), math.cos(2 * math.pi / 8 * i)))

    # traversal points of instance and calculate centerness of each instance
    # point. Then selecting out the point with max centerness
    for i in range(H):
        for j in range(W):
            if instance_mask[i][j] > 0:
                max_distance = 0
                min_distance = 10000000
                for k in range(8):
                    smaller_bound = 0
                    larger_bound = 1000000
                    # binary search for min distance between point and
                    # background when searching along direction K.
                    while abs(smaller_bound - larger_bound) > 0.1:
                        mid = (smaller_bound + larger_bound) / 2
                        x_offset = round(i + Directions[k][0] * mid)
                        y_offset = round(j + Directions[k][1] * mid)
                        if (x_offset >= 0 and y_offset < W and y_offset >= 0 and x_offset < H
                                and instance_mask[x_offset][y_offset] > 0):
                            smaller_bound = mid
                        else:
                            larger_bound = mid
                    max_distance = max(max_distance, larger_bound)
                    min_distance = min(min_distance, smaller_bound)
                assert (max_distance > 0 and min_distance > 0)
                centerness = min_distance / max_distance
                if centerness > max_centerness:
                    max_centerness = centerness
                    x = i
                    y = j
    return [int(x), int(y)]
