import numpy as np
import torch
import torch.nn.functional as F


# TODO: Add comments and doc string.
class Sobel:

    _caches = {}
    ksize = 11

    @staticmethod
    def _generate_sobel_kernel(shape, axis):
        """shape must be odd: eg.

        (5,5) axis is the direction, with 0 to positive x and 1 to positive y
        """
        k = np.zeros(shape, dtype=np.float32)
        p = [(j, i) for j in range(shape[0]) for i in range(shape[1])
             if not (i == (shape[1] - 1) / 2.0 and j == (shape[0] - 1) / 2.0)]

        for j, i in p:
            j_ = int(j - (shape[0] - 1) / 2.0)
            i_ = int(i - (shape[1] - 1) / 2.0)
            k[j, i] = (i_ if axis == 0 else j_) / float(i_ * i_ + j_ * j_)
        return torch.from_numpy(k).unsqueeze(0)

    @classmethod
    def kernel(cls, ksize=None):
        if ksize is None:
            ksize = cls.ksize
        if ksize in cls._caches:
            return cls._caches[ksize]

        sobel_x, sobel_y = (cls._generate_sobel_kernel((ksize, ksize), i) for i in (0, 1))
        sobel_ker = torch.cat([sobel_y, sobel_x], dim=0).view(2, 1, ksize, ksize)
        cls._caches[ksize] = sobel_ker
        return sobel_ker


def calculate_gradient(input_map, ksize=11):
    """Calculate dx & dy Graidient for single channel map."""
    sobel_kernel = Sobel.kernel(ksize=ksize)
    assert len(input_map.shape) == 2
    # format input image
    input_map = torch.from_numpy(input_map).float().reshape(1, 1, *input_map.shape)
    gradient = F.conv2d(input_map, sobel_kernel, padding=ksize // 2)
    # unformat
    gradient = gradient.squeeze().permute(1, 2, 0).numpy()
    return gradient
