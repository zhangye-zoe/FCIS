import random

import mmcv
import numpy as np
import torch
from PIL import Image


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:
        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


def tensor2maps(tensor):
    """Convert tensor to ground truth instance map.

    Args:
        tensor (Tensor): Tensor that contains multiple ground truth instance
            maps, shape (N, C, H, W).

    Returns:
        list[np.ndarray]: A list that contains multiple ground truth instance
            maps.
    """
    if torch is None:
        raise RuntimeError('pytorch is not installed')
    assert torch.is_tensor(tensor) and tensor.ndim == 4

    num_maps = tensor.size(0)
    maps = []
    for map_id in range(num_maps):
        map = tensor[map_id, ...].cpu().numpy()[0, :, :].astype(np.uint8)
        maps.append(np.ascontiguousarray(map))
    return maps


def pillow_save(array, save_path=None, palette=None):
    """save array to a image by using pillow package.

    Args:
        array (np.ndarry): The numpy array which is need to save as an image.
        save_path (str, optional): The save path of numpy array image.
            Default: None
        palette (np.ndarry, optional): The palette for save image.
    """
    image = Image.fromarray(array.astype(np.uint8))

    if palette is not None:
        image = image.convert('P')
        image.putpalette(palette)

    if save_path is not None:
        image.save(save_path)

    return image


def blend_image(image, mask, save_path=None, mask_palette=None, alpha=0.5):
    """blend input image & mask with alpha value."""
    if isinstance(image, str):
        image = np.array(Image.open(image))
    if isinstance(mask, str):
        mask = np.array(Image.open(mask))

    mask_id_list = list(np.unique(mask))
    if mask_palette is None:
        mask_palette = [(0, 0, 0)]
        for id in mask_id_list:
            if id == 0:
                continue
            color = [int(random.random() * 255) for _ in range(3)]
            mask_palette.append(color)

    mask_canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for id, color in zip(mask_id_list, mask_palette):
        mask_canvas[mask == id, :] = color

    image_canvas = np.zeros_like(image, dtype=np.uint8)
    for i in range(3):
        image_canvas[:, :, i] = alpha * image[:, :, i] + (1 - alpha) * mask_canvas[:, :, i]

    if save_path is not None:
        pillow_save(image_canvas, save_path=save_path)

    return image_canvas


def image_addition(arrays, save_path, palette=None):
    """composition of several arrays."""
    if mmcv.is_list_of(arrays, str):
        arrays = [np.array(Image.open(array)) for array in arrays]

    arrays = sum(arrays)

    pillow_save(arrays, save_path=save_path, palette=palette)

    return arrays


def get_bounding_box(img):
    """Get the bounding box coordinates of a binary input- assumes a single object.

    Args:
        img: input binary image.

    Returns:
        bounding box coordinates

    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]
