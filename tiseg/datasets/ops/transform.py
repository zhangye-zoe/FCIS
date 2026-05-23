import random

import albumentations as A
import cv2
import mmcv
import numpy as np


class ColorJitter(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(0, 2):
            return self.convert(img, beta=random.uniform(-self.brightness_delta, self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(0, 2):
            return self.convert(img, alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(0, 2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1], alpha=random.uniform(self.saturation_lower, self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(0, 2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 0] = (img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, data):
        img = data['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(0, 2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        data['img'] = img
        return data


class AlbuColorJitter(object):

    def __init__(self, brightness, contrast, saturation, hue, prob=0.5):
        self.trans = A.ColorJitter(brightness, contrast, saturation, hue, prob=prob)

    def __call__(self, data):
        img = data['img']
        res_dict = self.trans(image=img)

        data['img'] = res_dict['image']
        return data


class Resize(object):

    def __init__(self, min_size=None, max_size=None, scale_factor=None, resize_mode='fix'):
        self.min_size = min_size
        self.max_size = max_size
        self.scale_factor = scale_factor
        self.resize_mode = resize_mode

    def _img_resize(self, img):
        if self.resize_mode == 'fix':
            img = cv2.resize(img, (self.min_size, self.min_size))
        elif self.resize_mode == 'ratio':
            h, w = img.shape[:2]
            min_len = min(h, w)
            max_len = max(h, w)
            scale_f = self.min_size / min_len
            if scale_f * max_len > self.max_size:
                scale_f = self.max_size / max_len
            scale_h, scale_w = h * scale_f, w * scale_f
            img = cv2.resize(img, (scale_h, scale_w))
        elif self.resize_mode == 'scale':
            h, w = img.shape[:2]
            scale_h, scale_w = self.scale_factor * h, self.scale_factor * w
            img = cv2.resize(img, (scale_h, scale_w))

        return img

    def _seg_resize(self, seg):
        if self.resize_mode == 'fix':
            seg = cv2.resize(seg, (self.min_size, self.min_size), interpolation=cv2.INTER_NEAREST)
        elif self.resize_mode == 'ratio':
            h, w = seg.shape[:2]
            min_len = min(h, w)
            max_len = max(h, w)
            scale_f = self.min_size / min_len
            if scale_f * max_len > self.max_size:
                scale_f = self.max_size / max_len
            scale_h, scale_w = h * scale_f, w * scale_f
            seg = cv2.resize(seg, (scale_h, scale_w), interpolation=cv2.INTER_NEAREST)
        elif self.resize_mode == 'scale':
            h, w = seg.shape[:2]
            scale_h, scale_w = self.scale_factor * h, self.scale_factor * w
            seg = cv2.resize(seg, (scale_h, scale_w), interpolation=cv2.INTER_NEAREST)

        return seg

    def __call__(self, data):
        seg_fields = data['seg_fields']
        data['img'] = self._img_resize(data['img'])
        for seg_key in seg_fields:
            data[seg_key] = self._seg_resize(data[seg_key])

        return data


class CenterCrop(object):

    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size

    def __call__(self, data):
        img = data['img']
        seg_fields = data['seg_fields']

        h, w = img.shape[:2]
        ch, cw = self.crop_size[0], self.crop_size[1]

        diff_h = (h - ch) // 2
        diff_w = (w - cw) // 2

        data['img'] = img[diff_h:diff_h + ch, diff_w:diff_w + cw]
        for seg_key in seg_fields:
            data[seg_key] = data[seg_key][diff_h:diff_h + ch, diff_w:diff_w + cw]

        return data


class RandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, list[str], optional): The flipping direction. Options
            are 'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, prob=None, direction='horizontal'):
        if prob is not None:
            assert prob >= 0 and prob <= 1
        else:
            prob = 0
        self.prob = prob

        if isinstance(direction, list):
            assert sum([x in ['horizontal', 'vertical', 'diagonal'] for x in direction]) == len(direction)
        else:
            assert direction in ['horizontal', 'vertical', 'diagonal']
            direction = [direction]
        self.direction = direction

    def __call__(self, data):
        img = data['img']
        seg_fields = data['seg_fields']
        flip = True if np.random.rand() < self.prob else False
        # random select from direction list.
        select_index = np.random.randint(0, len(self.direction))
        flip_direction = self.direction[select_index]
        if flip:
            # flip image
            img = mmcv.imflip(img, direction=flip_direction)

            # flip segs
            for seg_key in seg_fields:
                data[seg_key] = mmcv.imflip(data[seg_key], direction=flip_direction)
        data['img'] = img

        return data


class RandomRotate(object):
    """Rotate the image & seg.

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 0.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self, prob, degree, pad_val=0, seg_pad_val=0, center=None, auto_bound=False):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    def __call__(self, data):
        img = data['img']
        seg_fields = data['seg_fields']
        rotate = True if np.random.rand() < self.prob else False
        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        if rotate:
            # rotate image
            img = mmcv.imrotate(
                img, angle=degree, border_value=self.pad_val, center=self.center, auto_bound=self.auto_bound)

            # rotate segs
            for seg_key in seg_fields:
                data[seg_key] = mmcv.imrotate(
                    data[seg_key],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')
        data['img'] = img

        return data


# TODO: Add doc string for this transform
class RandomSparseRotate(object):

    def __init__(self, degree_list=[90, 180, 270], prob=0.5, pad_val=0, seg_pad_val=0, center=None, auto_bound=False):
        self.degree_list = degree_list
        self.prob = prob
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    def __call__(self, data):
        img = data['img']
        seg_fields = data['seg_fields']
        rotate = True if np.random.rand() < self.prob else False
        # random select from degree list.
        select_index = np.random.randint(0, len(self.degree_list))
        degree = self.degree_list[select_index]

        if rotate:
            # rotate image
            img = mmcv.imrotate(
                img, angle=degree, border_value=self.pad_val, center=self.center, auto_bound=self.auto_bound)

            # rotate segs
            for seg_key in seg_fields:
                data[seg_key] = mmcv.imrotate(
                    data[seg_key],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')
        data['img'] = img

        return data


# TODO: Add doc string for this transform
class RandomElasticDeform(object):

    def __init__(
        self,
        prob=0.5,
        alpha=1,
        sigma=50,
        alpha_affine=50,
    ):
        self.trans = A.ElasticTransform(
            p=prob,
            alpha=alpha,
            sigma=sigma,
            alpha_affine=alpha_affine,
            interpolation=0,
            border_mode=0,
            value=(0, 0, 0))

    def __call__(self, data):
        img = data['img']
        seg_fields = data['seg_fields']
        segs = [data[seg_key] for seg_key in seg_fields]

        res_dict = self.trans(image=img, masks=segs)

        data['img'] = res_dict['image']
        segs = res_dict['masks']
        for seg_key, seg in zip(seg_fields, segs):
            data[seg_key] = seg
        return data


class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1.):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, data):
        img = data['img']
        seg_fields = data['seg_fields']
        segs = [data[seg_key] for seg_key in seg_fields]

        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(segs[0], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop semantic seg
        new_segs = []
        for seg in segs:
            seg = self.crop(seg, crop_bbox)
            new_segs.append(seg)
        segs = new_segs

        data['img'] = img
        for seg_key, seg in zip(seg_fields, segs):
            data[seg_key] = seg

        return data


class AlbuImgWarpper(object):
    """The image process warpper for albumentation library"""

    def __init__(self, albu_trans):
        self.trans = albu_trans

    def __call__(self, img):
        res = self.trans(image=img)

        return res['image']


class Identity(object):
    """The placeholder of transform."""

    def __call__(self, *args):
        if len(args) == 1:
            return args[0]
        else:
            return args


class Affine(object):

    def __init__(self, scale=(0.8, 1.2), shear=5, rotate_degree=[-180, 180], translate_frac=(0, 0.01)):
        self.trans = A.Affine(scale=scale, shear=shear, rotate=rotate_degree, translate_percent=translate_frac)

    def __call__(self, data):
        img = data['img']
        seg_fields = data['seg_fields']
        segs = [data[seg_key] for seg_key in seg_fields]

        res_dict = self.trans(image=img, masks=segs)

        img = res_dict['image']
        segs = res_dict['masks']

        data['img'] = img
        for seg_key, seg in zip(seg_fields, segs):
            data[seg_key] = seg

        return data


class RandomBlur(object):
    """Random use filter to blur image.

    Filters:
        1. BLUR (模糊滤波);
        2. GaussianBlur (高斯滤波);
        3. MedianBlur (中值滤波);
    """

    def __init__(self, prob=0.5):
        self.prob = prob

        self.blur = AlbuImgWarpper(A.Blur())
        self.gauss = AlbuImgWarpper(A.GaussianBlur())
        self.median = AlbuImgWarpper(A.MedianBlur())

        self.trans = [self.blur, self.gauss, self.median]

    def __call__(self, data):
        img = data['img']
        seg_fields = data['seg_fields']
        segs = [data[seg_key] for seg_key in seg_fields]

        if np.random.rand() < self.prob:
            index = random.randint(0, len(self.trans) - 1)
            img = self.trans[index](img)

        data['img'] = img
        for seg_key, seg in zip(seg_fields, segs):
            data[seg_key] = seg

        return data


class Normalize(object):
    """z-score standardization or simple div 255."""

    def __init__(self, mean=None, std=None, if_zscore=False):
        self.mean = mean
        self.std = std
        self.if_zscore = if_zscore

    def __call__(self, data):
        img = data['img']
        seg_fields = data['seg_fields']
        segs = [data[seg_key] for seg_key in seg_fields]

        img = img / 255.
        if self.if_zscore:
            img = (img - self.mean) / (self.std)

        data['img'] = img
        for seg_key, seg in zip(seg_fields, segs):
            data[seg_key] = seg

        return data


class Pad(object):
    """Pad image."""

    def __init__(self, pad_size):

        if isinstance(pad_size, int):
            self.pad_size = (pad_size, pad_size)
        else:
            self.pad_size = pad_size

    def __call__(self, data):
        img = data['img']
        seg_fields = data['seg_fields']
        segs = [data[seg_key] for seg_key in seg_fields]

        h, w = img.shape[:2]

        pad_h = max(self.pad_size[0], h) - h
        pad_w = max(self.pad_size[1], w) - w

        img_canvas = np.zeros((h + pad_h, w + pad_w, 3), dtype=img.dtype)
        img_canvas[pad_h // 2:pad_h // 2 + h, pad_w // 2:pad_w // 2 + w, :] = img
        img = img_canvas

        new_segs = []
        for seg in segs:
            seg_canvas = np.zeros((h + pad_h, w + pad_w), dtype=seg.dtype)
            seg_canvas[pad_h // 2:pad_h // 2 + h, pad_w // 2:pad_w // 2 + w] = seg
            new_segs.append(seg_canvas)
        segs = new_segs

        data['img'] = img
        for seg_key, seg in zip(seg_fields, segs):
            data[seg_key] = seg

        return data
