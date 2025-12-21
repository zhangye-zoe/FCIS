from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from mmcv.runner import BaseModule

from tiseg.utils import resize
from ..losses import mdice, tdice


def _parse_losses(losses):
    """Parse the raw outputs (losses) of the network.

    Args:
        losses (dict): Raw output of the network, which usually contain
            losses and other necessary information.

    Returns:
        tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
            which may be a weighted sum of all losses, log_vars contains
            all the variables to be sent to the logger.
    """
    log_vars = OrderedDict()
    extra_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        elif isinstance(loss_value, dict):
            extra_vars[loss_name] = loss_value
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars, extra_vars


class BaseSegmentor(BaseModule, metaclass=ABCMeta):
    """Segmentor supports multiplt data & multiple labels.

    For example (CDNet):
        Data: image;
        Label: sem_gt, dir_gt, point_gt;
    """

    @abstractmethod
    def calculate(self, img):
        """Calculate the semantic logit result."""
        pass

    @abstractmethod
    def forward(self, **kwargs):
        """When training, the module is required to return loss dict. When
        evaluation, the module is required to return instance-level or
        semantic-level results."""
        pass

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)
        loss, log_vars, extra_vars = _parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, extra_vars=extra_vars, num_samples=len(data_batch['metas']))

        return outputs

    def val_step(self, data_batch, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        output = self(**data_batch, **kwargs)
        return output

    # def split_axes(self, window_size, overlap_size, height, width):
    #     """Calculate patch coordinates of split inference."""
    #     ws = window_size
    #     os = overlap_size

    #     i_axes = [0]
    #     j_axes = [0]
    #     cur = 0
    #     edge_base = ws - os // 2
    #     middle_base = ws - os
    #     while True:
    #         if cur == 0:
    #             cur += edge_base
    #         else:
    #             cur += middle_base

    #         if i_axes[-1] < height:
    #             i_axes.append(cur)
    #         if j_axes[-1] < width:
    #             j_axes.append(cur)

    #         if cur + edge_base == height:
    #             i_axes.append(cur + edge_base)
    #         if cur + edge_base == width:
    #             j_axes.append(cur + edge_base)
    #         print(cur, i_axes[-1], j_axes[-1], height, width)
    #         if cur > max(height, width):
    #             exit(0)
    #         if i_axes[-1] == height and j_axes[-1] == width:
    #             break

    #     return i_axes, j_axes

    # def split_inference(self, img, meta, rescale):
    #     """Split inference: split img into several patches for network inference. Then merge them with a overlap.

    #     How does this function split the img? For example:
    #         img (torch.Tensor): (1, 3, 1000, 1000) -> H = 1000, W = 1000
    #         crop_size (int): 256
    #         overlap_size (int): 80

    #     H dimension:
    #         patch 0: (0, 216)
    #         patch 1: (216, 392)  overlap 0-1: (176, 256)
    #         patch 2: (392, 568)  overlap 1-2: (352, 432)
    #         patch 3: (568, 744)  overlap 2-3: (528, 608)
    #         patch 4: (744, 920)  overlap 3-4: (704, 784)
    #         patch 5: (920, 1136) overlap 4-5: (880, 960)
    #     W dimension is same.
    #     """
    #     ws = self.test_cfg.crop_size[0]
    #     os = self.test_cfg.overlap_size[0]

    #     B, C, H, W = img.shape

    #     # zero pad for border patches
    #     pad_h = 0
    #     pad_w = 0
    #     if H - ws > 0:
    #         temp_h = H - 2 * (ws - os // 2)
    #         if temp_h <= 0:
    #             pad_h = temp_h * -1
    #         else:
    #             pad_h = (ws - os) - (temp_h) % (ws - os)
    #     else:
    #         pad_h = ws - H
    #     if W - ws > 0:
    #         temp_w = W - 2 * (ws - os // 2)
    #         if temp_w <= 0:
    #             pad_w = temp_w * -1
    #         else:
    #             pad_w = (ws - os) - (temp_w) % (ws - os)
    #     else:
    #         pad_w = ws - W

    #     H1 = pad_h + H
    #     W1 = pad_w + W
    #     img_canvas = torch.zeros((B, C, H1, W1), dtype=img.dtype, device=img.device)
    #     img_canvas.fill_(0)
    #     img_canvas[:, :, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W] = img

    #     _, _, H1, W1 = img_canvas.shape
    #     sem_output = torch.zeros((B, self.num_classes, H1, W1))
    #     i_axes, j_axes = self.split_axes(ws, os, H1, W1)

    #     for i in range(len(i_axes) - 1):
    #         for j in range(len(j_axes) - 1):
    #             r_patch_s = i_axes[i] if i == 0 else i_axes[i] - os // 2
    #             r_patch_e = r_patch_s + ws
    #             c_patch_s = j_axes[j] if j == 0 else j_axes[j] - os // 2
    #             c_patch_e = c_patch_s + ws
    #             img_patch = img_canvas[:, :, r_patch_s:r_patch_e, c_patch_s:c_patch_e]
    #             sem_patch = self.calculate(img_patch)

    #             # patch overlap remove
    #             r_valid_s = i_axes[i] - r_patch_s
    #             r_valid_e = i_axes[i + 1] - r_patch_s
    #             c_valid_s = j_axes[j] - c_patch_s
    #             c_valid_e = j_axes[j + 1] - c_patch_s
    #             sem_patch = sem_patch[:, :, r_valid_s:r_valid_e, c_valid_s:c_valid_e]
    #             sem_output[:, :, i_axes[i]:i_axes[i + 1], j_axes[j]:j_axes[j + 1]] = sem_patch

    #     sem_output = sem_output[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
    #     if rescale:
    #         sem_output = resize(sem_output, size=meta['ori_hw'], mode='bilinear', align_corners=False)
    #     return sem_output

    # NOTE: slide_inference isn't practical for special seg task.
    # def slide_inference(self, img, meta, rescale):
    #     """Inference by sliding-window with overlap.

    #     If h_crop > h_img or w_crop > w_img, the small patch will be used to
    #     decode without padding.
    #     """
    #     h_stride = w_stride = self.test_cfg.crop_size[0] - self.test_cfg.overlap_size[0]
    #     h_crop, w_crop = self.test_cfg.crop_size
    #     batch_size, _, h_img, w_img = img.size()
    #     num_classes = self.num_classes
    #     h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    #     w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    #     preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
    #     count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
    #     for h_idx in range(h_grids):
    #         for w_idx in range(w_grids):
    #             y1 = h_idx * h_stride
    #             x1 = w_idx * w_stride
    #             y2 = min(y1 + h_crop, h_img)
    #             x2 = min(x1 + w_crop, w_img)
    #             y1 = max(y2 - h_crop, 0)
    #             x1 = max(x2 - w_crop, 0)
    #             crop_img = img[:, :, y1:y2, x1:x2]
    #             crop_seg_logit = self.calculate(crop_img)
    #             preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

    #             count_mat[:, :, y1:y2, x1:x2] += 1
    #     assert (count_mat == 0).sum() == 0
    #     preds = preds / count_mat
    #     if rescale:
    #         preds = resize(preds, size=meta['ori_hw'], mode='bilinear', align_corners=False)
    #     return preds

    # NOTE: old style split inference
    def split_inference(self, img, meta, rescale, encoding=None):
        """using half-and-half strategy to slide inference."""
        window_size = self.test_cfg.crop_size[0]
        overlap_size = self.test_cfg.overlap_size[0]

        B, C, H, W = img.shape

        # zero pad for border patches
        pad_h = 0
        if H - window_size > 0:
            pad_h = (window_size - overlap_size) - (H - window_size) % (window_size - overlap_size)
        else:
            pad_h = window_size - H

        if W - window_size > 0:
            pad_w = (window_size - overlap_size) - (W - window_size) % (window_size - overlap_size)
        else:
            pad_w = window_size - W

        H1, W1 = pad_h + H, pad_w + W
        img_canvas = torch.zeros((B, C, H1, W1), dtype=img.dtype, device=img.device)
        img_canvas[:, :, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W] = img

        sem_logit = torch.zeros((B, 16, H1, W1), dtype=img.dtype, device=img.device)
        cls_logit = torch.zeros((B, self.num_classes, H1, W1), dtype=img.dtype, device=img.device)

        # sem_logit = torch.zeros((B, 16, H1, W1), dtype=img.dtype, device=img.device)
        # cls_logit = torch.zeros((B, 2, H1, W1), dtype=img.dtype, device=img.device)

        # sem_feas = torch.zeros((B, 16, H1, W1), dtype=img.dtype, device=img.device)
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (i + window_size - overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                img_patch = img_canvas[:, :, i:r_end, j:c_end]
                sem_pred, cls_pred = self.calculate(img_patch, encoding)

                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (j + window_size - overlap_size // 2 if j + window_size < W1 else W1)
                sem_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = sem_pred[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]
                cls_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = cls_pred[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]
                # sem_feas[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = sem_faeture[:, :, ind1_s - i:ind1_e - i,
                #                                                           ind2_s - j:ind2_e - j]

        sem_logit = sem_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        cls_logit = cls_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        
        return sem_logit, cls_logit

        # return None, cls_logit

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        sem_logit = self.calculate(img)

        return sem_logit

    def inference(self, img, meta, rescale, encoding=None):
        """Inference with split/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            meta (dict): Image info dict.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['split', 'whole']

        self.rotate_degrees = self.test_cfg.get('rotate_degrees', [0])
        self.flip_directions = self.test_cfg.get('flip_directions', ['none'])
        sem_logit_list = []
        cls_logit_list = []
        img_ = img
        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                img = self.tta_transform(img_, rotate_degree, flip_direction)

                # inference patch or whole img
                if self.test_cfg.mode == 'split':
                    # if encoding is not None:
                    # print('image', img)
                    # print('=' * 100)
                    sem_logit, cls_logit = self.split_inference(img, meta, rescale, encoding)
                else:
                    sem_logit = self.whole_inference(img, meta, rescale)
                    
                # return None

                sem_logit = self.reverse_tta_transform(sem_logit, rotate_degree, flip_direction)
                cls_logit = self.reverse_tta_transform(cls_logit, rotate_degree, flip_direction)

                sem_logit = F.softmax(sem_logit, dim=1)
                cls_logit = F.softmax(cls_logit, dim=1)
                sem_logit_list.append(sem_logit)
                cls_logit_list.append(cls_logit)
                

        sem_logit = sum(sem_logit_list) / len(sem_logit_list)
        cls_logit = sum(cls_logit_list) / len(cls_logit_list)

        if rescale:
            # sem_feature = resize(sem_feature, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            sem_logit = resize(sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            cls_logit = resize(cls_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)


        return sem_logit, cls_logit

    @classmethod
    def tta_transform(self, img, rotate_degree, flip_direction):
        """TTA transform function.

        Support transform:
            rotation: 0, 90, 180, 270
            flip: horizontal, vertical, diagonal
        """
        rotate_num = (rotate_degree // 90) % 4
        img = torch.rot90(img, k=rotate_num, dims=(-2, -1))

        if flip_direction == 'horizontal':
            img = torch.flip(img, dims=[-1])
        if flip_direction == 'vertical':
            img = torch.flip(img, dims=[-2])
        if flip_direction == 'diagonal':
            img = torch.flip(img, dims=[-2, -1])

        return img

    @classmethod
    def reverse_tta_transform(self, img, rotate_degree, flip_direction):
        """reverse TTA transform function.

        Support transform:
            rotation: 0, 90, 180, 270
            flip: horizontal, vertical, diagonal
        """
        rotate_num = 4 - (rotate_degree // 90) % 4
        if flip_direction == 'horizontal':
            img = torch.flip(img, dims=[-1])
        if flip_direction == 'vertical':
            img = torch.flip(img, dims=[-2])
        if flip_direction == 'diagonal':
            img = torch.flip(img, dims=[-2, -1])

        img = torch.rot90(img, k=rotate_num, dims=(-2, -1))

        return img

    def _training_metric(self, sem_logit, sem_label):
        """metric calculation when training."""
        wrap_dict = {}

        # loss
        clean_sem_logit = sem_logit.clone().detach()
        clean_sem_label = sem_label.clone().detach()

        # print("clean sem logit", clean_sem_logit.shape)
        # print("clean sem label", clean_sem_label.shape)

        wrap_dict['sem_tdice'] = tdice(clean_sem_logit, clean_sem_label, self.num_classes)
        wrap_dict['sem_mdice'] = mdice(clean_sem_logit, clean_sem_label, self.num_classes)

        # NOTE: training aji calculation metric calculate (This will be deprecated.)
        # (the edge id is set `self.num_classes - 1` in default)
        # sem_pred = torch.argmax(sem_logit, dim=1).cpu().numpy().astype(np.uint8)
        # sem_pred[sem_pred == (self.num_classes - 1)] = 0
        # sem_target = sem_label.cpu().numpy().astype(np.uint8)
        # sem_target[sem_target == (self.num_classes - 1)] = 0

        # N = sem_pred.shape[0]
        # wrap_dict['aji'] = 0.
        # for i in range(N):
        #     aji_single_image = aggregated_jaccard_index(sem_pred[i], sem_target[i])
        #     wrap_dict['aji'] += 100.0 * torch.tensor(aji_single_image)
        # # distributed environment requires cuda tensor
        # wrap_dict['aji'] /= N
        # wrap_dict['aji'] = wrap_dict['aji'].cuda()

        return wrap_dict
