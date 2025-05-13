import torch
import torch.nn as nn
import numpy as np
from skimage import morphology, measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes

from ..backbones import TorchVGG16BN
from ..builder import SEGMENTORS
from ..heads import UNetHead
from ..losses import BatchMultiClassDiceLoss
from .base import BaseSegmentor
import torch.nn.functional as F

@SEGMENTORS.register_module()
class UNet(BaseSegmentor):
    """Implementation of `U-Net: Convolutional Networks for Biomedical Image Segmentation`"""

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(UNet, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes

        self.backbone = TorchVGG16BN(in_channels=3, pretrained=True, out_indices=[0, 1, 2, 3, 4, 5])
        self.head = UNetHead(
            num_classes=self.num_classes,
            bottom_in_dim=512,
            skip_in_dims=(64, 128, 256, 512, 512),
            stage_dims=[16, 32, 64, 128, 256],
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN'))

    def calculate(self, img, encodint=None):
        img_feats = self.backbone(img)
        bottom_feat = img_feats[-1]
        skip_feats = img_feats[:-1]
        mask_feature, mask_logit = self.head(bottom_feat, skip_feats)

        return mask_feature, mask_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        # print("data", label.keys())
        # print("=" * 100)
        if self.training:
            sem_feature, sem_logit = self.calculate(data['img'])
            assert label is not None
            sem_gt_wb = label['sem_gt_inner']
            weight_map = label['loss_weight_map']
            
            loss = dict()
            sem_gt_wb = sem_gt_wb.squeeze(1)
            sem_loss = self._sem_loss(sem_logit, sem_gt_wb, weight_map)
            
            loss.update(sem_loss)

            # calculate training metric
            training_metric_dict = self._training_metric(sem_logit, sem_gt_wb)
            loss.update(training_metric_dict)
            return loss
        else:
            assert metas is not None
            # NOTE: only support batch size = 1 now.
            _, sem_logit = self.inference(data['img'], metas[0], True)
            # print("shape", sem_logit.shape)
            # import matplotlib.pyplot as plt
            # plt.imshow(sem_logit.cpu().numpy()[0,1,...])
            # plt.show()
            # plt.savefig("z_1.png")

            # plt.imshow(sem_logit.cpu().numpy()[0,2,...])
            # plt.show()
            # plt.savefig("z_2.png")

            # plt.imshow(sem_logit.cpu().numpy()[0,3,...])
            # plt.show()
            # plt.savefig("z_3.png")

            # plt.imshow(sem_logit.cpu().numpy()[0,4,...])
            # plt.show()
            # plt.savefig("z_4.png")



            sem_pred = sem_logit.argmax(dim=1)
            # Extract inside class
            sem_pred = sem_pred.cpu().numpy()[0]
            sem_pred, inst_pred = self.postprocess(sem_pred)
            # plt.imshow(sem_pred)
            # plt.show()
            # plt.savefig("z_2.png")

            # unravel batch dim
            ret_list = []
            ret_list.append({'sem_pred': sem_pred, 'inst_pred': inst_pred})
            return ret_list

    def postprocess(self, pred):
        """model free post-process for both instance-level & semantic-level."""
        sem_id_list = list(np.unique(pred))
        inst_pred = np.zeros_like(pred).astype(np.int32)
        sem_pred = np.zeros_like(pred).astype(np.uint8)
        cur = 0
        for sem_id in sem_id_list:
            # 0 is background semantic class.
            if sem_id == 0:
                continue
            sem_id_mask = pred == sem_id
            # fill instance holes
            sem_id_mask = binary_fill_holes(sem_id_mask)
            sem_id_mask = remove_small_objects(sem_id_mask, 5)
            inst_sem_mask = measure.label(sem_id_mask)
            inst_sem_mask = morphology.dilation(inst_sem_mask, selem=morphology.disk(self.test_cfg.get('radius', 1)))
            inst_sem_mask[inst_sem_mask > 0] += cur
            inst_pred[inst_sem_mask > 0] = 0
            inst_pred += inst_sem_mask
            cur += len(np.unique(inst_sem_mask))
            sem_pred[inst_sem_mask > 0] = sem_id

        return sem_pred, inst_pred

    def _sem_loss(self, sem_logit, sem_gt, weight_map):
        """calculate mask branch loss."""
        sem_loss = {}
        sem_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
        # Assign weight map for each pixel position
        sem_ce_loss = sem_ce_loss_calculator(sem_logit, sem_gt) * weight_map
        sem_ce_loss = torch.mean(sem_ce_loss)
        sem_dice_loss = sem_dice_loss_calculator(sem_logit, sem_gt)
        # loss weight
        alpha = 5
        beta = 0.5
        sem_loss['sem_ce_loss'] = alpha * sem_ce_loss
        sem_loss['sem_dice_loss'] = beta * sem_dice_loss

        return sem_loss
