import torch
import torch.nn as nn
import numpy as np
from skimage import morphology, measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
import torch.nn.functional as F
import yaml
import random
from scipy.ndimage import convolve
import math
from ..backbones import TorchVGG16BN
from ..builder import SEGMENTORS
from ..heads import UNetHead
from ..losses import BatchMultiClassDiceLoss
from .base import BaseSegmentor
from ..utils.cellgraph import CellGraph, GINNodeFeatureUpdate, MatrixPredictorGNN


@SEGMENTORS.register_module()
class FCISNet3(BaseSegmentor):
    """Implementation of `U-Net: Convolutional Networks for Biomedical Image Segmentation`"""

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(FCISNet3, self).__init__()
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
        
        self.gin = GINNodeFeatureUpdate(5, 8)

        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2)
        )

        self.gcn = MatrixPredictorGNN(16, 8, 5)
        self.position_encoding = self.generate_sinusoidal_position_encoding(256, 256, 16)

    def calculate(self, img, encoding):
        img_feats = self.backbone(img)
        bottom_feat = img_feats[-1]
        skip_feats = img_feats[:-1]
        mask_feature, mask_logit = self.head(bottom_feat, skip_feats, encoding)

        return mask_feature, mask_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            sem_feature, sem_logit = self.calculate(data['img'], self.position_encoding)
            assert label is not None
            sem_gt_wb = label['sem_gt_inner']
            sem_gt_wb = sem_gt_wb.squeeze(1)
            weight_map = label['loss_weight_map']
            inst_gt = label['inst_gt']
            sem_gt = label['sem_gt']
            adj_gt = label['adj_gt']
           

            loss = dict()
            
            level_set_loss = self.level_set_loss(sem_logit, sem_gt)
            loss.update({"level set loss": level_set_loss})
            
            
            # calculate training metric
            training_metric_dict = self._training_metric(sem_logit, sem_gt_wb)
            loss.update(training_metric_dict)
            return loss

        else:
            assert metas is not None
            # NOTE: only support batch size = 1 now.
            sem_feature, sem_logit = self.inference(data['img'], metas[0], True, encoding=self.position_encoding)
            sem_pred = sem_logit.argmax(dim=1)
            # Extract inside class
            sem_pred = sem_pred.cpu().numpy()[0]
            sem_pred, inst_pred = self.postprocess(sem_pred)
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

    def _sem_loss(self, sem_logit, sem_gt, weight_map=None):
        """calculate mask branch loss."""
        sem_loss = {}
        sem_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
        # Assign weight map for each pixel position
        sem_ce_loss = sem_ce_loss_calculator(sem_logit, sem_gt) #* weight_map
        sem_ce_loss = torch.mean(sem_ce_loss)
        sem_dice_loss = sem_dice_loss_calculator(sem_logit, sem_gt)
        # loss weight
        alpha = 5
        beta = 0.5
        sem_loss['sem_ce_loss'] = alpha * sem_ce_loss
        sem_loss['sem_dice_loss'] = beta * sem_dice_loss

        return sem_loss
    

    def _sem_loss2(self, sem_logit, sem_gt, weight_map=None):
        """calculate mask branch loss."""
        sem_loss = {}
        sem_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
        # Assign weight map for each pixel position
        sem_ce_loss = sem_ce_loss_calculator(sem_logit, sem_gt) #* weight_map
        sem_ce_loss = torch.mean(sem_ce_loss)
        sem_dice_loss = sem_dice_loss_calculator(sem_logit, sem_gt)
        # loss weight
        alpha = 5
        beta = 0.5
        sem_loss['sem_ce_loss2'] = alpha * sem_ce_loss
        sem_loss['sem_dice_loss2'] = beta * sem_dice_loss

        return sem_loss
    



    def _inter_loss(self, pred_features, instance_mask, adjacency_dicts, alpha=0.1):
        """
        Compute orthogonal loss between neighboring nucleus features.

        Args:
            pred_features (torch.Tensor): Predicted feature maps, shape (B, C, H, W), softmax-activated.
            instance_mask (torch.Tensor): Instance segmentation masks, shape (B, H, W), unique IDs for nuclei.
            adjacency_dicts (list[dict]): List of adjacency dictionaries, one per batch.
            alpha (float): Sampling rate, [0, 1].

        Returns:
            dict: A dictionary with the computed loss value.
        """
        # print("pred feature", pred_features.shape)
        # print("instance mask", instance_mask.shape)
        # print("adjaceny dict",  adjacency_dicts)

        batch_size, num_channels, height, width = pred_features.shape
        loss = 0
        device = pred_features.device

        for b in range(batch_size):
            mask = instance_mask[b]
            features = pred_features[b]
            adjacency_dict = adjacency_dicts[b]

            for nucleus_id, neighbors in adjacency_dict.items():
                nucleus_pixels = (mask == nucleus_id).nonzero(as_tuple=False)
                num_pixels = nucleus_pixels.size(0)
                # print("num pixel", num_pixels)

                # Skip if no pixels or no neighbors
                if num_pixels == 0 or len(neighbors) == 0:
                    continue

                # Sample pixels
                sample_size = max(1, int(alpha * num_pixels))
                sampled_indices = random.sample(range(num_pixels), sample_size)
                sampled_pixels = nucleus_pixels[sampled_indices]

                # Extract features for sampled pixels
                nucleus_feature = features[:, sampled_pixels[:, 0], sampled_pixels[:, 1]]  # (C, sample_size)

                # Compare with neighbors
                for neighbor_id in neighbors:
                    neighbor_pixels = (mask == neighbor_id).nonzero(as_tuple=False)
                    num_neighbor_pixels = neighbor_pixels.size(0)

                    if num_neighbor_pixels == 0:
                        continue

                    # Sample neighbor pixels
                    neighbor_sample_size = max(1, int(alpha * num_neighbor_pixels))
                    neighbor_sampled_indices = random.sample(range(num_neighbor_pixels), neighbor_sample_size)
                    neighbor_sampled_pixels = neighbor_pixels[neighbor_sampled_indices]

                    # Extract features for neighbor sampled pixels
                    neighbor_feature = features[:, neighbor_sampled_pixels[:, 0], neighbor_sampled_pixels[:, 1]]  # (C, sample_size)

                    # print("nuclei feature", nucleus_feature.shape)
                    # print("neughbor feature", neighbor_feature.shape)

                    # Compute pairwise dot product and orthogonal loss
                    dot_product = torch.matmul(nucleus_feature.T, neighbor_feature)  # Shape: (sample_size_nucleus, sample_size_neighbor)
                    orthogonality_loss = torch.mean(dot_product ** 2)
                    loss += orthogonality_loss

        loss = loss / batch_size

        return {"inter_loss": loss}

    def _calculate_matrix_loss(self, A_pred, lambda_orth=1.0, lambda_det=1.0):
        """
        Calculate loss for the transformation matrix A_pred.

        Args:
            A_pred (torch.Tensor): The predicted transformation matrix (num_classes x num_classes).
            lambda_orth (float): Weight for orthogonality constraint.
            lambda_det (float): Weight for determinant constraint.

        Returns:
            torch.Tensor: The computed loss for A_pred.
        """
        # Orthogonality loss: ||A_pred^T * A_pred - I||_F^2
        identity = torch.eye(A_pred.size(0), device=A_pred.device)
        orth_loss = torch.norm(torch.mm(A_pred.T, A_pred) - identity, p='fro') ** 2

        # Determinant loss: Regularize determinant to be close to 1
        det_loss = (torch.det(A_pred) - 1) ** 2

        # Weighted sum of losses
        matrix_loss = lambda_orth * orth_loss + lambda_det * det_loss
        return matrix_loss



    

    def compute_adjacency_dict(self, instance_mask):
        """
        Generate adjacency dictionary for instances in the mask.

        Args:
            instance_mask (torch.Tensor): Tensor of shape (B, H, W) with unique IDs for each nucleus.

        Returns:
            list[dict]: A list of adjacency dictionaries for each image in the batch.
        """
        # print("shape", instance_mask.shape)
        batch_size, _,  height, width = instance_mask.shape
        adjacency_dicts = []

        # Define a kernel for 8-connected neighbors
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        for b in range(batch_size):
            mask = instance_mask[b,0].cpu().numpy()
            unique_ids = np.unique(mask)
            adjacency_dict = {int(instance_id): set() for instance_id in unique_ids if instance_id != 0}

            for instance_id in adjacency_dict.keys():
                binary_mask = (mask == instance_id).astype(np.int32)
                boundary = convolve(binary_mask, kernel, mode='constant', cval=0)

                # Identify neighbors
                neighbors = np.unique(mask[boundary > 0])
                adjacency_dict[instance_id].update(int(neighbor) for neighbor in neighbors if neighbor != instance_id and neighbor != 0)

            # Convert sets to lists for compatibility
            adjacency_dicts.append({key: list(value) for key, value in adjacency_dict.items()})

        return adjacency_dicts


    def generate_sinusoidal_position_encoding(self, height, width, channels):
        """
        Generate sinusoidal position encoding for a 2D grid of size (height x width).

        Args:
            height (int): Height of the grid (e.g., 256).
            width (int): Width of the grid (e.g., 256).
            channels (int): Number of encoding channels (must be even).

        Returns:
            torch.Tensor: Position encoding of shape (height, width, channels).
        """
        assert channels % 2 == 0, "Number of channels must be even."

        # Create coordinate grids for height and width
        y_positions = torch.arange(height, dtype=torch.float32).view(-1, 1)  # Shape: (height, 1)
        x_positions = torch.arange(width, dtype=torch.float32).view(-1, 1)  # Shape: (1, width)

        # Calculate the sinusoidal frequencies
        div_term = torch.exp(torch.arange(0, channels // 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / channels))

        # Adjust dimensions for broadcasting
        y_encodings = torch.sin(y_positions * div_term.view(1, -1))  # Shape: (height, channels/2)
        x_encodings = torch.cos(x_positions * div_term.view(1, -1))  # Shape: (width, channels/2)

        # Combine x and y encodings into a 2D grid
        position_encoding = torch.zeros((height, width, channels))
        position_encoding[:, :, 0::2] = y_encodings.unsqueeze(1)  # Add x-encoding along the width
        position_encoding[:, :, 1::2] = x_encodings.unsqueeze(0)  # Add y-encoding along the height

        return position_encoding.permute(2,0,1).unsqueeze(0)


    def level_set_loss(self, preds, targets, epsilon=1.0):
        # heaviside = 0.5 * (1*(2/math.pi) * torch.atan(preds / epsilon))

        region_loss = torch.mean((preds - targets) ** 2)

        disjoint_loss = 0
        for i in range(preds.size(1)):
            for j in range(i+1, preds.size(1)):
                disjoint_loss += torch.mean(torch.relu(-preds[:,i] * preds[:,j]))
        return region_loss + disjoint_loss
        
