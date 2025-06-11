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

from ..backbones import TorchVGG16BN
from ..builder import SEGMENTORS
from ..heads import UNetHead
from ..losses import BatchMultiClassDiceLoss
from .base import BaseSegmentor
from ..utils.cellgraph import CellGraph, GINNodeFeatureUpdate


@SEGMENTORS.register_module()
class FCISNet(BaseSegmentor):
    """Implementation of `U-Net: Convolutional Networks for Biomedical Image Segmentation`"""

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(FCISNet, self).__init__()
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

    def calculate(self, img):
        img_feats = self.backbone(img)
        bottom_feat = img_feats[-1]
        skip_feats = img_feats[:-1]
        mask_logit = self.head(bottom_feat, skip_feats)

        return mask_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            sem_logit = self.calculate(data['img'])
            assert label is not None
            sem_gt_wb = label['sem_gt_inner']
            weight_map = label['loss_weight_map']
            inst_gt = label['inst_gt']
            sem_gt = label['sem_gt']
            # print(label.keys())
            # print("GT", sem_gt_wb.shape, inst_gt.shape)
            # print("=" * 100)

            loss = dict()
            sem_gt_wb = sem_gt_wb.squeeze(1)
            # sem_loss = self._sem_loss(sem_logit, sem_gt_wb, weight_map)
            sem_loss = self._sem_loss(sem_logit, sem_gt[:,0,...])
            loss.update(sem_loss)



            ### =============================
            ### calculate iso loss
            ### =============================


            cell_graph = CellGraph(inst_gt, sem_gt)
            node_features = cell_graph.node_feas
            edge_index = cell_graph.edge_ind

            
            predicted_features = []
            # print("sem logits", sem_logit.shape)

            batch_size = sem_gt.size(0)

            for batch_idx in range(batch_size):
                inst_single_gt = inst_gt[batch_idx]
                sem_single_logit = sem_logit[batch_idx]
                inst_ids = torch.unique(inst_single_gt[inst_single_gt != 0])

                sample_size = 50  # Number of pixels to sample per mask

                
                for inst_id in inst_ids:
                    mask = inst_single_gt == inst_id
                    # Get indices of all masked pixels
                    mask_indices = mask.nonzero(as_tuple=False)
                    
                    # If there are fewer pixels than the sample size, use all pixels; otherwise, sample
                    if mask_indices.size(0) > sample_size:
                        sampled_indices = mask_indices[torch.randperm(mask_indices.size(0))[:sample_size]]
                    else:
                        sampled_indices = mask_indices

                    # print("sample indice", sampled_indices)
                    # Extract and average the sampled features
                    sampled_features = sem_single_logit.permute(1, 2, 0)[sampled_indices[:, 0], sampled_indices[:, 1]]
                    features = sampled_features.mean(dim=0)
                    # print("features", features.shape)

                    predicted_features.append(features)

                # print("num", len(predicted_features))


            predicted_features = torch.stack(predicted_features)
            device = predicted_features.device
            edge_index = edge_index.to(device)
            node_features = node_features.type(torch.int64).to(device)


            print("feature shape", predicted_features.shape)
            print("edge shape", edge_index[0].shape)
            print("node shape", node_features.shape)

            updated_features = self.gin(predicted_features, edge_index)
            iso_loss = F.cross_entropy(updated_features, node_features)

            loss.update({"iso_loss": iso_loss})

            """
            #     for inst_id in inst_ids:
            #         mask = inst_single_gt == inst_id
            #         features = sem_single_logit.permute(1,2,0)[mask[0]].mean(dim=0)
                    
            #         predicted_features.append(features)
            # predicted_features = torch.stack(predicted_features)
            # device = predicted_features.device
            # edge_index = edge_index.to(device)
            # node_faetures = node_faetures.type(torch.int64).to(device)
            
            """
            

            


            # calculate training metric
            training_metric_dict = self._training_metric(sem_logit, sem_gt_wb)
            loss.update(training_metric_dict)
            return loss
        else:
            assert metas is not None
            # NOTE: only support batch size = 1 now.
            sem_logit = self.inference(data['img'], metas[0], True)
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

                    
