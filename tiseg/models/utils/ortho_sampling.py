import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
# from torch_geometric.nn import GCNConv
# from torch_scatter import scatter_mean
# from torch_geometric.nn import GINConv

import torch
import random

import torch
import random
import numpy as np
from scipy.ndimage import binary_dilation



class OrthoLoss:
    def __init__(self, inst_gt, adj_gt, sem_gt, sem_pred, sample_ratio=0.3):
        self.inst_gt = inst_gt
        self.adj_gt = adj_gt
        self.sem_gt = sem_gt
        self.sem_pred = F.softmax(sem_pred, dim=1)
        # print("sem pred", self.sem_pred.shape)
        self.sample_ratio = sample_ratio
        self.device = inst_gt.device  # Ensure all tensors are on the same device
        self.loss = self.calculate_ortho_loss()

    def calculate_ortho_loss(self):

        total_loss = 0.0
        
        batch_size = self.inst_gt.size(0)

        for batch_idx in range(batch_size):
            pairs = 0
            loss = 0.0
            single_inst_gt = self.inst_gt[batch_idx]
            # sem_single_feature = self.sem_features[batch_idx]
            # sem_single_logit = self.sem_logits[batch_idx]
            single_adj_gt = self.adj_gt[batch_idx]
            single_sem_pred = self.sem_pred[batch_idx]

            for cur_node, neighbors in single_adj_gt.items():
                cur_sampled_mask = self.sample_from_mask(single_inst_gt == cur_node, self.sample_ratio)
                if cur_sampled_mask.sum() == 0:
                    continue
                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.imshow(cur_sampled_mask.cpu().numpy()[0])
                # plt.show()
                # plt.savefig("z_cur_mask.png")
                for nei_node in neighbors:
                    nei_sampled_mask = self.sample_from_mask(single_inst_gt == nei_node, self.sample_ratio)
                    if nei_sampled_mask.sum() == 0:
                        continue
                    ortho_loss = self.ortho_loss(cur_sampled_mask, nei_sampled_mask, batch_idx)

                    # plt.figure()
                    # plt.imshow(cur_sampled_mask.cpu().numpy()[0] + nei_sampled_mask.cpu().numpy()[0])
                    # plt.show()
                    # plt.savefig("z_nei_mask.png")

                    pairs += 1
                    loss += ortho_loss

            total_loss += loss / pairs if pairs > 0 else 0.0

        return  {"ortho_loss": total_loss/batch_size * 2.0}

    @staticmethod
    def sample_from_mask(mask, sample_ratio):
        """
        Optimized version of sample_from_mask to speed up the sampling process.
        Args:
            mask (torch.Tensor): Boolean mask tensor of shape (H, W).
            sample_ratio (float): Fraction of elements to sample.
        Returns:
            torch.Tensor: A boolean mask with the same shape as input, where only sampled elements are True.
        """
        # Flatten the mask for easier sampling
        flat_mask = mask.flatten()
        
        # Count the number of True elements in the mask
        true_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
        num_true = true_indices.numel()
        
        # Determine the number of samples to take
        num_samples = max(1, int(sample_ratio * num_true))
        
        if num_samples >= num_true:  # If the sample size exceeds the number of available elements
            sampled_indices = true_indices
        else:
            # Use random sampling on the indices
            sampled_indices = true_indices[torch.randint(0, num_true, (num_samples,), device=mask.device)]
        
        # Create a new mask with the same shape as the input
        sampled_mask = torch.zeros_like(flat_mask, dtype=torch.bool)
        sampled_mask[sampled_indices] = True
        
        # Reshape the mask back to the original shape
        return sampled_mask.view(mask.shape)
    

    def ortho_loss(self, cur_mask, nei_mask, batch_idx):
        cur_pred = self.sem_pred[batch_idx, :, cur_mask[0]]
        nei_pred = self.sem_pred[batch_idx, :, nei_mask[0]]

        # cur_pred_norms = torch.clamp(torch.norm(cur_pred, dim=0, keepdim=True), min=1e-8)
        # nei_pred_norms = torch.clamp(torch.norm(nei_pred, dim=0, keepdim=True), min=1e-8)

        # cur_pred_normalized = cur_pred / cur_pred_norms
        # nei_pred_normalized = nei_pred / nei_pred_norms

        # print("cur pred", cur_pred.shape)
        # print("nei pred", nei_pred.shape)

        # cosine_similarity = torch.exp(torch.matmul(cur_pred_normalized.T, nei_pred_normalized)/0.3)
        pos_cosine_similarity = torch.matmul(cur_pred.T, nei_pred)
        neg_cosine_similarity = torch.matmul(cur_pred.T, nei_pred)
        # print("cosine similarity", cosine_similarity)

        # loss1 = -1 * torch.log(torch.exp(pos_cosine_similarity/0.3).sum() / (torch.exp(pos_cosine_similarity).sum() + torch.exp(neg_cosine_similarity).sum()))
        loss2 =  neg_cosine_similarity.mean()
        
        # print("loss", loss)

        
        # print("=" * 100)
        # print("sem gt", self.sem_gt.shape)
        # print("cur mask", cur_mask.shape)
        # print("nei mask", nei_mask.shape)
        # cur_gt = self.sem_gt[batch_idx, :, cur_mask[0]].type(torch.float32)
        # nei_gt = self.sem_gt[batch_idx, :, nei_mask[0]].type(torch.float32)
        # print("cur pred", cur_pred.shape)
        # print("cur gt", torch.unique(cur_gt))

        # print("nei pred", nei_pred.shape)
        # print("nei gt", torch.unique(nei_gt))
        # print("=" * 50)

        # diff_pred = cur_pred.unsqueeze(-1) - nei_pred.unsqueeze(-1).permute(0, 2, 1)
        # diff_gt = cur_gt.unsqueeze(-1) - nei_gt.unsqueeze(-1).permute(0, 2, 1)
        # # print("diff gt", diff_gt)

        # print("diff pred", diff_pred.shape)
        # print("diff gt", diff_gt.shape)

        # 公式中的损失函数：|Pi - Pj| / d(i, j)
        # loss = torch.norm(torch.norm(diff_pred) - torch.norm(diff_gt)).mean()
        # loss = 0.0

        return loss2 * 2

    
   