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



# class CellGraph:
#     def __init__(self, inst_gt, sem_gt, sem_features, sem_logits, neg_sample_ratio=0.01, sample_ratio=0.3):
#         self.inst_gt = inst_gt
#         self.sem_gt = sem_gt
#         self.sem_features = sem_features
#         self.sem_logits = sem_logits
#         self.neg_sample_ratio = neg_sample_ratio
#         self.sample_ratio = sample_ratio
#         self.device = inst_gt.device
#         self.edge_ind = self.construct_batch_graph()

#     def construct_batch_graph(self):
#         """
#         Construct graph, including node features, edge connections, and edge labels.
#         """
#         batch_edge_index = []
#         batch_size = self.inst_gt.size(0)

#         for batch_idx in range(batch_size):
#             inst_single_gt = self.inst_gt[batch_idx]
#             inst_ids = torch.unique(inst_single_gt)
#             inst_ids = inst_ids[inst_ids != 0]  # Ignore background
#             # id_map = {inst_id.item(): idx for idx, inst_id in enumerate(inst_ids)}

#             edge_index = []
#             for inst_id in inst_ids:
#                 mask = inst_single_gt == inst_id
#                 neighbors = torch.unique(inst_single_gt[self.get_neighbors(mask)])
#                 neighbors = neighbors[(neighbors != 0) & (neighbors != inst_id)]

#                 for nei_id in neighbors:
#                     edge_index.append([inst_id.item(), nei_id.item()])
                  
#             edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).T if edge_index else torch.empty((2, 0), dtype=torch.long, device=self.device)
#             batch_edge_index.append(edge_index)

#         # print("batch edge index", len(batch_edge_index), len(batch_edge_index[0]), len(batch_edge_index[0][0]))
#         # print(batch_edge_index)
#         # print("=" * 100)

#         return batch_edge_index

#     @staticmethod
#     def sample_from_mask(mask, sample_ratio):
#         # print("mask shape", mask.shape)
#         indices = mask.nonzero(as_tuple=False)
#         num_samples = max(1, int(sample_ratio * indices.size(0)))
#         sampled_indices = indices[torch.randperm(indices.size(0))[:num_samples]]
#         # print("sampled indices", sampled_indices.shape)
#         sampled_mask = torch.zeros_like(mask, dtype=torch.bool)
#         sampled_mask[sampled_indices[:, 0], sampled_indices[:, 1]] = True
#         return sampled_mask

#     def sample_representation(self, batch_idx, mask):
#         """
#         Sample pixels from a mask and compute their average representation.
#         Args:
#             mask (torch.Tensor): Boolean mask indicating the pixels to sample from.
#         Returns:
#             torch.Tensor: The averaged representation for the sampled pixels.
#         """
#         # print("mask", mask.shape)
#         # import matplotlib.pyplot as plt
#         # plt.imshow(mask.cpu().numpy())
#         # plt.show()
#         # plt.savefig("z_mask.png")

#         sampled_mask = self.sample_from_mask(mask, self.sample_ratio)
#         # plt.imshow(sampled_mask.cpu().numpy())
#         # plt.show()
#         # plt.savefig("z_sample_mask.png")

#         # print("sampled mask", sampled_mask.shape)
#         # print("sem feature", self.sem_features.shape)
#         # print("=" * 100)
#         sampled_features = self.sem_features[batch_idx,:, sampled_mask]
#         # print("sampled feature", sampled_features.shape)
#         # return torch.unsqueeze(sampled_features.mean(dim=1), dim=-1)
#         return sampled_features.mean(dim=1)



#     def compute_infonce_loss(self, num_positive_samples=10, num_negative_samples=10, temperature=0.1):
#         """
#         Compute InfoNCE loss for cell consistency and adjacency orthogonality.
#         Args:
#             num_positive_samples (int): Number of positive sets to sample for each cell.
#             num_negative_samples (int): Number of negative sets to sample for each cell.
#             temperature (float): Temperature parameter for InfoNCE loss.
#         Returns:
#             float: The computed InfoNCE loss.
#         """
#         total_loss = 0.0
#         total_pairs = 0
#         # print("inst gt", self.inst_gt.shape)
#         for batch_idx in range(len(self.edge_ind)):
#             cur_edge_ind = self.edge_ind[batch_idx]
#             cur_inst_gt = self.inst_gt[batch_idx]

#             for edge_idx in range(cur_edge_ind.size(1)):
#                 src, tgt = cur_edge_ind[:, edge_idx]
#                 src_mask = cur_inst_gt == src
#                 tgt_mask = cur_inst_gt == tgt

#                 # Sample one query from the source cell
#                 query = self.sample_representation(batch_idx, src_mask[0])
#                 query = torch.unsqueeze(query, dim=0)

#                 # Sample multiple positive keys from the source cell
#                 positive_keys = torch.stack([self.sample_representation(batch_idx, src_mask[0]) for _ in range(num_positive_samples)])

#                 # Sample multiple negative keys from the target (adjacent) cell
#                 negative_keys = torch.stack([self.sample_representation(batch_idx, tgt_mask[0]) for _ in range(num_negative_samples)])

#                 # print("query", query.shape)
#                 # print("pos key", positive_keys.shape)
#                 # print("neg key", negative_keys.shape)
#                 # print("=" * 100)

#                 loss = self.infoloss(query, positive_keys, negative_keys, temperature)

#                 total_loss += loss
#                 total_pairs += 1

#         avg_loss = total_loss / total_pairs if total_pairs > 0 else 0.0

#         return {"cons_loss": avg_loss*0.5}
    
#     def infoloss(self, query, pos_keys, neg_keys, tem):
#         N = pos_keys.shape[0]
#         N = max(1, N)

#         pos_keys = pos_keys - query
#         neg_keys = neg_keys - query

#         Q_pos = F.cosine_similarity(query, pos_keys, dim=1)
#         Q_neg = F.cosine_similarity(query, neg_keys, dim=1)
#         Q_neg_exp_sum = torch.sum(torch.exp(Q_neg/tem), dim=0)

#         # print("Q pos", Q_pos.shape)
#         # print("Q neg", Q_neg.shape)
#         # print("Q neg sum", Q_neg_exp_sum.shape)
#         # print("=" * 100)
#         single_in_log = torch.exp(Q_pos/tem)/(torch.exp(Q_pos) + Q_neg_exp_sum)
#         batch_log = torch.sum(-1 * torch.log(single_in_log), dim=0) / N

#         return batch_log
    
#     @staticmethod
#     def get_neighbors(mask):
#         """Find neighbors of a mask using binary dilation."""
#         structuring_element = np.ones((5, 5), dtype=bool)
#         dilated = binary_dilation(mask.cpu().numpy()[0], structure=structuring_element)
#         border = dilated ^ mask.cpu().numpy()
#         return torch.from_numpy(border).bool()






class CellGraph:
    def __init__(self, inst_gt, sem_gt, sem_features, sem_logits, adj_gt, neg_sample_ratio=0.01, sample_ratio=0.1, contrastive_loss=True):
        self.inst_gt = inst_gt
        self.sem_gt = sem_gt
        self.sem_features = sem_features
        self.sem_logits = sem_logits
        self.adj_gt = adj_gt
        self.neg_sample_ratio = neg_sample_ratio
        self.sample_ratio = sample_ratio
        self.device = inst_gt.device  # Ensure all tensors are on the same device
        self.contrastive_loss = contrastive_loss
        self.node_features, self.node_logits, self.node_labels, self.edge_ind, self.edge_labels, self.contrastive_loss = self.construct_batch_graph()

    def construct_batch_graph(self):
        """
        Construct graph, including node features, edge connections, and edge labels.
        """
        batch_node_features = []
        batch_node_logits = []
        batch_node_labels = []
        batch_edge_index = []
        batch_edge_labels = []
        # contrastive_pairs = []

        total_loss = 0.0
        # total_pairs = 0

        total_nodes = 0
        batch_size = self.inst_gt.size(0)

        for batch_idx in range(batch_size):
            inst_single_gt = self.inst_gt[batch_idx]
            sem_single_feature = self.sem_features[batch_idx]
            sem_single_logit = self.sem_logits[batch_idx]
            single_adj_gt = self.adj_gt[batch_idx]

            inst_ids = torch.unique(inst_single_gt)
            inst_ids = inst_ids[inst_ids != 0]  # Ignore background
            id_map = {inst_id.item(): idx for idx, inst_id in enumerate(inst_ids)}

            node_features = []
            node_logits = []
            node_labels = []
            edge_index = []
            edge_labels = []
            # instance_contrastive_pairs = []
            # print("sem single feature", sem_single_feature.shape)

            # print("inst id", inst_ids)
            # print("single adj gt", single_adj_gt)

            for inst_id in inst_ids:
                mask = inst_single_gt == inst_id

                sampled_mask = self.sample_from_mask(mask, self.sample_ratio)[0]

                node_features.append(sem_single_feature[:, sampled_mask].mean(dim=1))
                # node_logits.append(sem_single_logit[:, sampled_mask].mean(dim=1))
                node_labels.append(inst_id.item())

                # positive_pairs = self.sample_contrastive_pairs(inst_id, inst_ids, single_adj_gt, positive=True)
                # instance_contrastive_pairs.extend(positive_pairs)

                # negative_pairs = self.sample_contrastive_pairs(inst_id, inst_ids, single_adj_gt, positive=False)
                # instance_contrastive_pairs.extend(negative_pairs)

                # neighbors = torch.unique(inst_single_gt[self.get_neighbors(mask)])
                # neighbors = neighbors[(neighbors != 0) & (neighbors != inst_id)]
                
                # print("inst id", inst_id)
                # print("adj gt", single_adj_gt)
                # print("=" * 100)

                neighbors = single_adj_gt[int(inst_id.item())]

                for nei_id in inst_ids:
                    if nei_id in neighbors:
                        edge_index.append([id_map[inst_id.item()], id_map[nei_id.item()]])
                        edge_labels.append(1)  # Connected edge
                        if self.contrastive_loss:
                            # print("edge length", edge_index)
                            # edge_index_f = edge_index[:, edge_labels==1]
                            # print("f edge length", edge_index_f)
                            cont_loss = self.compute_infonce_loss([inst_id.item(), nei_id.item()], inst_single_gt, sem_single_feature)
                            total_loss += cont_loss
                            # total_pairs += 1
                    elif random.random() < self.neg_sample_ratio:
                        edge_index.append([id_map[inst_id.item()], id_map[nei_id.item()]])
                        edge_labels.append(0)  # Non-connected edge

            if node_features:
                node_features = torch.stack(node_features)
                # node_logits = torch.stack(node_logits)
                node_labels = torch.tensor(node_labels, dtype=torch.float32, device=self.device).view(-1, 1)
                c = node_labels.size(0)
            else:
                node_features = torch.empty(size=(0, sem_single_feature.size(0)), device=self.device)
                # node_logits = torch.empty(size=(0, sem_single_logit.size(0)), device=self.device)
                # node_labels = torch.empty(size=(0, 1), device=self.device)
                c = 0

            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).T if edge_index else torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_labels = torch.tensor(edge_labels, dtype=torch.long, device=self.device) if edge_labels else torch.empty(0, dtype=torch.long, device=self.device)

            edge_index += total_nodes
            total_nodes += c

            # print("node feature", node_features.shape)

            batch_node_features.append(node_features)
            # batch_node_logits.append(node_logits)
            # batch_node_labels.append(node_labels)
            batch_edge_index.append(edge_index)
            batch_edge_labels.append(edge_labels)

            # contrastive_pairs.append(instance_contrastive_pairs)

            

        batch_node_features = torch.cat(batch_node_features, dim=0) if batch_node_features else torch.empty((0, self.sem_features.size(2)), device=self.device)
        # batch_node_logits = torch.cat(batch_node_logits, dim=0) if batch_node_logits else torch.empty((0, self.sem_logits.size(2)), device=self.device)
        # batch_node_labels = torch.cat(batch_node_labels, dim=0).squeeze() if batch_node_labels else torch.empty(0, device=self.device)
        batch_edge_index = torch.cat(batch_edge_index, dim=1) if batch_edge_index else torch.empty((2, 0), dtype=torch.long, device=self.device)
        batch_edge_labels = torch.cat(batch_edge_labels, dim=0) if batch_edge_labels else torch.empty(0, dtype=torch.long, device=self.device)

        # avg_cont_loss = total_loss / total_pairs if total_pairs > 0 else 0.0

        return batch_node_features, batch_node_logits, batch_node_labels, batch_edge_index, batch_edge_labels, total_loss

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
    
    # def sample_contrastive_pairs(self, inst_id, inst_ids, adj_gt, positive=True):
    #     """
    #     Sample contrastive pairs based on instance adjacency.
    #     Args:
    #         inst_id (int): The instance id for which we are generating the pairs.
    #         inst_ids (list): List of all unique instance ids.
    #         adj_gt (Tensor): Adjacency tensor for the current batch.
    #         positive (bool): Flag indicating whether to sample positive or negative pairs.
    #     Returns:
    #         list: List of contrastive pairs [(idx1, idx2)].
    #     """
    #     contrastive_pairs = []
    #     # print("inst id", inst_id)

    #     # print("positive", positive)
        
    #     if positive:
    #         # Sample positive pairs (same instance)
    #         contrastive_pairs = [(inst_id.item(), inst_id.item())]  # Positive samples come from the same instance
    #     else:
    #         # Sample negative pairs (adjacent or different instance)
    #         neighbors = adj_gt[int(inst_id.item())]
    #         # print("neighbors", neighbors)
    #         for neighbor_id in inst_ids:
    #             if neighbor_id != inst_id.item() and neighbor_id in neighbors:
    #                 # print("neighbor id", neighbor_id)
                    
    #                 contrastive_pairs.append((inst_id.item(), neighbor_id))
    #     # print("=" * 100)

        # return contrastive_pairs
    
    def compute_infonce_loss(self, edge_ind, inst_gt, sem_fea, num_positive_samples=10, num_negative_samples=10, temperature=0.3):
        """
        Compute InfoNCE loss for cell consistency and adjacency orthogonality.
        Args:
            num_positive_samples (int): Number of positive sets to sample for each cell.
            num_negative_samples (int): Number of negative sets to sample for each cell.
            temperature (float): Temperature parameter for InfoNCE loss.
        Returns:
            float: The computed InfoNCE loss.
        """
        total_loss = 0.0
        total_pairs = 0
        # print("inst gt", self.inst_gt.shape)
        
        # cur_edge_ind = self.edge_ind[batch_idx]
        # cur_inst_gt = self.inst_gt[batch_idx]



        # for edge_idx in range(edge_ind.size(1)):
        src, tgt = edge_ind[0], edge_ind[1]
        src_mask = inst_gt == src
        tgt_mask = inst_gt == tgt

        # print("src mask", src_mask.shape)
        # print("tgt mask", tgt_mask.shape)

        # Sample one query from the source cell
        query = self.sample_representation(sem_fea, src_mask[0])
        query = torch.unsqueeze(query, dim=0)

        # print("sem feature", sem_fea)

        # Sample multiple positive keys from the source cell
        positive_keys = torch.stack([self.sample_representation(sem_fea, src_mask[0]) for _ in range(num_positive_samples)])

        # Sample multiple negative keys from the target (adjacent) cell
        negative_keys = torch.stack([self.sample_representation(sem_fea, tgt_mask[0]) for _ in range(num_negative_samples)])

        # print("query", query)
        # print("pos key", positive_keys)
        # print("neg key", negative_keys)
        # print("=" * 100)

        loss = self.infoloss(query, positive_keys, negative_keys, temperature)

        total_loss += loss
        total_pairs += 1
        # print("contrastive loss", loss)

        avg_loss = total_loss / total_pairs if total_pairs > 0 else 0.0

        return avg_loss
    
    def infoloss(self, query, pos_keys, neg_keys, tem):
        N = pos_keys.shape[0]
        N = max(1, N)

        pos_keys = pos_keys - query
        neg_keys = neg_keys - query

        Q_pos = F.cosine_similarity(query, pos_keys, dim=1)
        Q_neg = F.cosine_similarity(query, neg_keys, dim=1)
        Q_neg_exp_sum = torch.sum(torch.exp(Q_neg/tem), dim=0)

        # print("Q pos", Q_pos)
        # print("Q neg", Q_neg)
        # print("Q neg sum", Q_neg_exp_sum)
        # print("=" * 100)
        single_in_log = torch.exp(Q_pos/tem)/(torch.exp(Q_pos) + Q_neg_exp_sum)
        batch_log = torch.sum(-1 * torch.log(single_in_log), dim=0) / N

        return batch_log
    
    def sample_representation(self, sem_feature, mask):
        """
        Sample pixels from a mask and compute their average representation.
        Args:
            mask (torch.Tensor): Boolean mask indicating the pixels to sample from.
        Returns:
            torch.Tensor: The averaged representation for the sampled pixels.
        """
        # print("mask", mask.shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(mask.cpu().numpy())
        # plt.show()
        # plt.savefig("z_mask.png")

        sampled_mask = self.sample_from_mask(mask, self.sample_ratio)
        # print("mask length", (mask>0).sum())
        # import matplotlib.pyplot as plt
        # plt.imshow(sampled_mask.cpu().numpy())
        # plt.show()
        # plt.savefig("z_sample_mask.png")

        # print("sampled mask", sampled_mask.shape)
        # print("sem feature", self.sem_features.shape)
        # print("=" * 100)
        sampled_features = sem_feature[:, sampled_mask]
        # print("sampled feature", sampled_features.shape)
        # return torch.unsqueeze(sampled_features.mean(dim=1), dim=-1)
        return sampled_features.mean(dim=1)

 
    # @staticmethod
    # def sample_from_mask(mask, sample_ratio):
    #     """
    #     Sample a fraction of `sample_ratio` elements from the mask.
    #     Args:
    #         mask (torch.Tensor): Boolean mask tensor of shape (H, W).
    #         sample_ratio (float): Fraction of elements to sample.
    #     Returns:
    #         torch.Tensor: A boolean mask with the same shape as input, where only sampled elements are True.
    #     """
    #     indices = mask.nonzero(as_tuple=False)  # Get indices of True elements
    #     num_samples = max(1, int(sample_ratio * indices.size(0)))  # Calculate the number of samples
    #     sampled_indices = indices[torch.randperm(indices.size(0))[:num_samples]]  # Randomly sample
    #     sampled_mask = torch.zeros_like(mask, dtype=torch.bool)  # Initialize a new empty mask
    #     sampled_mask[sampled_indices[:, 0], sampled_indices[:, 1]] = True  # Set sampled indices to True
    #     return sampled_mask
    
    # @staticmethod
    # def get_neighbors(mask):
    #     """Find neighbors of a mask using binary dilation."""
    #     structuring_element = np.ones((5, 5), dtype=bool)
    #     dilated = binary_dilation(mask.cpu().numpy()[0], structure=structuring_element)
    #     border = dilated ^ mask.cpu().numpy()
    #     return torch.from_numpy(border).bool()

### ====================================================
### Splitting Line
### ====================================================
# class CellGraph:
#     def __init__(self, inst_gt, sem_gt, sem_features, sem_logits, neg_sample_ratio=0.01):
#         self.inst_gt = inst_gt
#         self.sem_gt = sem_gt
#         self.sem_features = sem_features
#         self.sem_logits = sem_logits
#         self.neg_sample_ratio = neg_sample_ratio
#         self.device = inst_gt.device  # Ensure all tensors are on the same device
#         self.node_features, self.node_logits, self.node_labels, self.edge_ind, self.edge_labels = self.construct_batch_graph()

#     def construct_batch_graph(self):
#         """
#         Construct graph, including node features, edge connections, and edge labels.
#         """
#         batch_node_features = []
#         batch_node_logits = []
#         batch_node_labels = []
#         batch_edge_index = []
#         batch_edge_labels = []

#         total_nodes = 0
#         batch_size = self.inst_gt.size(0)

#         for batch_idx in range(batch_size):
#             inst_single_gt = self.inst_gt[batch_idx]
#             sem_single_feature = self.sem_features[batch_idx]
#             sem_single_logit = self.sem_logits[batch_idx]

#             inst_ids = torch.unique(inst_single_gt)
#             inst_ids = inst_ids[inst_ids != 0]  # Ignore background
#             id_map = {inst_id.item(): idx for idx, inst_id in enumerate(inst_ids)}

#             node_features = []
#             node_logits = []
#             node_labels = []
#             edge_index = []
#             edge_labels = []
#             # print("inst ids", inst_ids)
#             # print("=" * 100)

#             for inst_id in inst_ids:
#                 mask = inst_single_gt == inst_id
#                 node_features.append(sem_single_feature[:, mask[0, ...]].mean(dim=1))
#                 node_logits.append(sem_single_logit[:, mask[0, ...]].mean(dim=1))
#                 node_labels.append(inst_id.item())

#                 neighbors = torch.unique(inst_single_gt[self.get_neighbors(mask)])
#                 neighbors = neighbors[(neighbors != 0) & (neighbors != inst_id)]

#                 for nei_id in inst_ids:
#                     if nei_id in neighbors:
#                         edge_index.append([id_map[inst_id.item()], id_map[nei_id.item()]])
#                         edge_labels.append(1)  # Connected edge
#                     elif random.random() < self.neg_sample_ratio:
#                         edge_index.append([id_map[inst_id.item()], id_map[nei_id.item()]])
#                         edge_labels.append(0)  # Non-connected edge

#             if node_features:
#                 node_features = torch.stack(node_features)
#                 node_logits = torch.stack(node_logits)
#                 node_labels = torch.tensor(node_labels, dtype=torch.float32, device=self.device).view(-1, 1)
#                 c = node_labels.size(0)
#             else:
#                 node_features = torch.empty(size=(0, sem_single_feature.size(0)), device=self.device)
#                 node_logits = torch.empty(size=(0, sem_single_logit.size(0)), device=self.device)
#                 node_labels = torch.empty(size=(0, 1), device=self.device)
#                 c = 0

#             edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).T if edge_index else torch.empty((2, 0), dtype=torch.long, device=self.device)
#             edge_labels = torch.tensor(edge_labels, dtype=torch.long, device=self.device) if edge_labels else torch.empty(0, dtype=torch.long, device=self.device)

#             edge_index += total_nodes
#             total_nodes += c

#             batch_node_features.append(node_features)
#             batch_node_logits.append(node_logits)
#             batch_node_labels.append(node_labels)
#             batch_edge_index.append(edge_index)
#             batch_edge_labels.append(edge_labels)

#         batch_node_features = torch.cat(batch_node_features, dim=0) if batch_node_features else torch.empty((0, self.sem_features.size(2)), device=self.device)
#         batch_node_logits = torch.cat(batch_node_logits, dim=0) if batch_node_logits else torch.empty((0, self.sem_logits.size(2)), device=self.device)
#         batch_node_labels = torch.cat(batch_node_labels, dim=0).squeeze() if batch_node_labels else torch.empty(0, device=self.device)
#         batch_edge_index = torch.cat(batch_edge_index, dim=1) if batch_edge_index else torch.empty((2, 0), dtype=torch.long, device=self.device)
#         batch_edge_labels = torch.cat(batch_edge_labels, dim=0) if batch_edge_labels else torch.empty(0, dtype=torch.long, device=self.device)

#         return batch_node_features, batch_node_logits, batch_node_labels, batch_edge_index, batch_edge_labels
    

#     @staticmethod
#     def get_neighbors(mask):
#         from scipy.ndimage import binary_dilation
#         import numpy as np
#         structuring_element = np.ones((5, 5), dtype=bool)
#         dilated = binary_dilation(mask.cpu().numpy()[0], structure=structuring_element)
#         boarder = dilated ^ mask.cpu().numpy()
        
#         return torch.from_numpy(boarder).bool()




    
class GINNodeFeatureUpdate(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(GINNodeFeatureUpdate, self).__init__()

        self.layers = nn.ModuleList(
            [GINConvCustom(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )

        self.fc_out = nn.Linear(hidden_dim, input_dim)

        # self.conv = GINConv(
        #     nn.Sequential(
        #         nn.Linear(input_dim, input_dim),
        #         nn.ReLU(),
        #         nn.Linear(hidden_dim, hidden_dim)
        #     ))
        # nn.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, node_features, edge_index):
        # update node features
        # updated_fea = self.conv(node_features, edge_index)
        # updated_fea = self.fc(updated_fea)

        for layer in self.layers:
            node_features = layer(node_features, edge_index)

        return self.fc_out(node_features)


class GINConvCustom(nn.Module):
    def __init__(self, input_dim, hidden_dim, eps=0.0):
        super(GINConvCustom, self).__init__()
        self.eps = nn.Parameter(torch.tensor(eps))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, node_features, edge_index):
        src, dst = edge_index
        aggregated_features = torch.zeros_like(node_features)
        # print("aggregate features", aggregated_features.shape)
        # print("index", dst, src)

        aggregated_features = aggregated_features.index_add(0, dst, node_features[src])

        updated_features = (1+self.eps) * node_features + aggregated_features

        return self.mlp(updated_features)
    
class MatrixPredictorGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MatrixPredictorGNN, self).__init__()

        self.conv1 = GCNLayer(in_channels, hidden_channels)
        self.conv2 = GCNLayer(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels*out_channels)
        self.out_channels = out_channels

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        A_flat = self.fc(x.mean(dim=0))

        return A_flat.view(self.out_channels, self.out_channels)
    

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, node_features, edge_index):
        # Get adjacency matrix with self-loops
        num_nodes = node_features.size(0)
        edge_index = torch.cat([edge_index, torch.arange(0, num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)], dim=1)
        
        # Compute degree matrix
        degrees = torch.zeros(num_nodes, device=edge_index.device).scatter_add(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))

        # Normalize adjacency matrix
        deg_inv_sqrt = degrees.pow(-0.5)
        deg_inv_sqrt[degrees == 0] = 0
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]

        # Message passing (aggregate and combine features)
        edge_messages = node_features[edge_index[1]] * norm.unsqueeze(1)
        aggregated_messages = torch.zeros_like(node_features).scatter_add_(0, edge_index[0].unsqueeze(-1).expand_as(edge_messages), edge_messages)

        # Apply learnable weights
        out = self.linear(aggregated_messages)
        return out