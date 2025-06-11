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
from ..heads import UNetHead_FCIS as UNetHead
from ..losses import BatchMultiClassDiceLoss, GradientMSELoss
from .base import BaseSegmentor
from ..utils.ortho_sampling import OrthoLoss


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
        self.position_encoding = self.generate_sinusoidal_position_encoding(256, 256, 16)
        

    def calculate(self, img, encoding):
        img_feats = self.backbone(img)
        bottom_feat = img_feats[-1]
        skip_feats = img_feats[:-1]
        sem_pred, inter_pred = self.head(bottom_feat, skip_feats, encoding)

        # sem_logit = torch.cat([sem_pred[:, :1, ...], cls_pred], dim=1)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(sem_pred[0,0,...].detach().cpu().numpy())
        # plt.show()
        # plt.savefig("z_sem_back.png")

        # plt.figure()
        # plt.imshow(sem_pred[0,1,...].detach().cpu().numpy())
        # plt.show()
        # plt.savefig("z_sem_fore.png")

        # for i in range(5):
        #     plt.figure()
        #     plt.imshow(inter_pred[0,i,...].detach().cpu().numpy())
        #     plt.show()
        #     plt.savefig(f"z_cls_pred_{i}.png")

        return sem_pred, inter_pred

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            sem_pred, inter_pred = self.calculate(data['img'], encoding=None)

            # sem_pred, inter_pred = self.calculate(data['img'], self.position_encoding)

            assert label is not None

            sem_gt_wb = label['sem_gt_inner'].squeeze(1)
            sem_gt = label['sem_gt']
            ins_gt = label['inst_gt']
            adj_gt = label['adj_gt']
            # print("sem gt", sem_gt.shape)
           
            # weight_map = label['loss_weight_map']

            bi_sem_gt = (sem_gt_wb.clone()>0).long()
            # bi_sem_weight = (sem_gt_wb.clone() < 2).squeeze(1)

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(sem_gt[0,0,...].cpu().numpy())
            # plt.show()
            # plt.savefig("z_sem_gt.png")

            # plt.figure()
            # plt.imshow(bi_sem_gt[0,...].cpu().numpy())
            # plt.show()
            # plt.savefig("z_bi_sem_gt.png")
       

            loss = dict()
            
            # Calculate the foreground and background semantic loss
            # print('inter pred', sem_pred.shape)
            # print('inter gt', bi_sem_gt.shape)
            # print('=' * 100)
            sem_loss = self._sem_loss(sem_pred, bi_sem_gt)
            loss.update(sem_loss)

            # print('inter pred', inter_pred.shape)
            # print('inter gt', sem_gt_wb.shape)
            # print('=' * 100)
            
            
            cls_loss = self._cls_loss(inter_pred, sem_gt_wb, bi_sem_gt)
            loss.update(cls_loss)
            


            # Calculate the ortholodical loss inner and inter cells
            ortho = OrthoLoss(ins_gt, adj_gt, sem_gt, inter_pred, sample_ratio=0.7)
            ortho_loss = ortho.calculate_ortho_loss()
            loss.update(ortho_loss)
            

            # calculate training metric
            training_metric_dict = self._training_metric(sem_pred, sem_gt_wb)
            loss.update(training_metric_dict)
            return loss
        else:
            assert metas is not None
            
            # NOTE: only support batch size = 1 now.
            # sem_logit = self.inference(data['img'], metas[0], True, encoding=self.position_encoding)
            # print('data shape', data.shape)
            # print('data', data)
            sem_logit, cls_logit = self.inference(data['img'], metas[0], True, encoding=None)
            # print('data', data)

            # np.save("z_sem_logit.npy", sem_logit.cpu().numpy())
            # np.save("z_cls_logit.npy", cls_logit.cpu().numpy())

            # print("sem pred shape", sem_pred.shape)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(sem_logit.cpu().numpy()[0,0,...])
            # plt.show()
            # plt.savefig("z_sem_logit.png")

            # import matplotlib.pyplot as plt
            # for i in range(5):
            #     plt.figure()
            #     plt.imshow(cls_logit[0,i,...].detach().cpu().numpy())
            #     plt.show()
            #     plt.savefig(f"z_clsss_pred_{i}.png")

            sem_pred = sem_logit.argmax(dim=1)
            cls_pred = cls_logit.argmax(dim=1)
            cls_pred[sem_pred != 1] = 0

            # plt.figure()
            # plt.imshow(cls_pred.cpu().numpy()[0])
            # plt.show()
            # plt.savefig('z_cls_pred.png')
            # print('cls pred', cls_pred.shape)

            # Extract inside class
            cls_pred = cls_pred.cpu().numpy()[0]

            
            cls_pred, inst_pred = self.postprocess(cls_pred)
            # plt.figure()
            # plt.imshow(inst_pred)
            # plt.show()
            # plt.savefig("z_inst_pred.png")
            # unravel batch dim
            ret_list = []
            ret_list.append({'sem_pred': cls_pred, 'inst_pred': inst_pred})
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
        # sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
        sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=2)
        # Assign weight map for each pixel position
        sem_ce_loss = sem_ce_loss_calculator(sem_logit, sem_gt) #* weight_map
        sem_ce_loss = torch.mean(sem_ce_loss)
        sem_dice_loss = sem_dice_loss_calculator(sem_logit, sem_gt) #* weight_map
        # loss weight
        alpha = 5
        beta = 0.5
        sem_loss['sem_ce_loss'] = alpha * sem_ce_loss
        sem_loss['sem_dice_loss'] = beta * sem_dice_loss

        return sem_loss
    

    # def _cls_loss(self, sem_logit, sem_gt, weight_map=None):
    #     """calculate mask branch loss."""
    #     sem_loss = {}
    #     sem_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
    #     # sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
    #     sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=5)
    #     # Assign weight map for each pixel position
    #     sem_ce_loss = sem_ce_loss_calculator(sem_logit, sem_gt) * weight_map
    #     sem_ce_loss = torch.mean(sem_ce_loss)
    #     sem_dice_loss = sem_dice_loss_calculator(sem_logit, sem_gt) * weight_map
    #     # loss weight
    #     alpha = 10
    #     beta = 1
    #     sem_loss['cls_ce_loss'] = alpha * sem_ce_loss
    #     sem_loss['cls_dice_loss'] = beta * sem_dice_loss

    #     return sem_loss
    

    def _cls_loss(self, sem_logit, sem_gt, weight_map=None):
        """Calculate mask branch loss with channel regularization."""
        sem_loss = {}

        # 交叉熵损失
        sem_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        sem_ce_loss = sem_ce_loss_calculator(sem_logit, sem_gt)
        if weight_map is not None:
            sem_ce_loss = sem_ce_loss * weight_map
        sem_ce_loss = torch.mean(sem_ce_loss)

        # Dice 损失
        sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=5)
        sem_dice_loss = sem_dice_loss_calculator(sem_logit, sem_gt)
        if weight_map is not None:
            sem_dice_loss = sem_dice_loss * weight_map

        # 通道正则化 (L2 正则化或余弦相似度)
        # 计算预测概率（Softmax 后）
        pred_prob = F.softmax(sem_logit, dim=1)

        # 方法 1: 使用 L2 正则化限制通道值大小
        # l2_reg = torch.mean(torch.sum(pred_prob**2, dim=1))

        # 方法 2: 使用通道相关性正则化（鼓励通道预测独立性）
        # 计算每个样本在通道维度上的均值
        # pred_mean = torch.mean(pred_prob, dim=(2, 3), keepdim=True)
        # channel_reg = torch.mean(torch.sum((pred_prob - pred_mean)**2, dim=1))

        # 损失权重
        alpha = 10
        beta = 1
        gamma = 1  # 正则化项权重

        # 组合损失
        sem_loss['cls_ce_loss'] = alpha * sem_ce_loss
        sem_loss['cls_dice_loss'] = beta * sem_dice_loss
        # sem_loss['cls_channel_reg'] = gamma * (l2_reg + channel_reg)

        return sem_loss

    
    
    def _ortho_loss(self, pred, gt, num_samples=1000, rare_labels=(2, 3, 4), radius=15, device="cuda"):
        """
        计算基于公式的距离约束损失。
        
        参数：
        - pred: 预测输出的概率值（形状：[N, C, H, W]）
        - gt: 实例分割的真实标签（形状：[N, H, W]）
        - num_samples: 每次采样的样本点对数量
        - rare_labels: 稀有颜色的标签值（例如 [2, 3, 4]）
        - radius: 邻域采样的范围，用于构造样本点对
        - device: 设备（默认为 "cuda"）
        
        返回：
        - total_loss: 基于公式计算的总损失
        """
        N, C, H, W = pred.shape
        gt_enc = F.one_hot(gt, num_classes=5).float().permute(0, 3, 1, 2)[:,1:,...]
        # 1. 筛选稀有颜色的区域
        rare_mask = torch.zeros_like(gt, dtype=torch.bool, device=device)
        for label in rare_labels:
            rare_mask |= (gt == label)

        # 如果没有稀有颜色区域，返回零损失
        if not rare_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 2. 获取稀有颜色像素的索引并随机采样
        rare_indices = torch.nonzero(rare_mask, as_tuple=False)  # 稀有颜色的位置
        if rare_indices.size(0) < num_samples:  # 如果稀有点不足，则使用全部
            sampled_indices = rare_indices
        else:
            sampled_indices = rare_indices[torch.randint(0, rare_indices.size(0), (num_samples,))]

        total_loss = 0.0

        # 3. 遍历每个采样点
        for idx in sampled_indices:
            n, h, w = idx
            h_min, h_max = max(h - radius, 0), min(h + radius + 1, H)
            w_min, w_max = max(w - radius, 0), min(w + radius + 1, W)

            # 中心点预测值和坐标
            pred_center = pred[n, :, h, w]
            gt_center = gt_enc[n, :, h, w]
            center_coords = torch.tensor([h, w], dtype=torch.float32, device=device)

            # 提取邻域范围内的预测值和坐标
            pred_patch = pred[n, :, h_min:h_max, w_min:w_max]
            gt_patch = gt[n, h_min:h_max, w_min:w_max]
            gt_enc_patch = gt_enc[n, :, h_min:h_max, w_min:w_max]

            # 计算邻域范围内的像素相对坐标
            neighbor_coords = torch.stack(torch.meshgrid(
                torch.arange(h_min, h_max, device=device),
                torch.arange(w_min, w_max, device=device)
            ), dim=-1).reshape(-1, 2)  # [K, 2]

            # 提取非背景点的预测值和坐标
            neighbor_mask = (gt_patch != 0)  # 忽略背景点
            neighbor_coords = neighbor_coords[neighbor_mask.view(-1)]
            pred_neighbors = pred_patch[:, neighbor_mask]
            gt_neighbors = gt_enc_patch[:, neighbor_mask]

            if neighbor_coords.size(0) == 0:  # 如果邻域内没有前景点，跳过
                continue

            # 计算邻域范围内的欧几里得距离 d(i, j)
            distances = torch.norm(neighbor_coords - center_coords, dim=-1)  # [K]

            # 计算公式中的 |Pi - Pj|
            diff_pred = torch.norm(pred_neighbors.T - pred_center.unsqueeze(0), dim=-1)  # [K]
            diff_gt = torch.norm(gt_neighbors.T - gt_center.unsqueeze(0), dim=-1) 
            # print("diff pred", diff_pred.shape)
            # print("diff gt", diff_gt.shape)
            # print("gt neighbors", gt_neighbors.shape)
            # print("gt center", gt_center.shape)
            

            # 公式中的损失函数：|Pi - Pj| / d(i, j)
            loss = torch.norm(diff_pred - diff_gt).mean() #/ (distances + 1e-6))
            # print("distance", distances.shape)
            # print("loss shape", loss)
            # print("*" * 100)
            # .mean()  # 避免除零
            # print("loss", loss.mean()/loss.size(0), sampled_indices.size(0) )
            total_loss += loss

        # 4. 对所有样本对的损失进行归一化
        total_loss = total_loss / sampled_indices.size(0)
        return {"ortho_loss": total_loss}
    
    def _dis_loss(self, hv_logit, hv_gt, fore_gt):
        hv_loss = {}
        hv_mse_loss_calculator = nn.MSELoss()
        hv_msge_loss_calculator = GradientMSELoss()
        # print("hv logit", hv_logit.shape)
        # print("hv gt", hv_gt.shape)
        hv_mse_loss = hv_mse_loss_calculator(hv_logit, hv_gt)
        hv_msge_loss = hv_msge_loss_calculator(hv_logit, hv_gt, fore_gt)
        # loss weight
        alpha = 1.0
        beta = 1.0
        hv_loss['hv_mse_loss'] = alpha * hv_mse_loss
        hv_loss['hv_msge_loss'] = beta * hv_msge_loss

        return hv_loss

    # def distance_based_loss(self, pred, gt, num_samples=5000):
    #     B, C, H, W = pred.shape
    #     total_loss = 0.0
        
    #     gt_enc = F.one_hot(gt, num_classes=5).float().permute(0, 3, 1, 2)[:,1:,...]
    #     # print("pred", pred.shape)
    #     # print("gt", gt_enc.shape)

    #     for b in range(B):
    #         foreground_mask = (gt[b] != 0)
    #         foreground_indices = torch.nonzero(foreground_mask, as_tuple=False)

    #         if foreground_indices.size(0) < 2:
    #             continue
    #         idx = torch.randint(0, foreground_indices.size(0), (num_samples, 2))
    #         idx1, idx2 = foreground_indices[idx[:,0]], foreground_indices[idx[:,1]]
    #         x1, y1 = idx1[:, 0], idx1[:, 1]
    #         x2, y2 = idx2[:, 0], idx2[:, 1]

    #         # idx1 = torch.randint(0, H*W, (num_samples))
    #         # idx2 = torch.randint(0, H*W, (num_samples))

    #         # x1, y1 = idx1 // W, idx1 % W
    #         # x2, y2 = idx2 // W, idx2 % W

    #         pred_i = pred[b,:, x1, y1]
    #         pred_j = pred[b,:, x2, y2]
    
    #         gt_i = gt_enc[b, :, x1, y1]
    #         gt_j = gt_enc[b, :, x2, y2]

    #         pred_diff = torch.abs(pred_i - pred_j)
    #         gt_diff = torch.abs(gt_i - gt_j)

    #         distances = torch.sqrt((x1-x2).float() ** 2 + (y1-y2).float() ** 2)
    #         loss_per_pair = torch.abs(pred_diff - gt_diff) / (distances+1e-6)
    #         print("loss per pair", loss_per_pair.shape, loss_per_pair.mean())

    #         total_loss += loss_per_pair.mean()

    #     return {"ortho_loss": total_loss/B}




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


                        
