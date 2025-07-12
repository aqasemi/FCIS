import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class BatchMultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(BatchMultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self,- input, target, weights=None):
        """
        Forward pass
        :param input: torch.Tensor, shape (N, C, H, W)
        :param target: torch.Tensor, shape (N, H, W)
        :param weights: torch.Tensor, shape (C)
        :return:
        """
        smooth = 1e-5
        input = F.softmax(input, dim=1)

        # one-hot encode target
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if weights is None:
            weights = torch.ones(self.num_classes).to(input.device)

        # compute dice loss for each class
        dice_class = torch.zeros(self.num_classes).to(input.device)
        for i in range(self.num_classes):
            intersection = torch.sum(input[:, i] * target_one_hot[:, i])
            union = torch.sum(input[:, i]) + torch.sum(target_one_hot[:, i])
            dice_class[i] = (2 * intersection + smooth) / (union + smooth)

        # compute weighted dice loss
        dice_loss = 1 - torch.sum(weights * dice_class) / torch.sum(weights)
        return dice_loss

class GradientMSELoss(nn.Module):
    def __init__(self):
        super(GradientMSELoss, self).__init__()

    def forward(self, pred, gt, fore_gt):
        # compute gradient of pred and gt
        pred_grad_x = F.conv2d(pred, weight=torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float32).to(pred.device), padding=1)
        pred_grad_y = F.conv2d(pred, weight=torch.tensor([[[[-1], [0], [1]]]], dtype=torch.float32).to(pred.device), padding=1)
        gt_grad_x = F.conv2d(gt, weight=torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float32).to(gt.device), padding=1)
        gt_grad_y = F.conv2d(gt, weight=torch.tensor([[[[-1], [0], [1]]]], dtype=torch.float32).to(gt.device), padding=1)

        # compute mse loss
        loss_x = F.mse_loss(pred_grad_x, gt_grad_x, reduction='none')
        loss_y = F.mse_loss(pred_grad_y, gt_grad_y, reduction='none')
        loss = (loss_x + loss_y) * fore_gt

        return loss.mean()

class OrthoLoss(nn.Module):
    def __init__(self, inst_gt, adj_gt, sem_gt, inter_pred, sample_ratio=0.7):
        super(OrthoLoss, self).__init__()
        self.inst_gt = inst_gt
        self.adj_gt = adj_gt
        self.sem_gt = sem_gt
        self.inter_pred = inter_pred
        self.sample_ratio = sample_ratio

    def calculate_ortho_loss(self):
        # Implementation of the OrthoLoss
        # This is a simplified version. The actual implementation might be more complex.

        loss = 0
        for i in range(self.inst_gt.shape[0]):
            inst_gt = self.inst_gt[i]
            adj_gt = self.adj_gt[i]
            sem_gt = self.sem_gt[i]
            inter_pred = self.inter_pred[i]

            inst_list = list(torch.unique(inst_gt))
            if 0 in inst_list:
                inst_list.remove(0)

            for inst_id in inst_list:
                if inst_id not in adj_gt:
                    continue

                neighbors = adj_gt[inst_id]
                if not neighbors:
                    continue

                inst_mask = (inst_gt == inst_id)
                neighbor_mask = torch.zeros_like(inst_gt).bool()
                for neighbor_id in neighbors:
                    neighbor_mask |= (inst_gt == neighbor_id)

                # Sample pixels from the instance and its neighbors
                inst_pixels = torch.where(inst_mask)
                neighbor_pixels = torch.where(neighbor_mask)

                if len(inst_pixels[0]) == 0 or len(neighbor_pixels[0]) == 0:
                    continue

                num_inst_pixels = int(len(inst_pixels[0]) * self.sample_ratio)
                num_neighbor_pixels = int(len(neighbor_pixels[0]) * self.sample_ratio)

                inst_indices = random.sample(range(len(inst_pixels[0])), num_inst_pixels)
                neighbor_indices = random.sample(range(len(neighbor_pixels[0])), num_neighbor_pixels)

                inst_features = inter_pred[:, inst_pixels[0][inst_indices], inst_pixels[1][inst_indices]]
                neighbor_features = inter_pred[:, neighbor_pixels[0][neighbor_indices], neighbor_pixels[1][neighbor_indices]]

                # Compute orthogonality loss
                dot_product = torch.matmul(inst_features.T, neighbor_features)
                ortho_loss = torch.mean(dot_product**2)
                loss += ortho_loss

        return {"ortho_loss": loss / self.inst_gt.shape[0]}
