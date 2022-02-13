import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from efficientPS.utils.parallel import PackedSequence
from efficientPS.utils.sequence import pad_packed_images
from efficientPS.utils.optical_flow_ops import *


class PanopticDepthAlgo:
    def __init__(self, depth_loss):
        self.depth_loss = depth_loss

    def training(self, sem_depth_head, ms_feat, sem_feat, depth_gt, po_mask=None, sem_class_mask=None, img=None):
        # Run the semantic and instance heads here
        sem_depth_feat, sem_class_depth_pred, sem_depth_pred = sem_depth_head(ms_feat, sem_feat)
        # inst_depth_feat = inst_depth_head(inst_feat)
        # depth_pred = depth_fusion(sem_depth_feat, inst_depth_feat)

        # Compute the loss
        po_mask, _ = pad_packed_images(po_mask)
        po_mask = po_mask.type(torch.float)
        bts_loss, class_loss, panoptic_edge_loss, depth_stats = self.depth_loss(sem_depth_pred, sem_class_depth_pred, depth_gt, po_mask=po_mask, sem_class_mask=sem_class_mask, img=img)

        return sem_class_depth_pred, sem_depth_pred, bts_loss, class_loss, panoptic_edge_loss, depth_stats

    def inference(self, sem_depth_head, ms_feat, sem_feat):
        # Run the head
        sem_depth_feat, sem_class_depth_pred, sem_depth_pred = sem_depth_head(ms_feat, sem_feat)

        # Return the depth
        return sem_class_depth_pred, sem_depth_pred


class PanopticDepthLoss:
    def __init__(self, var_wt, alpha, dataset, bts_wt=10, po_edge_wt=10):
        self.var_wt = var_wt
        self.alpha = alpha  # alpha is the scaling constant according to the paper
        self.dataset = dataset

        self.bts_wt = bts_wt
        self.po_edge_wt = po_edge_wt

        self.edge_mask_alpha = 0.1
        self.nbd_max_limit = 15
        self.nbd_min_limit = 5
        self.pointwise_thresh = 0.03

        self.sobel_x = [[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]]
        self.sobel_y = [[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]]

    def __call__(self, depth_pred, class_depth_pred, depth_gt, po_mask=None, sem_class_mask=None, img=None):
        # Record all the losses here
        stats = {}
        depth_gt, input_sizes = pad_packed_images(depth_gt)

        if self.dataset in ["Cityscapes", "CityscapesSample", "CityscapesDepth", "CityscapesDepthSample", "CityscapesSeam"]:
            mask = (depth_gt > 0.1) & (depth_gt < 100.)
        elif self.dataset in ["KittiRawDepth", "KittiDepthValSelection", "KittiPanoptic"]:
            mask = (depth_gt > 0.1) & (depth_gt < 80.)
        else:
            raise NotImplementedError()

        # View the grads in the backward pass
        def print_grad(name):
            def hook(grad):
                print("{}------: {}".format(name, grad.mean()))
            return hook

        # depth_pred.register_hook(print_grad("Gradient"))
        # print("mask", torch.sum(mask))

        # Compute the BTS loss
        bts_loss = self.bts_wt * self.computeBTSLoss(depth_pred[mask], depth_gt[mask])
        # bts_loss = self.computeDistanceAwareBerhuLoss(depth_pred, depth_gt, max_range)
        stats['depth_bts'] = bts_loss

        # Compute the Class loss
        class_loss = self.bts_wt * self.computeClassDepthLoss(class_depth_pred, sem_class_mask=sem_class_mask, depth_gt=depth_gt, valid_mask=mask)
        stats["depth_class"] = class_loss

        # Compute the panoptic loss
        panoptic_edge_loss = self.po_edge_wt * self.computePanopticEdgeLoss(depth_pred, depth_gt, po_mask, valid_mask=mask)
        stats['depth_po_edge'] = panoptic_edge_loss

        # Compute the normal scale invariant loss
        si_loss = self.computeScaleInvariantLoss(depth_pred[mask], depth_gt[mask])
        stats["depth_si"] = si_loss

        # depth_loss = self.bts_wt * bts_loss + self.bts_wt * class_loss + self.po_edge_wt * panoptic_edge_loss
        # depth_loss = depth_loss.reshape(1)

        for stat_type, stat_value in stats.items():
            stats[stat_type] = stat_value.reshape(1)

        return bts_loss, class_loss, panoptic_edge_loss, stats

    def computeSmoothDepthRegularisation(self, depth_pred, iseg_object_mask):
        grad_x, grad_y = self.computeGradient(depth_pred)  # Compute the first derivative
        grad_xx, grad_xy = self.computeGradient(grad_x)  # Compute the second derivative
        grad_yx, grad_yy = self.computeGradient(grad_y)  # Compute the second derivative

        # Compute the Frobenius norm of the second derivative
        depth_pred_second_derivative = (grad_xx.pow(2) + grad_yy.pow(2) + grad_xy.pow(2) + grad_yx.pow(2)).sqrt()

        depth_pred_second_derivative = torch.mul(depth_pred_second_derivative, iseg_object_mask)
        object_mask = depth_pred_second_derivative > 0
        num_elements = torch.sum(object_mask)
        if num_elements > 0:
            # L2 regularisation of the body smoothness
            reg = (depth_pred_second_derivative[object_mask].pow(2).sum().sqrt()) / num_elements
        else:
            reg = torch.tensor(0.).to(depth_pred.device)
        return reg

    def computeGradient(self, depth_pred):
        _, C, _, _ = depth_pred.shape

        grad_x_wts = torch.nn.Parameter(torch.tensor(self.sobel_x, dtype=torch.float32, requires_grad=False).expand(1, C // 1, 3, 3).to(depth_pred.device))  # (out_ch, in_ch // groups, H_K, W_K)
        grad_y_wts = torch.nn.Parameter(torch.tensor(self.sobel_y, dtype=torch.float32, requires_grad=False).expand(1, C // 1, 3, 3).to(depth_pred.device))  # (out_ch, in_ch // groups, H_K, W_K)
        sobel_conv_x = nn.Conv2d(1, 1, 3, 1, 1, padding_mode="replicate").to(depth_pred.device)
        sobel_conv_y = nn.Conv2d(1, 1, 3, 1, 1, padding_mode="replicate").to(depth_pred.device)
        sobel_conv_x.weight = grad_x_wts
        sobel_conv_y.weight = grad_y_wts

        grad_x = sobel_conv_x(depth_pred)
        grad_y = sobel_conv_y(depth_pred)

        return grad_x, grad_y

    def computeScaleInvariantLoss(self, depth_pred, depth_gt):
        log_error = torch.log(depth_pred) - torch.log(depth_gt)
        si_loss = (log_error ** 2).mean() - (log_error.mean()) ** 2
        return si_loss

    def computeScaleInvariantRegularisation(self, depth_pred):
        EPS = 1e-6
        log_error = torch.log(depth_pred + EPS)
        si_loss = (log_error ** 2).mean() - (log_error.mean()) ** 2
        return si_loss

    def computeMSELoss(self, depth_pred, depth_gt):
        return (depth_pred - depth_gt).pow(2).mean()

    def computeL1Loss(self, depth_pred, depth_gt):
        return torch.abs(depth_pred - depth_gt).mean()

    def computeL1Regularisation(self, depth_pred):
        return torch.abs(depth_pred).mean()

    def computeBTSLoss(self, depth_pred, depth_gt):
        log_error = torch.log(depth_pred) - torch.log(depth_gt)
        bts_loss = torch.sqrt((log_error ** 2).mean() - self.var_wt * (log_error.mean() ** 2))
        return bts_loss

    def computeDistanceAwareBerhuLoss(self, depth_pred, depth_gt, max_range):
        abs_error = torch.abs(depth_pred - depth_gt)
        c = 0.2 * torch.max(abs_error)

        false_condition_error = ((depth_pred - depth_gt).pow(2) + c.pow(2)) / (2 * c)
        berhu_loss_tensor = torch.where(abs_error <= c, abs_error, false_condition_error)

        # Concatenate the pred and gt depth maps to compute the maxima and minima easily
        cat_tensor = torch.cat([depth_pred.unsqueeze(1), depth_gt.unsqueeze(1)], dim=1)
        reg_tensor = torch.ones_like(depth_pred) - (torch.min(torch.log(cat_tensor), dim=1)[0] / torch.max(torch.log(cat_tensor), dim=1)[0])
        weight_tensor = (depth_gt / max_range) + reg_tensor
        weighted_berhu_loss = torch.mean(weight_tensor * berhu_loss_tensor)

        return weighted_berhu_loss

    def computeClassDepthLoss(self, class_depth, sem_class_mask, depth_gt, valid_mask):
        sem_class_count = sem_class_mask.shape[1]
        depth_gt_class = torch.cat(sem_class_count * [depth_gt], dim=1)
        # depth_gt_class[~sem_class_mask] = 0.1


        valid_mask_cat = torch.cat(sem_class_count * [valid_mask], dim=1)
        non_zero_mask = class_depth > 0
        total_class_mask = (sem_class_mask & valid_mask_cat) & non_zero_mask
        total_non_class_mask = (~sem_class_mask & valid_mask_cat) & non_zero_mask
        # total_mask = valid_mask_cat & non_zero_mask

        class_loss = self.computeScaleInvariantLoss(class_depth[total_class_mask], depth_gt_class[total_class_mask])
        non_class_loss = self.computeScaleInvariantRegularisation(class_depth[total_non_class_mask])

        class_loss = class_loss + non_class_loss

        return class_loss

    def computePanopticEdgeLoss(self, depth_pred, depth_gt, po_mask, valid_mask=None):
        # Compute the instance mask
        po_edge_map, grad_x, grad_y, grad_mag = self.computePanopticEdgeMap(po_mask)

        # Sample points from the instance edges and neighbouring points
        po_edge_points = torch.nonzero(po_edge_map, as_tuple=True)
        points_a, points_b = self.getNeighbourhoodPointPairs(po_edge_map, po_edge_points, grad_x, grad_y, grad_mag)

        # Remove samples where the GT is not valid --> valid_mask = False
        points_valid = valid_mask[points_a] & valid_mask[points_b]
        points_a_valid = (points_a[0][points_valid], points_a[1][points_valid], points_a[2][points_valid], points_a[3][points_valid])
        points_b_valid = (points_b[0][points_valid], points_b[1][points_valid], points_b[2][points_valid], points_b[3][points_valid])

        # Handle the case when the valid mask has no valid points
        if points_a_valid[0].shape[0] == 0:
            return torch.tensor(0.).to(depth_pred.device)

        # Sample the indices from the set of points
        sampled_indices = torch.randint(0, points_a_valid[0].shape[0], (po_edge_points[0].shape))
        points_a_sampled = (points_a_valid[0][sampled_indices], points_a_valid[1][sampled_indices], points_a_valid[2][sampled_indices], points_a_valid[3][sampled_indices])
        points_b_sampled = (points_b_valid[0][sampled_indices], points_b_valid[1][sampled_indices], points_b_valid[2][sampled_indices], points_b_valid[3][sampled_indices])

        # Compute the edge loss
        edge_loss = self.computePairwiseLoss(depth_pred, depth_gt, points_a_sampled, points_b_sampled)

        return edge_loss

    def computePanopticEdgeMap(self, po_mask):
        # Apply Sobel filter to get the X and Y gradients
        grad_x, grad_y = self.computeGradient(po_mask)
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))

        # Compute the instance edge map
        B, C, H, W = po_mask.shape
        po_edge_map = torch.zeros((B, C, H, W), dtype=torch.bool).to(po_mask.device)
        for b in range(po_edge_map.shape[0]):
            po_edge_map[b, :, :, :] = grad_mag[b, :, :, :] > 0.5  # This implies change of label.

            # Zero out the edges of the image to prevent computation wastage
            po_edge_map[b, :, :11, :] = 0
            po_edge_map[b, :, H - 10:, :] = 0
            po_edge_map[b, :, :, :11] = 0
            po_edge_map[b, :, :, W - 10:] = 0

        return po_edge_map, grad_x, grad_y, grad_mag

    def computeEdgeLoss(self, depth_pred, depth_gt, img, valid_mask=None):
        # Compute the edge map
        edge_map, grad_x, grad_y, grad_mag = self.computeEdgeMap(img)

        # Sample edge and neighbouring points
        edge_points = torch.nonzero(edge_map, as_tuple=True)
        points_a, points_b = self.getNeighbourhoodPointPairs(edge_map, edge_points, grad_x, grad_y, grad_mag)

        # Remove samples where the GT is not valid --> valid_mask = False
        points_valid = valid_mask[points_a] & valid_mask[points_b]
        points_a_valid = (points_a[0][points_valid], points_a[1][points_valid], points_a[2][points_valid], points_a[3][points_valid])
        points_b_valid = (points_b[0][points_valid], points_b[1][points_valid], points_b[2][points_valid], points_b[3][points_valid])

        # Handle the case when the valid mask has no valid points
        if points_a_valid[0].shape[0] == 0:
            return torch.tensor(0.).to(depth_pred.device)

        # Sample the indices from the set of points
        sampled_indices = torch.randint(0, points_a_valid[0].shape[0], (edge_points[0].shape))
        points_a_sampled = (points_a_valid[0][sampled_indices], points_a_valid[1][sampled_indices], points_a_valid[2][sampled_indices], points_a_valid[3][sampled_indices])
        points_b_sampled = (points_b_valid[0][sampled_indices], points_b_valid[1][sampled_indices], points_b_valid[2][sampled_indices], points_b_valid[3][sampled_indices])

        # Compute the edge loss
        edge_loss = self.computePairwiseLoss(depth_pred, depth_gt, points_a_sampled, points_b_sampled)

        return edge_loss


    def computeEdgeMap(self, img):
        # Grayscale = 0.299 * red + 0.587 * green + 0.114 * blue
        gray_img = (img[:, 0, :, :] * 0.299).unsqueeze(1) + (img[:, 1, :, :] * 0.587).unsqueeze(1) + (img[:, 2, :, :] * 0.114).unsqueeze(1)

        # Apply Sobel filter to get the X and Y gradients
        grad_x, grad_y = self.computeGradient(gray_img)
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))

        # Compute the edge map
        B, C, H, W = gray_img.shape
        edge_map = torch.zeros((B, C, H, W), dtype=torch.bool)
        for b in range(edge_map.shape[0]):
            edge_map[b, :, :, :] = grad_mag[b, :, :, :] > 0.1 * torch.max(grad_mag[b, :, :, :])

        return edge_map, grad_x, grad_y, grad_mag

    def getNeighbourhoodPointPairs(self, edge_map, edge_points, grad_x, grad_y, grad_mag):
        points_count = edge_points[0].shape[0]
        H_min = 0
        H_max = edge_map.shape[2] - 1
        W_min = 0
        W_max = edge_map.shape[3] - 1

        batch = edge_points[0]
        channel = edge_points[1]
        edge_points_x = edge_points[2]
        edge_points_y = edge_points[3]

        # batch_samp_a = torch.zeros((4 * points_count), dtype=torch.long).to(edge_points[0].device)
        # ch_samp_a = torch.zeros((4 * points_count), dtype=torch.long).to(edge_points[0].device)
        edge_x_samp_a = torch.zeros((4 * points_count), dtype=torch.long).to(edge_map.device)
        edge_y_samp_a = torch.zeros((4 * points_count), dtype=torch.long).to(edge_map.device)
        # batch_samp_b = torch.zeros((4 * points_count), dtype=torch.long).to(edge_points[0].device)
        # ch_samp_b = torch.zeros((4 * points_count), dtype=torch.long).to(edge_points[0].device)
        edge_x_samp_b = torch.zeros((4 * points_count), dtype=torch.long).to(edge_map.device)
        edge_y_samp_b = torch.zeros((4 * points_count), dtype=torch.long).to(edge_map.device)

        delta_a = torch.randint(-self.nbd_max_limit, -self.nbd_min_limit, torch.Size([points_count]), device=edge_map.device).unsqueeze(0)
        delta_b = torch.randint(-self.nbd_max_limit, -self.nbd_min_limit, torch.Size([points_count]), device=edge_map.device).unsqueeze(0)
        delta_c = torch.randint(self.nbd_min_limit, self.nbd_max_limit, torch.Size([points_count]), device=edge_map.device).unsqueeze(0)
        delta_d = torch.randint(self.nbd_min_limit, self.nbd_max_limit, torch.Size([points_count]), device=edge_map.device).unsqueeze(0)
        delta = torch.cat([delta_a, delta_b, delta_c, delta_d], dim=0)

        batch_repeat = torch.repeat_interleave(batch, repeats=delta.shape[0], dim=0)
        channel_repeat = torch.repeat_interleave(channel, repeats=delta.shape[0], dim=0)
        batch_samp_a = batch_repeat
        ch_samp_a = channel_repeat
        batch_samp_b = batch_repeat
        ch_samp_b = channel_repeat

        grad_mag_pts = grad_mag[batch, channel, edge_points_x, edge_points_y]
        grad_x_pts = grad_x[batch, channel, edge_points_x, edge_points_y]
        grad_y_pts = grad_y[batch, channel, edge_points_x, edge_points_y]
        for sp_idx in range(delta.shape[0]):
            x_sample = edge_points_x + (delta[sp_idx] * grad_x_pts / grad_mag_pts).type(torch.long)
            y_sample = edge_points_y + (delta[sp_idx] * grad_y_pts / grad_mag_pts).type(torch.long)

            # Clamp the range of the indices
            x_sample[x_sample < H_min] = H_min
            x_sample[x_sample > H_max] = H_max
            y_sample[y_sample < W_min] = W_min
            y_sample[y_sample > W_max] = W_max

            sp_indices = torch.arange(0, 4 * points_count)

            edge_x_samp_a[(sp_indices % delta.shape[0]) == sp_idx] = x_sample
            edge_y_samp_a[(sp_indices % delta.shape[0]) == sp_idx] = y_sample

            edge_x_samp_b[(sp_indices % delta.shape[0]) == ((sp_idx + (delta.shape[0] - 1)) % delta.shape[0])] = x_sample
            edge_y_samp_b[(sp_indices % delta.shape[0]) == ((sp_idx + (delta.shape[0] - 1)) % delta.shape[0])] = y_sample

        return (batch_samp_a, ch_samp_a, edge_x_samp_a, edge_y_samp_a), (
        batch_samp_b, ch_samp_b, edge_x_samp_b, edge_y_samp_b)

    def computePairwiseLoss(self, depth_pred, depth_gt, points_a_sampled, points_b_sampled):
        depth_pred_a = depth_pred[points_a_sampled]
        depth_pred_b = depth_pred[points_b_sampled]
        depth_gt_a = depth_gt[points_a_sampled]
        depth_gt_b = depth_gt[points_b_sampled]

        plusone_tensor = torch.ones_like(depth_pred_a)
        minusone_tensor = torch.ones_like(depth_pred_a) * -1
        zero_tensor = torch.zeros_like(depth_pred_a)
        plus1_condition = depth_gt_a / depth_gt_b >= 1 + self.pointwise_thresh
        minus1_condition = depth_gt_a / depth_gt_b <= 1 / (1 + self.pointwise_thresh)
        l = torch.where(plus1_condition, plusone_tensor, torch.where(minus1_condition, minusone_tensor, zero_tensor))

        pairwise_loss = torch.where(l == zero_tensor,
                                    self.computeScaleInvariantLoss(depth_pred_a, depth_gt_b), # True
                                    torch.log(1 + torch.exp(-l * (torch.log(depth_pred_a) - torch.log(depth_pred_b))))).mean()
                                    # (depth_pred_a - depth_pred_b).pow(2),  # True
                                    # torch.log(1 + torch.exp(-l * (depth_pred_a - depth_pred_b)))).mean()

        return pairwise_loss
