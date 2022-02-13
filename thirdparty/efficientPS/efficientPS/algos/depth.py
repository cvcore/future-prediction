import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientPS.utils.parallel import PackedSequence
from efficientPS.utils.optical_flow_ops import *


class DepthAlgo:
    def __init__(self, depth_loss):
        self.depth_loss = depth_loss

    def training(self, head, ms_feat, depth_gt, iseg_object=None, iseg_boundary=None, sem_class_mask=None):
        # Run the head
        class_depth, depth_pred = head(ms_feat, iseg_object, iseg_boundary)

        # Compute the loss
        depth_loss = self.depth_loss(depth_pred, depth_gt, class_depth=class_depth, iseg_object_mask=iseg_object, iseg_boundary_mask=iseg_boundary, sem_class_mask=sem_class_mask)

        return class_depth, depth_pred, depth_loss

    def inference(self, head, ms_feat):
        # Run the head
        ms_depth, depth_pred = head(ms_feat)

        # Return the flow map and the flow confidence
        return ms_depth, depth_pred


class DepthLoss:
    def __init__(self, var_wt, alpha, dataset, object_wt=100, boundary_wt=10, bts_wt=10):
        self.var_wt = var_wt
        self.alpha = alpha  # alpha is the scaling constant according to the paper
        self.dataset = dataset

        self.object_wt = object_wt
        self.boundary_wt = boundary_wt
        self.bts_wt = bts_wt
        self.distance_attention_wt = 10
        self.class_wt = 10

        if dataset in ["Cityscapes", "CityscapesSample", "KittiRawDepth", "KittiDepthValSelection"]:
            # To compute the gradient of the predicted depth image
            self.sobel_x = [[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]]
            self.sobel_y = [[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]

        print("OBJECT WEIGHT: ", self.object_wt)
        print("BOUNDARY WEIGHT: ", self.boundary_wt)
        print("BTS WEIGHT: ", self.bts_wt)

    def __call__(self, depth_pred, depth_gt, mask=None, class_depth=None, sem_class_mask=None, iseg_boundary_mask=None, iseg_object_mask=None):
        """
        Depth loss

        Parameters
        ----------
        depth_pred : torch.Tensor
            The depth predicted by the network
        flow_gt : torch.Tensor
            A tensor with shape B x 1 x H x W containing the true depth of the image

        Returns
        -------
        of_loss : torch.Tensor
            A scalar tensor with the loss
        """
        losses = {}

        if self.dataset == "NYUv2":
            mask = depth_gt > 0.1
        elif self.dataset in ["Cityscapes", "CityscapesSample", "CityscapesDepth", "CityscapesDepthSample"]:
            B, C, H, W = depth_gt.size()
            # edge_size = 50
            # edge_mask = torch.ones_like(depth_gt, dtype=torch.bool)
            # edge_mask[:, :, :edge_size, :] = False
            # edge_mask[:, :, H - edge_size - 1:, :] = False
            # edge_mask[:, :, :, :edge_size] = False
            # edge_mask[:, :, :, W - edge_size - 1:] = False

            # mask = torch.logical_and(depth_gt > 1.0, edge_mask)
            mask = torch.logical_and(depth_gt > 0.1, depth_gt < 100.)
            max_range = 100.
            # print("Mask", mask.shape, torch.sum(mask))

        # print(torch.isnan(depth_pred).any(), torch.isnan(depth_gt).any(), torch.isnan(iseg_boundary_mask).any(), torch.isnan(iseg_object_mask).any())
        # print(depth_pred.shape, depth_gt.shape, iseg_boundary_mask.shape, iseg_object_mask.shape)

        elif self.dataset in ["KittiRawDepth", "KittiDepthValSelection", "KittiPanoptic"]:
            mask = torch.logical_and(depth_gt > 0.1, depth_gt < 80.)
            max_range = 80.

        # View the grads in the backward pass
        def print_grad(name):
            def hook(grad):
                print("{}------: {}".format(name, grad.mean()))
            return hook

        # depth_pred.register_hook(print_grad("Gradient"))

        # Compute the predicted depth gradient using the Sobel filter and compute the regularisation term
        # The second derivative of the objects should be 0 for them to be smooth. Minimize the second derivative
        if iseg_object_mask is not None:
            # print(torch.max(iseg_object_mask))
            # obj_smooth_depth_reg = self.computeSmoothDepthRegularisation(depth_pred, iseg_object_mask)
            # losses["object"] = self.object_wt * obj_smooth_depth_reg
            losses['object'] = torch.tensor(0.).to(iseg_object_mask.device)
            # losses['object'].register_hook(print_grad("Object"))

        # if mask is not None:
            # depth_pred = depth_pred * mask
            # depth_gt = depth_gt * mask
            # if iseg_boundary_mask is not None:
            #     iseg_boundary_mask = iseg_boundary_mask[mask]
            # if class_depth is not None:
            #     mask_cat = torch.cat(class_depth.shape[1] * [mask], dim=1)
            #     class_depth = class_depth * mask_cat

        # Compute the weighted boundary loss
        if iseg_boundary_mask is not None:
            # boundary_loss = self.computeBoundaryLoss(depth_pred, depth_gt, iseg_boundary_mask)
            # losses['boundary'] = self.boundary_wt * boundary_loss
            losses['boundary'] = torch.tensor(0.).to(iseg_object_mask.device)

        # Compute the BTS loss
        bts_loss = self.computeBTSLoss(depth_pred[mask], depth_gt[mask])
        # bts_loss = self.computeDistanceAwareBerhuLoss(depth_pred, depth_gt, max_range)
        losses['bts'] = self.bts_wt * bts_loss

        # Compute the class-based depth loss
        if class_depth is not None and sem_class_mask is not None:
            class_loss = self.computeClassDepthLoss(class_depth, sem_class_mask, depth_gt, mask)
            losses['class'] = self.bts_wt * class_loss
        else:
            losses['class'] = torch.tensor(0.).to(depth_gt.device)

        # Compute the normal scale invariant loss
        si_loss = self.computeScaleInvariantLoss(depth_pred[mask], depth_gt[mask])
        losses["si"] = si_loss

        # print(obj_smooth_depth_reg, boundary_loss, si_loss)

        # if (iseg_boundary_mask is not None) and (iseg_object_mask is not None):
        #     # losses['total'] = self.bts_wt * bts_loss + self.boundary_wt * boundary_loss + self.object_wt * obj_smooth_depth_reg
        #     losses['total'] = self.bts_wt * bts_loss
        # else:
        # losses['total'] = si_loss

        if class_depth is not None and sem_class_mask is not None:
            losses['total'] = self.bts_wt * bts_loss + self.bts_wt * class_loss
        else:
            losses['total'] = self.bts_wt * si_loss

        for loss_type, loss_value in losses.items():
            losses[loss_type] = loss_value.reshape(1)
            print(losses[loss_type])
        return losses

    def computeBoundaryLoss(self, depth_pred, depth_gt, iseg_boundary_mask):
        depth_pred_boundary = torch.mul(depth_pred, iseg_boundary_mask)
        depth_gt_boundary = torch.mul(depth_gt, iseg_boundary_mask)
        mask = torch.logical_and(depth_gt_boundary > 1e-4, depth_pred_boundary > 1e-4)
        if torch.sum(mask) == 0:
            return torch.tensor(0.).to(depth_pred.device)

        boundary_loss = self.computeMSELoss(depth_pred_boundary[mask], depth_gt_boundary[mask])
        # boundary_loss = self.computeL1Loss(depth_pred_boundary[mask], depth_gt_boundary[mask])
        return boundary_loss

    def computeSmoothDepthRegularisation(self, depth_pred, iseg_object_mask):
        grad_x, grad_y = self.computeGradient(depth_pred)  # Compute the first derivative
        grad_xx, grad_xy = self.computeGradient(grad_x)  # Compute the second derivative
        grad_yx, grad_yy = self.computeGradient(grad_y)  # Compute the second derivative

        # depth_pred_first_derivative = (grad_x.pow(2) + grad_y.pow(2)).sqrt()
        # Compute the Frobenius norm of the second derivative
        depth_pred_second_derivative = (grad_xx.pow(2) + grad_yy.pow(2) + grad_xy.pow(2) + grad_yx.pow(2)).sqrt()

        # depth_pred_second_derivative = self.computeGradient(depth_pred_first_derivative)  # Compute the second derivative
        depth_pred_second_derivative = torch.mul(depth_pred_second_derivative, iseg_object_mask)
        object_mask = depth_pred_second_derivative > 0
        num_elements = torch.sum(object_mask)
        if num_elements > 0:
            # L2 regularisation of the body smoothness
            reg = (depth_pred_second_derivative[object_mask].pow(2).sum().sqrt()) / num_elements
            # reg = torch.abs(depth_pred_grad[object_mask]).sum() / num_elements
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

    def computeMSELoss(self, depth_pred, depth_gt):
        return (depth_pred - depth_gt).pow(2).mean()

    def computeL1Loss(self, depth_pred, depth_gt):
        return torch.abs(depth_pred - depth_gt).mean()

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

    def computeClassDepthLoss(self, class_depth, sem_seg_mask, depth_gt, valid_mask):
        sem_class_count = sem_seg_mask.shape[1]
        depth_gt_class = torch.cat(sem_class_count * [depth_gt], dim=1)

        valid_mask_cat = torch.cat(sem_class_count * [valid_mask], dim=1)
        non_zero_mask = class_depth > 0
        total_mask = torch.logical_and(torch.logical_and(sem_seg_mask, valid_mask_cat), non_zero_mask)

        class_loss = self.computeScaleInvariantLoss(class_depth[total_mask], depth_gt_class[total_mask])

        return class_loss


