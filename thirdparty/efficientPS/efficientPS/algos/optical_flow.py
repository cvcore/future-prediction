import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientPS.utils.parallel import PackedSequence
from efficientPS.utils.sequence import pad_packed_images
from efficientPS.utils.optical_flow_ops import *


class OpticalFlowAlgo:
    def __init__(self, optical_flow_loss):
        self.optical_flow_loss = optical_flow_loss

    def training(self, head, prev_ms_feat, curr_ms_feat, flow_gt):
        # Run the head
        ms_flow_prob, ms_flow_pred = head(prev_ms_feat, curr_ms_feat)

        # Compute the loss
        of_loss = self.optical_flow_loss(ms_flow_prob, ms_flow_pred, flow_gt)

        return ms_flow_prob, ms_flow_pred, of_loss

    def inference(self, head, prev_ms_feat, curr_ms_feat):
        # Run the head
        ms_flow_prob, ms_flow_pred = head(prev_ms_feat, curr_ms_feat)

        # Return the flow map and the flow confidence
        return ms_flow_prob, ms_flow_pred


class OpticalFlowAlgoMFN:
    def __init__(self, optical_flow_loss):
        self.optical_flow_loss = optical_flow_loss

    def training(self, head, prev_ms_feat, curr_ms_feat, prev_img, curr_img, flow_gt, mask):
        # Run the head
        ms_flow_pred, flow_vis = head(prev_ms_feat, curr_ms_feat, prev_img, curr_img)

        # Compute the loss
        of_loss = self.optical_flow_loss(ms_flow_pred, flow_gt, mask)

        return ms_flow_pred, flow_vis, of_loss

    def inference(self, head, prev_ms_feat, curr_ms_feat, prev_img, curr_img, mask):
        # Run the head
        ms_flow_pred, flow_vis = head(prev_ms_feat, curr_ms_feat, prev_img, curr_img, mask)

        # Return the flow map and the flow confidence
        return ms_flow_pred, flow_vis


class OpticalFlowAlgoMFN_S:
    def __init__(self, optical_flow_loss):
        self.optical_flow_loss = optical_flow_loss

    def training(self, head, prev_ms_feat, curr_ms_feat, prev_img, curr_img, flow_gt, mask):
        # Run the head
        ms_flow_pred, flow_vis = head(prev_ms_feat, curr_ms_feat)

        # Compute the loss
        of_loss = self.optical_flow_loss(ms_flow_pred, flow_gt, mask)

        return ms_flow_pred, flow_vis, of_loss

    def inference(self, head, prev_ms_feat, curr_ms_feat, prev_img, curr_img, mask):
        # Run the head
        ms_flow_pred, flow_vis = head(prev_ms_feat, curr_ms_feat, mask)

        # Return the flow map and the flow confidence
        return ms_flow_pred, flow_vis

class OpticalFlowAlgoMFNInstance:
    def __init__(self, optical_flow_loss):
        self.optical_flow_loss = optical_flow_loss

    def training(self, head, prev_ms_feat, curr_ms_feat, prev_img, curr_img, flow_gt, mask, points_2d, valid_2d, points_3d, valid_3d, points_flag, k):
        # Run the head
        ms_flow_pred, flow_vis, reproj_loss = head(prev_ms_feat, curr_ms_feat, prev_img, curr_img, flow_gt, points_2d, valid_2d, points_3d, valid_3d, points_flag, k)

        # Compute the loss
        of_loss = self.optical_flow_loss(ms_flow_pred, flow_gt, mask, reproj_loss)

        return ms_flow_pred, flow_vis, of_loss

    def inference(self, head, prev_ms_feat, curr_ms_feat, prev_img, curr_img, mask):
        # Run the head
        ms_flow_pred, flow_vis = head(prev_ms_feat, curr_ms_feat, prev_img, curr_img, mask)

        # Return the flow map and the flow confidence
        return ms_flow_pred, flow_vis

class OpticalFlowAlgoRAFT:
    def __init__(self, optical_flow_loss):
        self.optical_flow_loss = optical_flow_loss

    def training(self, head, prev_ms_feat, curr_ms_feat, flow_gt, valid_mask, flow_shape):
        # Run the head.
        # We get the intermediate GRU iteration prediction, the H/8 flow, and the upsampled flow
        flow_pred_iter, flow_pred, flow_pred_up = head(prev_ms_feat, curr_ms_feat, flow_shape=flow_shape)

        of_loss = self.optical_flow_loss(flow_pred_iter, flow_gt, valid_mask)

        return flow_pred_iter, flow_pred, flow_pred_up, of_loss



class OpticalFlowLoss:
    def __init__(self, corr_range, ds=6):
        self.corr_range = corr_range
        self.ds = ds

    def __call__(self, ms_flow_prob, ms_flow_pred, flow_gt):
        """
        Optical flow loss

        Parameters
        ----------
        ms_flow_prob : torch.Tensor
            Confidence map of the predicted optical flow in multiple scales
        ms_flow_pred : torch.Tensor
            Predicted multi-scale optical flow
        flow_gt : torch.Tensor
            A tensor with shape B x 3 x H x W containing the true flow of the image pair
        corr_range :
            The correlation range
        ds :
            The downsampling factor to the coarsest level

        Returns
        -------
        of_loss : torch.Tensor
            A scalar tensor with the loss
        """

        B, C, H, W = flow_gt.size()
        level_count = len(ms_flow_prob)
        criterion = nn.KLDivLoss(reduction="batchmean").cuda()
        losses = {}
        kld_loss = 0

        for l in range(level_count):
            scaled_flow_gt, valid_mask = downsample_flow(flow_gt, 1/2**(self.ds-l))
            if l > 0:
                scaled_flow_gt = scaled_flow_gt - F.interpolate(ms_flow_pred[l-1], scale_factor=2, mode='bilinear', align_corners=True, recompute_scale_factor=True)
            scaled_flow_gt = scaled_flow_gt / 2**(self.ds-l)

            scaled_flow_distribution = vector2density(scaled_flow_gt, self.corr_range[l]) * valid_mask
            kld_loss += 4**(self.ds-l) / (H * W) * criterion(F.log_softmax(ms_flow_prob[l], dim=1), scaled_flow_distribution.detach())

        losses['total'] = kld_loss
        for loss_type, loss_value in losses.items():
            losses[loss_type] = loss_value.reshape(1)
        return losses


class OpticalFlowLossMFN:
    def __init__(self, scales, weights, match="upsampling", eps=1e-8, q=None):
        self.scales = scales
        self.weights = weights
        self.match = match
        self.eps = eps
        self.q = q

        self.ms_epe = MultiScaleEPELoss(self.scales, self.weights, self.match, self.eps, self.q)

    def __call__(self, flow_pred_list, flow_gt, mask):
        losses = {}
        ms_epe_loss = self.ms_epe(flow_pred_list, flow_gt, mask)

        losses['total'] = ms_epe_loss
        # print(ms_epe_loss)
        # for loss_type, loss_value in losses.items():
        #     losses[loss_type] = loss_value.reshape(1)
        return losses


class OpticalFlowLossMFNInstance:
    def __init__(self, scales, weights, match="upsampling", eps=1e-8, q=None):
        self.scales = scales
        self.weights = weights
        self.match = match
        self.eps = eps
        self.q = q

        self.ms_epe = MultiScaleEPELoss(self.scales, self.weights, self.match, self.eps, self.q)

    def __call__(self, flow_pred_list, flow_gt, mask, reproj_loss):
        losses = {}
        ms_epe_loss = self.ms_epe(flow_pred_list, flow_gt, mask)

        losses["pose_error"] = reproj_loss['pose_error']
        # losses['reproj_pnp_gt'] = reproj_loss["pnp_gt"]
        # losses['reproj_pnp_pred'] = reproj_loss["pnp_pred"]

        losses['total'] = ms_epe_loss + (reproj_loss['pose_error'])


        # print(ms_epe_loss)
        # for loss_type, loss_value in losses.items():
        #     losses[loss_type] = loss_value.reshape(1)
        return losses


class EPELossWithMask:
    def __init__(self, eps=1e-8, q=None):
        super(EPELossWithMask, self).__init__()
        self.eps = eps
        self.q = q

    def __call__(self, flow_pred, flow_gt, mask):
        # flow_pred = flow_pred[mask]
        # flow_gt = flow_gt[mask]
        # print(torch.min(mask), torch.max(mask))
        nan_count = torch.sum(flow_pred != flow_pred)
        if nan_count > 0:
            print("NAN FOUND!: {}".format(nan_count))

        if self.q is not None:
            loss = ((flow_pred - flow_gt).abs().sum(1) + self.eps) ** self.q
        else:
            loss = ((flow_pred - flow_gt).pow(2).sum(1) + self.eps).sqrt()
        loss = loss * mask.squeeze(1)
        loss = loss.view(loss.shape[0], -1).sum(1) / mask.view(mask.shape[0], -1).sum(1)

        loss[loss != loss] = 0.

        return loss


class EPELoss:
    def __init__(self, eps=0):
        super(EPELoss, self).__init__()
        self.eps = eps

    def __call__(self, flow_pred, flow_gt):
        loss = ((flow_pred - flow_gt).pow(2).sum(1) + self.eps).sqrt()
        return loss.view(loss.shape[0], -1).mean(1)


class MultiScaleEPELoss:
    def __init__(self, scales, weights, match, eps=1e-8, q=None):
        super(MultiScaleEPELoss, self).__init__()
        self.scales = scales
        self.weights = weights
        self.match = match
        self.eps = eps
        self.q = q

        # self.epe_loss_with_mask = EPELossWithMask(eps=self.eps, q=self.q)

    def __call__(self, flow_pred_list, flow_gt, mask):
        losses = 0

        if self.match == "upsampling":
            for pred_level, weight_level, scale_level in zip(flow_pred_list, self.weights, self.scales):
                losses += EPELossWithMask(eps=self.eps, q=self.q)(Upsample(pred_level, scale_level), flow_gt, mask) * weight_level
                # losses += self.epe_loss_with_mask(Upsample(pred_level, scale_level), flow_gt, mask) * weight_level
        else:
            raise NotImplementedError

        return losses

def EndPointError(output, gt):
    # output: [B, 1/2, H, W], stereo or flow prediction
    # gt: [B, C, H, W], 2D ground-truth annotation which may contain a mask
    # NOTE: To benchmark the result, please ensure the ground-truth keeps
    # its ORIGINAL RESOLUTION.
    output = resize_dense_vector(output, gt.size(2), gt.size(3))
    error = torch.norm(output - gt[:, :2, :, :], 2, 1, keepdim=False)
    if gt.size(1) == 3:
        mask = (gt[:, 2, :, :] > 0).float()
    else:
        mask = torch.ones_like(error)
    epe = (error * mask).sum() / mask.sum()
    return epe.reshape(1)

class OpticalFlowLossRAFT:
    def __init__(self, max_flow=400, gamma=0.8, eps=1e-8):
        super(OpticalFlowLossRAFT, self).__init__()
        self.eps = eps
        self.max_flow = max_flow
        self.gamma = gamma

    def __call__(self, flow_pred_iter, flow_gt, valid_mask):
        losses = {}

        flow_gt, _ = pad_packed_images(flow_gt)
        valid_mask, _ = pad_packed_images(valid_mask)

        # Check for nan in predictions
        nan_count = 0
        for i in range(len(flow_pred_iter)):
            nan_count = nan_count + torch.sum(flow_pred_iter[i] != flow_pred_iter[i])
        if nan_count > 0:
            print("NAN FOUND!: {}".format(nan_count))

        # Remove the invalid pixels and the pixels with large displacements
        # mag = torch.sum(flow_gt ** 2, dim=1).sqrt().unsqueeze(1)
        # valid_mask = valid_mask & (mag < self.max_flow)

        flow_seq_loss = self.computeSequenceLoss(flow_pred_iter, flow_gt, valid_mask)
        # flow_edge_loss = self.computePanopticEdgeLoss(flow_pred_iter, flow_gt, valid_mask, po_mask)

        losses['total'] = flow_seq_loss

        return losses

    def computeSequenceLoss(self, flow_pred_iter, flow_gt, valid_mask):
        num_preds = len(flow_pred_iter)
        flow_loss = 0.0
        for i in range(num_preds):
            i_weight = self.gamma ** (num_preds - i - 1)
            i_loss = (flow_pred_iter[i] - flow_gt).abs()
            flow_loss += i_weight * (valid_mask * i_loss).mean()

        return flow_loss

    def computeInstanceEdgeLoss(self, flow_pred_iter, flow_gt, valid_mask, po_mask):
        # Get the instance outline mask
        inst_edge_map, grad_x, grad_y, grad_mag = self.computeInstanceEdgeMap(po_mask)

        # Sample points from the instance edges and neighbouring points
        inst_edge_points = torch.nonzero(inst_edge_map, as_tuple=True)
        points_a, points_b = self.getNeighbourhoodPointPairs(inst_edge_map, inst_edge_points, grad_x, grad_y, grad_mag)

        # Remove samples where the GT is not valid --> valid_mask = False
        points_valid = valid_mask[points_a] & valid_mask[points_b]
        points_a_valid = (points_a[0][points_valid], points_a[1][points_valid], points_a[2][points_valid], points_a[3][points_valid])
        points_b_valid = (points_b[0][points_valid], points_b[1][points_valid], points_b[2][points_valid], points_b[3][points_valid])

        # Handle the case when the valid mask has no valid points
        if points_a_valid[0].shape[0] == 0:
            return torch.tensor(0.).to(flow_gt.device)

        # Sample the indices from the set of points
        sampled_indices = torch.randint(0, points_a_valid[0].shape[0], (inst_edge_points[0].shape))
        points_a_sampled = (points_a_valid[0][sampled_indices], points_a_valid[1][sampled_indices], points_a_valid[2][sampled_indices], points_a_valid[3][sampled_indices])
        points_b_sampled = (points_b_valid[0][sampled_indices], points_b_valid[1][sampled_indices], points_b_valid[2][sampled_indices], points_b_valid[3][sampled_indices])

        # Compute the edge loss
        inst_edge_loss = self.computePairwiseLoss(flow_pred_iter, flow_gt, points_a_sampled, points_b_sampled)

        return inst_edge_loss

    def computeInstanceEdgeMap(self, po_mask):
        # Apply Sobel filter to get the X and Y gradients
        grad_x, grad_y = self.computeGradient(po_mask)
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))

        # Compute the instance edge map
        B, C, H, W = po_mask.shape
        inst_edge_map = torch.zeros((B, C, H, W), dtype=torch.bool).to(po_mask.device)
        for b in range(inst_edge_map.shape[0]):
            inst_edge_map[b, :, :, :] = grad_mag[b, :, :, :] > 0.5  # This implies change of label.

            # Zero out the edges of the image to prevent computation wastage
            inst_edge_map[b, :, :11, :] = 0
            inst_edge_map[b, :, H - 10:, :] = 0
            inst_edge_map[b, :, :, :11] = 0
            inst_edge_map[b, :, :, W - 10:] = 0

        return inst_edge_map, grad_x, grad_y, grad_mag

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

        edge_x_samp_a = torch.zeros((4 * points_count), dtype=torch.long).to(edge_map.device)
        edge_y_samp_a = torch.zeros((4 * points_count), dtype=torch.long).to(edge_map.device)
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

        return (batch_samp_a, ch_samp_a, edge_x_samp_a, edge_y_samp_a), (batch_samp_b, ch_samp_b, edge_x_samp_b, edge_y_samp_b)

    def computePairwiseLoss(self, flow_pred_iter, flow_gt, points_a_sampled, points_b_sampled):
        edge_loss = 0.0
        num_preds = len(flow_pred_iter)

        # Define the common tensors outside
        plusone_tensor = torch.ones_like(flow_gt)
        minusone_tensor = torch.ones_like(flow_gt) * -1
        zero_tensor = torch.zeros_like(flow_gt)

        for i in range(num_preds):
            i_weight = self.gamma ** (num_preds - i - 1)

            flow_pred_a = flow_pred_iter[i][points_a_sampled]
            flow_pred_b = flow_pred_iter[i][points_b_sampled]
            flow_gt_a = flow_gt[points_a_sampled]
            flow_gt_b = flow_gt[points_b_sampled]

            plusone_condition = flow_gt_a / flow_gt_b >= 1 + self.pointwise_thresh
            minusone_condition = flow_gt_a / flow_gt_b <= 1 / (1 + self.pointwise_thresh)
            l = torch.where(plusone_condition, plusone_tensor, torch.where(minusone_condition, minusone_tensor, zero_tensor))

            pairwise_loss = torch.where(l == zero_tensor,
                                        (flow_pred_a - flow_pred_b).pow(2),  # True
                                        torch.log(1 + torch.exp(-l * (flow_pred_a - flow_pred_b)))).mean()

            edge_loss += i_weight * pairwise_loss

        return edge_loss