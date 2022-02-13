from collections import OrderedDict

import torch
import torch.nn as nn
from efficientPS.utils.visualisation import visualiseFlow, visualiseFlowMFN
from efficientPS.algos.optical_flow import EndPointError, EPELossWithMask, EPELoss

from efficientPS.utils.optical_flow_ops import *

from efficientPS.utils.sequence import pad_packed_images

NETWORK_INPUTS = ["img_pair", "flow"]


class OpticalFlowNet(nn.Module):
    def __init__(self, body, flow_head, flow_algo, ds=6):
        super(OpticalFlowNet, self).__init__()

        # Modules
        self.body = body
        self.flow_head = flow_head

        # Algorithms
        self.flow_algo = flow_algo
        self.computeEPE = EndPointError

        # Params
        self.ds = ds

    def forward(self, img_pair, flow_gt, do_loss=False, do_prediction=True, get_epe=False, get_vis=False):
        # Extract the multi-scale features for both the images
        prev_img = img_pair[0]
        curr_img = img_pair[1]
        prev_ms_feat = self.body(prev_img)
        curr_ms_feat = self.body(curr_img)

        result = OrderedDict()
        loss = OrderedDict()

        # Optical Flow part
        if do_loss:
            flow_prob, flow_pred, of_loss = self.flow_algo.training(self.flow_head, prev_ms_feat, curr_ms_feat, flow_gt)
        elif do_prediction:
            flow_prob, flow_pred = self.flow_algo.inference(self.flow_head, prev_ms_feat, curr_ms_feat)
        else:
            flow_prob, flow_pred, of_loss = None, None, None

        # Prepare outputs
        loss['loss'] = of_loss['total']

        # The predictions
        result["flow_prob"] = flow_prob
        result["flow_pred"] = flow_pred

        # Compute EPE
        if get_epe and flow_pred is not None:
            # Compute the EPE number
            scale_factor = 1 / 2**(self.ds - len(flow_pred) + 1)
            loss['epe'] = self.computeEPE(flow_pred[-1] * scale_factor, flow_gt)

        # Get the visualisation
        if get_vis:
            result['vis'] = visualiseFlow(img_pair, flow_gt, flow_pred, flow_prob, self.ds)
        return loss, result


class OpticalFlowNetMFN(nn.Module):
    def __init__(self, body, flow_head, flow_algo, dataset):
        super(OpticalFlowNetMFN, self).__init__()

        # Modules
        self.body = body
        self.flow_head = flow_head

        # Algorithms
        self.flow_algo = flow_algo
        self.epe_with_mask = EPELossWithMask()

        self.dataset = dataset

    def forward(self, img_pair, flow_gt, mask, do_loss=False, do_prediction=True, get_epe=False, get_vis=False):
        # Extract the multi-scale features for both the images
        prev_img = img_pair[0]
        curr_img = img_pair[1]
        prev_ms_feat = self.body(prev_img)
        curr_ms_feat = self.body(curr_img)

        result = OrderedDict()
        loss = OrderedDict()

        # Optical Flow part
        if do_loss:
            flow_pred, flow_vis, of_loss = self.flow_algo.training(self.flow_head, prev_ms_feat, curr_ms_feat, prev_img, curr_img, flow_gt, mask)
        elif do_prediction:
            flow_pred, flow_vis = self.flow_algo.inference(self.flow_head, prev_ms_feat, curr_ms_feat, prev_img, curr_img)
        else:
            flow_pred, flow_vis, of_loss = None, None, None

        # Prepare outputs
        loss['loss'] = of_loss['total']

        # The predictions
        result["flow_pred"] = flow_pred
        # result['vis'] = flow_vis

        # Compute EPE
        if get_epe and flow_pred is not None:
            # Compute the EPE number
            loss['epe'] = self.epe_with_mask(Upsample(flow_pred[0], 4), flow_gt, mask)

        # # Get the visualisation
        if get_vis:
            result['vis'] = visualiseFlowMFN(img_pair, flow_gt, flow_vis, mask, self.dataset)
        return loss, result


class OpticalFlowNetMFNInstance(nn.Module):
    def __init__(self, body, flow_head, flow_algo, dataset):
        super(OpticalFlowNetMFNInstance, self).__init__()

        # Modules
        self.body = body
        self.flow_head = flow_head

        # Algorithms
        self.flow_algo = flow_algo
        self.epe_with_mask = EPELossWithMask()

        self.dataset = dataset

    def forward(self, img_pair, flow_gt, mask, points_2d=None, valid_2d=None, points_3d=None, valid_3d=None, points_flag=None, k=None, do_loss=False, do_prediction=True, get_epe=False, get_vis=False, get_test_metrics=False):
        # Extract the multi-scale features for both the images
        prev_img = img_pair[0]
        curr_img = img_pair[1]
        prev_ms_feat = self.body(prev_img)
        curr_ms_feat = self.body(curr_img)

        result = OrderedDict()
        loss = OrderedDict()

        # Optical Flow part
        if do_loss:
            flow_pred, flow_vis, of_loss = self.flow_algo.training(self.flow_head, prev_ms_feat, curr_ms_feat, prev_img, curr_img, flow_gt, mask, points_2d, valid_2d, points_3d, valid_3d, points_flag, k)
        elif do_prediction:
            flow_pred, flow_vis = self.flow_algo.inference(self.flow_head, prev_ms_feat, curr_ms_feat, prev_img, curr_img)
        else:
            flow_pred, flow_vis, of_loss = None, None, None

        # Prepare outputs
        loss['loss'] = of_loss['total']

        if "reproj_pnp_gt" in of_loss.keys():
            loss['reproj_pnp_gt'] = of_loss['reproj_pnp_gt']

        if "reproj_pnp_pred" in of_loss.keys():
            loss['reproj_pnp_pred'] = of_loss['reproj_pnp_pred']

        if "pose_error" in of_loss.keys():
            loss['pose_error'] = of_loss['pose_error']

        # The predictions
        result["flow_pred"] = flow_pred
        # result['vis'] = flow_vis

        # Compute EPE
        if get_epe and flow_pred is not None:
            # Compute the EPE number
            loss['epe'] = self.epe_with_mask(Upsample(flow_pred[0], 4), flow_gt, mask)

        # # Get the visualisation
        if get_vis:
            result['vis'] = visualiseFlowMFN(img_pair, flow_gt, flow_vis, mask, self.dataset)

        # Get the test metrics
        if get_test_metrics:
            pred_flow = Upsample(flow_pred[0], 4)
            gt_flow = flow_gt

            # result['fl_bg'], result['abs_rel'], result['rms'], result['sq_rel'], result['log_rms'], result['d1'], result['d2'], result['d3'], result['si_log'] = computeTestMetrics(gt_flow, pred_flow, mask, fg_bg_mask)

        return loss, result


class OpticalFlowNetRAFT(nn.Module):
    def __init__(self, body, flow_head, flow_algo, dataset):
        super(OpticalFlowNetRAFT, self).__init__()

        # Modules
        self.body = body
        self.flow_head = flow_head

        # Algorithms
        self.flow_algo = flow_algo

        self.dataset = dataset

    def forward(self, img_pair, flow_gt, valid_mask, flow_shape=None, do_loss=False, do_prediction=True, get_vis=False, get_test_metrics=False):
        # Extract the multi-scale features for both the images
        prev_img = img_pair[0]
        curr_img = img_pair[1]
        prev_ms_feat = self.body(prev_img)
        curr_ms_feat = self.body(curr_img)

        result = OrderedDict()
        loss = OrderedDict()

        # Optical Flow part
        if do_loss:
            flow_pred_iter, flow_pred, flow_pred_up, of_loss = self.flow_algo.training(self.flow_head, prev_ms_feat, curr_ms_feat, flow_gt, valid_mask, flow_shape)
        elif do_prediction:
            flow_pred_iter, flow_pred, flow_pred_up, of_loss = self.flow_algo.inference(self.flow_head, prev_ms_feat, curr_ms_feat, flow_gt, valid_mask, flow_shape)
        else:
            flow_pred, flow_vis, of_loss = None, None, None

        # Prepare outputs
        loss['loss'] = of_loss['total']

        # The predictions
        result["flow_pred"] = flow_pred_up

        # Get the visualisation
        if get_vis:
            result['vis'] = visualiseFlowMFN(img_pair, flow_gt, flow_pred_up, valid_mask, self.dataset)

        # Get the test metrics
        if get_test_metrics:
            pred_flow = flow_pred_up
            gt_flow = flow_gt

            result['epe'] = computeTestMetrics(gt_flow, pred_flow, valid_mask, None)

        return loss, result


def computeErrorOutlier(gt, pred, valid_mask, fg_bg):
    ABS_THRESH = 3.0
    REL_THRESH = 0.05

    flow_diff_u = (gt[0, :, :] - pred[0, :, :]).unsqueeze(0)
    flow_diff_v = (gt[1, :, :] - pred[1, :, :]).unsqueeze(0)
    flow_diff_dist = (flow_diff_u.pow(2) + flow_diff_v.pow(2)).sqrt()
    flow_mag = (gt[0, :, :].pow(2) + gt[1, :, :].pow(2)).sqrt().unsqueeze(0)
    flow_err = torch.logical_and(flow_diff_dist > ABS_THRESH, (flow_diff_dist / flow_mag) > REL_THRESH)

    # Compute background outliers
    bg_mask = (fg_bg == 0)
    num_pixels_bg = torch.sum(torch.logical_and(bg_mask, valid_mask))  # Count the number of BG pixels
    num_errors_bg = torch.sum(torch.logical_and(torch.logical_and(bg_mask, valid_mask), flow_err))  # Count the number of erroneous BG pixels

    num_errors_bg_result = torch.sum(torch.logical_and(torch.logical_and(torch.logical_and(bg_mask, valid_mask), valid_mask), flow_err))  # Count the number of erroneous BG pixels
    num_pixels_bg_result = torch.sum(torch.logical_and(torch.logical_and(bg_mask, valid_mask), valid_mask))  # Count the number of erroneous BG pixels

    # Compute the foreground outliers
    fg_mask = torch.logical_not(bg_mask)
    num_pixels_fg = torch.sum(torch.logical_and(fg_mask, valid_mask))  # Count the number of FG pixels
    num_errors_fg = torch.sum(torch.logical_and(torch.logical_and(fg_mask, valid_mask), flow_err))  # Count the number of erroneous FG pixels

    num_errors_fg_result = torch.sum(torch.logical_and(torch.logical_and(torch.logical_and(fg_mask, valid_mask), valid_mask), flow_err))  # Count the number of erroneous FG pixels
    num_pixels_fg_result = torch.sum(torch.logical_and(torch.logical_and(fg_mask, valid_mask), valid_mask))  # Count the number of erroneous FG pixels

    # Compute all outliers
    num_pixels_all = torch.sum(valid_mask)
    num_errors_all = torch.sum(torch.logical_and(valid_mask, flow_err))
    num_pixels_all_result = num_pixels_all
    num_errors_all_result = num_errors_all

    if num_pixels_all > 0:
        density = num_pixels_all_result / num_pixels_all
    else:
        density = num_pixels_all_result

    return num_errors_bg, num_pixels_bg, num_errors_bg_result, num_pixels_bg_result, num_errors_fg, num_pixels_fg, num_errors_fg_result, num_pixels_fg_result, num_errors_all, num_pixels_all, num_errors_all_result, num_pixels_all_result, density


def computeTestMetrics(gt_batch, pred_batch, valid_mask_batch, fg_bg_batch):
    epe = torch.sum((pred_batch - gt_batch) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid_mask_batch.view(-1)].mean()

    return epe

    # for b in range(gt_batch.size(0)):
    #     valid_mask = valid_mask_batch[b, :, :, :]
    #
    #     gt = gt_batch[b, :, :, :]
    #     gt[0, valid_mask[0, :, :]] = 0
    #     gt[1, valid_mask[0, :, :]] = 0
    #
    #     pred = pred_batch[b, :, :, :]
    #     pred[0, valid_mask[0, :, :]] = 0
    #     pred[1, valid_mask[0, :, :]] = 0
    #
    #     fg_bg = fg_bg_batch[b, :, :, :]
    #
    #     num_errors_bg, num_pixels_bg, num_errors_bg_result, num_pixels_bg_result, num_errors_fg, num_pixels_fg, num_errors_fg_result, num_pixels_fg_result, num_errors_all, num_pixels_all, num_errors_all_result, num_pixels_all_result, density = computeErrorOutlier(gt, pred, valid_mask, fg_bg)


