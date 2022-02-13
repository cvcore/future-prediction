from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from efficientPS.utils.visualisation import visualiseDepth

NETWORK_INPUTS = ["img", "depth"]


class DepthNet(nn.Module):
    def __init__(self, body, depth_head, depth_algo, dataset, min_depth=1e-3, max_depth=10):
        super(DepthNet, self).__init__()

        # Modules
        self.body = body
        self.depth_head = depth_head

        # Algorithms
        self.depth_algo = depth_algo

        # Params
        self.max_depth = max_depth
        self.min_depth = min_depth

        self.dataset = dataset

    def forward(self, img, depth_gt, iseg_object=None, iseg_boundary=None, sem_class_mask=None, do_loss=False, do_prediction=True, get_test_metrics=False, get_vis=False):
        ms_feat = self.body(img)

        result = OrderedDict()
        loss = OrderedDict()

        # Depth part
        if do_loss:
            ms_depth, depth_pred, depth_loss = self.depth_algo.training(self.depth_head, ms_feat, depth_gt.clone(), iseg_object, iseg_boundary, sem_class_mask=sem_class_mask)
        elif do_prediction:
            ms_depth, depth_pred = self.depth_algo.inference(self.depth_head, ms_feat)
        else:
            ms_depth, depth_pred, depth_loss = None, None, None

        # Prepare outputs
        if do_loss:
            loss['loss'] = depth_loss['total']
            for key in depth_loss.keys():
                if key != "total":
                    loss[key] = depth_loss[key]
            # if (iseg_boundary is not None) and (iseg_object is not None):
            #     loss['si'] = depth_loss['si']
            #     loss['boundary'] = depth_loss['boundary']
            #     loss['object'] = depth_loss['object']

        # The predictions
        result["depth_pred"] = depth_pred

        # Compute EPE
        # if get_epe and flow_pred is not None:
        #     # Compute the EPE number
        #     scale_factor = 1 / 2**(self.ds - len(flow_pred) + 1)
        #     loss['epe'] = self.computeEPE(flow_pred[-1] * scale_factor, flow_gt)

        # Get the visualisation
        if get_vis:
            result['vis'] = visualiseDepth(img, depth_gt, depth_pred, ms_depth, self.min_depth, self.max_depth, self.dataset, iseg_object=iseg_object, iseg_boundary=iseg_boundary)

        # Get the test metrics
        if get_test_metrics:
            pred_depth = depth_pred#.cpu().numpy()
            gt_depth = depth_gt#.cpu().numpy()
            pred_depth[pred_depth < self.min_depth] = self.min_depth
            pred_depth[pred_depth > self.max_depth] = self.max_depth
            pred_depth[torch.isinf(pred_depth)] = self.max_depth
            pred_depth[torch.isnan(pred_depth)] = self.min_depth

            # print(self.min_depth, self.max_depth)

            valid_mask = torch.logical_and(gt_depth > self.min_depth, gt_depth < self.max_depth)
            # valid_mask = gt_depth > self.min_depth

            result['log10'], result['abs_rel'], result['rms'], result['sq_rel'], result['log_rms'], result['d1'], result['d2'], result['d3'], result['si_log'] = computeTestMetrics(gt_depth, pred_depth, valid_mask)

        return loss, result


def computeTestMetrics(gt_batch, pred_batch, mask):
    log10_batch = []
    abs_rel_batch = []
    rmse_batch = []
    sq_rel_batch = []
    rmse_log_batch = []
    d1_batch = []
    d2_batch = []
    d3_batch = []
    si_log_batch = []

    for b in range(gt_batch.size(0)):
        gt = gt_batch[b, :, :, :]
        gt = gt[mask[b, :, :, :]]
        pred = pred_batch[b, :, :, :]
        pred = pred[mask[b, :, :, :]]

        # mask = torch.astymask[b, :, :, :]
        if torch.sum(mask[b, :, :, :] > 0):
            thresh = torch.max((gt / pred), (pred / gt))
            # d1 = (thresh < 1.25).type(torch.float32).mean()
            # d2 = (thresh < (1.25 ** 2)).type(torch.float32).mean()
            # d3 = (thresh < (1.25 ** 3)).type(torch.float32).mean()

            d1 = torch.mean(torch.lt(thresh, 1.25).type(torch.float32))
            d2 = torch.mean(torch.lt(thresh, 1.25 ** 2).type(torch.float32))
            d3 = torch.mean(torch.lt(thresh, 1.25 ** 3).type(torch.float32))

            rmse = (gt - pred) ** 2
            rmse = torch.sqrt(torch.mean(rmse))

            rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
            rmse_log = torch.sqrt(torch.mean(rmse_log))

            abs_rel = torch.mean(torch.div(torch.abs(gt - pred), gt))
            sq_rel = torch.mean(torch.div((gt - pred) ** 2, gt))
            # abs_rel = torch.mean(torch.abs(gt - pred) / gt)
            # sq_rel = torch.mean(((gt - pred) ** 2) / gt)

            err = torch.abs(torch.log10(pred) - torch.log10(gt))
            log10 = torch.mean(err)

            log_error = torch.log(pred) - torch.log(gt)
            si_log = (log_error ** 2).mean() - (log_error.mean()) ** 2

            log10_batch.append(log10)
            abs_rel_batch.append(abs_rel)
            rmse_batch.append(rmse)
            sq_rel_batch.append(sq_rel)
            rmse_log_batch.append(rmse_log)
            d1_batch.append(d1)
            d2_batch.append(d2)
            d3_batch.append(d3)
            si_log_batch.append(si_log)
        else:
            return None, None, None, None, None, None, None, None, None

    return log10_batch, abs_rel_batch, rmse_batch, sq_rel_batch, rmse_log_batch, d1_batch, d2_batch, d3_batch, si_log_batch
