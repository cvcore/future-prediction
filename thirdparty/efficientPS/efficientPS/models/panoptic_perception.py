from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from efficientPS.utils.sequence import pad_packed_images
from efficientPS.utils.visualisation import visualisePanopticDepth, visualiseSemanticSegmentation, visualiseFlowMFN

NETWORK_INPUTS = ["img", "msk", "cat", "iscrowd", "bbx", "depth"]

class PanopticPerceptionNet(nn.Module):
    def __init__(self,
                 body,
                 rpn_head,
                 roi_head,
                 sem_head,
                 sem_depth_head,
                 flow_head,
                 multi_loss_head,
                 rpn_algo,
                 inst_algo,
                 sem_algo,
                 sem_depth_algo,
                 flow_algo,
                 po_algo,
                 multi_loss_algo,
                 dataset,
                 msk_predict,
                 min_depth=0.1,
                 max_depth=100,
                 sem_class_count=8,
                 classes=None,
                 combined_classes=None):
        super(PanopticPerceptionNet, self).__init__()

        # Backbone
        self.body = body

        # Modules
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.sem_head = sem_head
        self.sem_depth_head = sem_depth_head
        self.flow_head = flow_head
        self.multi_loss_head = multi_loss_head

        # Algorithms
        self.rpn_algo = rpn_algo
        self.inst_algo = inst_algo
        self.sem_algo = sem_algo
        self.sem_depth_algo = sem_depth_algo
        self.flow_algo = flow_algo
        self.po_algo = po_algo
        self.msk_predict = msk_predict
        self.multi_loss_algo = multi_loss_algo

        # Params
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.dataset = dataset
        self.sem_class_count = sem_class_count
        self.num_stuff = classes["stuff"]
        self.combined_classes = combined_classes

    def _makeSemanticClassMask(self, sem, sem_class_count):
        B = len(sem)
        H, W = sem[0].shape
        sem_class_mask = torch.zeros((B, sem_class_count, H, W), dtype=torch.bool).to(sem[0].device)

        for b_idx in range(B):
            sem_seg_combined_class = torch.zeros((sem_class_count, H, W), dtype=torch.bool)
            sem_seg_remaining_class = torch.ones((H, W), dtype=torch.bool)
            for ch_idx, class_list in enumerate(self.combined_classes):
                for cls in class_list:
                    sem_seg_combined_class[ch_idx, sem[b_idx] == cls] = True
                    sem_seg_remaining_class[sem[b_idx] == cls] = False
            sem_seg_combined_class[-1, sem_seg_remaining_class] = True

            sem_class_mask[b_idx, :, :, :] = sem_seg_combined_class

        return sem_class_mask

    def _prepare_inputs(self, msk, cat, iscrowd, bbx):
        cat_out, iscrowd_out, bbx_out, ids_out, sem_out = [], [], [], [], []

        for msk_i, cat_i, iscrowd_i, bbx_i in zip(msk, cat, iscrowd, bbx):
            msk_i = msk_i.squeeze(0)
            thing = (cat_i >= self.num_stuff) & (cat_i != 255)
            valid = thing & ~(iscrowd_i > 0)

            if valid.any().item():
                cat_out.append(cat_i[valid])
                bbx_out.append(bbx_i[valid])
                ids_out.append(torch.nonzero(valid))
            else:
                cat_out.append(None)
                bbx_out.append(None)
                ids_out.append(None)

            if iscrowd_i.any().item():
                iscrowd_i = (iscrowd_i > 0) & thing
                iscrowd_out.append(iscrowd_i[msk_i].type(torch.uint8))
            else:
                iscrowd_out.append(None)

            sem_out.append(cat_i[msk_i])

        return cat_out, iscrowd_out, bbx_out, ids_out, sem_out

    def forward(self, img_pair, msk=None, cat=None, iscrowd=None, bbx=None, depth_gt=None, flow_gt=None, flow_mask=None, do_loss=False, do_prediction=False, get_test_metrics=False, get_depth_vis=False, get_sem_vis=False, get_flow_vis=False):
        result = OrderedDict()
        loss = OrderedDict()
        stats = OrderedDict()

        # Get some parameters
        img_1, valid_size_1 = pad_packed_images(img_pair[0])
        img_2, valid_size_2 = pad_packed_images(img_pair[1])
        img_size = img_1.shape[-2:]

        if do_loss:
            cat, iscrowd, bbx, ids, sem_gt = self._prepare_inputs(msk, cat, iscrowd, bbx)
            sem_class_mask = self._makeSemanticClassMask(sem_gt, self.sem_class_count)

        # print(img_1.shape, img_2.shape)
        # Get the image features
        ms_img_feat_1 = self.body(img_1)
        ms_img_feat_2 = self.body(img_2)

        # RPN Part
        if do_loss:
            # print("MS_IMG_FEAT", ms_img_feat_1[0.shape)
            # print("BBX", bbx.shape)
            obj_loss, bbx_loss, proposals = self.rpn_algo.training(self.rpn_head, ms_img_feat_1, bbx, iscrowd, valid_size_1, training=self.training, do_inference=True)
        elif do_prediction:
            proposals = self.rpn_algo.inference(self.rpn_head, ms_img_feat_1, valid_size_1, self.training)
            obj_loss, bbx_loss = None, None
        else:
            obj_loss, bbx_loss, proposals = None, None, None

        # ROI Part
        if do_loss:
            roi_cls_loss, roi_bbx_loss, roi_msk_loss = self.inst_algo.training(self.roi_head, ms_img_feat_1, proposals, bbx, cat, iscrowd, ids, msk, img_size)
        else:
            roi_cls_loss, roi_bbx_loss, roi_msk_loss = None, None, None
        if do_prediction:
            bbx_pred, cls_pred, obj_pred, msk_pred = self.inst_algo.inference(self.roi_head, ms_img_feat_1, proposals, valid_size_1, img_size)
        else:
            bbx_pred, cls_pred, obj_pred, msk_pred = None, None, None, None

        # Segmentation Part
        if do_loss:
            sem_loss, sem_conf_mat, sem_pred, sem_logits, sem_feat = self.sem_algo.training(self.sem_head, ms_img_feat_1, sem_gt, valid_size_1, img_size)
        elif do_prediction:
            sem_pred, sem_logits, sem_feat = self.sem_algo.inference(self.sem_head, ms_img_feat_1, valid_size_1, img_size)
            sem_loss, sem_conf_mat = None, None
        else:
            sem_loss, sem_conf_mat, sem_pred, sem_logits, sem_feat = None, None, None, None, None

        # Depth Part
        # if do_loss:
        #     sem_class_depth_pred, sem_depth_pred, depth_bts_loss, depth_class_loss, depth_panoptic_edge_loss, sem_depth_stats = self.sem_depth_algo.training(self.sem_depth_head, ms_img_feat_1, sem_feat, depth_gt, po_mask=msk, sem_class_mask=sem_class_mask)
        # elif do_prediction:
        #     sem_class_depth_pred, sem_depth_pred = self.sem_depth_algo.inference(self.sem_depth_head, ms_img_feat_1, sem_feat)
        #     depth_bts_loss, depth_class_loss, depth_panoptic_edge_loss, sem_depth_stats = None, None, None, None
        # else:
        #     sem_class_depth_pred, sem_depth_pred, depth_bts_loss, depth_class_loss, depth_panoptic_edge_loss, sem_depth_stats = None, None, None, None, None, None

        # Flow Part
        # if do_loss:
        #     flow_pred_iter, flow_pred, flow_pred_up, flow_loss = self.flow_algo.training(self.flow_head, ms_img_feat_1, ms_img_feat_2, flow_gt, flow_mask, img_1.shape)
        #     flow_loss = flow_loss['total']
        # elif do_prediction:
        #     flow_pred_iter, flow_pred, flow_pred_up, flow_loss = self.flow_algo.training(self.flow_head, ms_img_feat_1, ms_img_feat_2, flow_gt, flow_mask, img_1.shape)
        #     flow_loss = flow_loss['total']
        # else:
        #     flow_pred_iter, flow_pred, flow_pred_up, flow_loss = None, None, None, None

        # Multi-loss head
        if do_loss:
            depth_bts_loss = depth_class_loss = depth_panoptic_edge_loss = flow_loss = 0
            panoptic_loss = obj_loss + bbx_loss + roi_cls_loss + roi_bbx_loss + roi_msk_loss + sem_loss
            total_loss, loss_weights = self.multi_loss_algo.computeMultiLoss(self.multi_loss_head,
                                                                            [panoptic_loss, depth_bts_loss, depth_class_loss, depth_panoptic_edge_loss, flow_loss],
                                                                            [True, False, False, False, False])

            # Prepare outputs
            # LOSSES
            loss['obj_loss'] = loss_weights[0] * obj_loss
            loss['bbx_loss'] = loss_weights[0] * bbx_loss
            loss['roi_cls_loss'] = loss_weights[0] * roi_cls_loss
            loss['roi_bbx_loss'] = loss_weights[0] * roi_bbx_loss
            loss['roi_msk_loss'] = loss_weights[0] * roi_msk_loss
            loss["sem_loss"] = loss_weights[0] * sem_loss
            # loss['depth_bts_loss'] = loss_weights[1] * depth_bts_loss
            # loss['depth_class_loss'] = loss_weights[2] * depth_class_loss
            # loss['depth_po_edge_loss'] = loss_weights[3] * depth_panoptic_edge_loss
            # loss['flow_loss'] = loss_weights[4] * flow_loss

            # OTHER STATISTICS
            # if sem_depth_stats is not None:
            #     for key in sem_depth_stats.keys():
            #         stats[key] = sem_depth_stats[key]
            stats['sem_conf'] = sem_conf_mat
        else:
            loss = None
            total_loss = None

        # PREDICTIONS
        result['bbx_pred'] = bbx_pred
        result['cls_pred'] = cls_pred
        result['obj_pred'] = obj_pred
        result['msk_pred'] = msk_pred
        result["sem_pred"] = sem_pred
        result['sem_logits'] = sem_logits
        # result["sem_depth_pred"] = sem_depth_pred
        # result['flow_pred'] = flow_pred_up
        # result["wt_panoptic"] = loss_weights[0]
        # result['wt_depth_bts'] = loss_weights[1]
        # result['wt_depth_class'] = loss_weights[2]
        # result['wt_depth_po_edge'] = loss_weights[3]
        # result['wt_flow'] = loss_weights[4]

        # Get the visualisation
        # if get_depth_vis:
        #     result['depth_vis'] = visualisePanopticDepth(img_1, depth_gt, sem_depth_pred, sem_class_depth_pred, sem_class_mask, self.dataset)

        if get_sem_vis and do_loss:
            result['sem_vis'] = visualiseSemanticSegmentation(img_1, sem_gt, sem_pred, self.dataset)

        # if get_flow_vis:
        #     result['flow_vis'] = visualiseFlowMFN([img_1, img_2], flow_gt, flow_pred_up, flow_mask, self.dataset)

        # Get the test metrics
        # if get_test_metrics:
        #     pred_depth = sem_depth_pred.detach().clone()  #.cpu().numpy()
        #     depth_gt, _ = pad_packed_images(depth_gt)
        #     gt_depth = depth_gt.detach().clone()  #.cpu().numpy()
        #     pred_depth[pred_depth < self.min_depth] = self.min_depth
        #     pred_depth[pred_depth > self.max_depth] = self.max_depth
        #     pred_depth[torch.isinf(pred_depth)] = self.max_depth
        #     pred_depth[torch.isnan(pred_depth)] = self.min_depth

            # if self.dataset == "KittiPanoptic":
            #     pred_disp = convertDepthToDisp(pred_depth)
            #     gt_disp = convertDepthToDisp(gt_depth)

            # valid_mask = (gt_depth > self.min_depth) & (gt_depth < self.max_depth)

            # if self.dataset == "KittiPanoptic":
            #     result['depth_log10'], result['depth_abs_rel'], result['depth_rms'], result['depth_sq_rel'], result['depth_log_rms'], result['depth_d1'], result['depth_d2'], result['depth_d3'], result['depth_si_log'] = computeDepthTestMetrics(gt_disp, pred_disp, valid_mask)
            # else:
            #     result['depth_log10'], result['depth_abs_rel'], result['depth_rms'], result['depth_sq_rel'], result[
            #         'depth_log_rms'], result['depth_d1'], result['depth_d2'], result['depth_d3'], result[
            #         'depth_si_log'] = computeDepthTestMetrics(gt_depth, pred_depth, valid_mask)

            # flow_gt, _ = pad_packed_images(flow_gt)
            # flow_mask, _ = pad_packed_images(flow_mask)
            # result['flow_epe'] = computeFlowTestMetrics(flow_gt, flow_pred_up, flow_mask, None)

        return total_loss, loss, result, stats, ms_img_feat_1


def convertDepthToDisp(depth):
    baseline = 0.54
    focal_length = 721
    disp = torch.reciprocal(depth) / float(baseline * focal_length)
    return disp


def computeDepthTestMetrics(gt_batch, pred_batch, mask):
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

            d1 = torch.mean(torch.lt(thresh, 1.25).type(torch.float32))
            d2 = torch.mean(torch.lt(thresh, 1.25 ** 2).type(torch.float32))
            d3 = torch.mean(torch.lt(thresh, 1.25 ** 3).type(torch.float32))

            rmse = (gt - pred) ** 2
            rmse = torch.sqrt(torch.mean(rmse))

            rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
            rmse_log = torch.sqrt(torch.mean(rmse_log))

            abs_rel = torch.mean(torch.div(torch.abs(gt - pred), gt))
            sq_rel = torch.mean(torch.div((gt - pred) ** 2, gt))

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


def computeFlowTestMetrics(gt_batch, pred_batch, valid_mask_batch, fg_bg_batch):
    epe = torch.sum((pred_batch - gt_batch) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid_mask_batch.view(-1)].mean()

    return [epe]
