import numpy as np
import torch
import warnings

from futurepred.criterions.scale_invariant_depth import ScaleInvariantDepthLoss
from futurepred.datasets.cityscapes import Cityscapes

from .meters import Meter


def confusion_matrix(pred, gt, MAX_CLS=256):
    """ produce confusion matrix of size MAX_CLS x MAX_CLS for mIoU calculation
        each element i, j denotes the number of pixels of groundtruth class i that have been predicted into class j.
        :param pred: tensor of shape (b) x H x W
        :param gt: tensor of shape (b) x H x W
    """
    conf_mat = gt.new_zeros(MAX_CLS * MAX_CLS, dtype=torch.float)
    conf_mat.index_add_(0, gt.reshape(-1) * MAX_CLS + pred.reshape(-1), conf_mat.new_ones(gt.numel()))
    return conf_mat.reshape(MAX_CLS, MAX_CLS)


def mean_iou(conf_mat, n_classes, select_classes=None, avg_only=False):
    """ calculate mIoU for each cityscapes class and return the results as dictionary
        :param conf_mat: confusion matrix
        :param n_classes: number of classes to consider with range 0 .. n_classes-1
        :param select_classes: (optional) list. If given, only calculate IoU from these classes
        :param avg_only: if True, return only average mIoU
        return:
        dictionary of CLS_miou for wandb logging
    """
    conf_mat = conf_mat[:n_classes, :]
    miou = conf_mat.diag() / (conf_mat.sum(dim=1) + conf_mat.sum(dim=0)[:n_classes] - conf_mat.diag()) # TP / (TP+FP+FN)

    results = {}
    for id, miou_cls in enumerate(miou):
        if select_classes is not None and id not in select_classes:
            continue
        results['{}_miou'.format(Cityscapes.train_id_to_name(id))] = miou_cls.item()
    if len(results.items()) == 0:
        avg_miou = 0
    else:
        avg_miou = np.mean(list(results.values()))

    if avg_only:
        results = {'average_miou': avg_miou}
    else:
        results['average_miou'] = avg_miou

    return results


class MIoUMeter(Meter):

    def __init__(self, n_classes, select_class=None, avg_only=False):
        """ args:
            n_classes: classes to use for final mIoU calculation. We assume here the valid classes are continuous from 0 .. n_classes-1
            select_class (list): if given, only report the classes in this list. Otherwise report all classes
            avg_only: return only average mIoU
        """
        super().__init__()
        MAX_CLS = 256
        self.conf_mat = torch.zeros(MAX_CLS, MAX_CLS, dtype=torch.double)
        self.n_classes = n_classes
        self.select_class = select_class
        self.avg_only = avg_only

    def add(self, pred_logit, gt):
        """ calculate data batch and return mIoU dictionary for wandb logging """
        pred = pred_logit.argmax(dim=1)
        gt = gt.squeeze()

        curr_cls = gt.unique() if self.select_class is None else self.select_class
        conf_mat = confusion_matrix(pred, gt).to(self.conf_mat)
        self.conf_mat += conf_mat
        return mean_iou(conf_mat, self.n_classes, curr_cls, self.avg_only) # return batch miou for plotting

    def value(self):
        return mean_iou(self.conf_mat, self.n_classes, self.select_class, self.avg_only)


class EPEMeter(Meter):
    """ average end-point error for measuring flow prediction """
    def __init__(self):
        self.reset()

    def reset(self):
        self.epe_history = []

    def add(self, pred, gt):
        epe = torch.norm((pred - gt)).item()
        self.epe_history.append(epe)

        return {"epe": epe}

    def value(self):
        return {"average_epe": np.mean(self.epe_history)}


class SILEMeter(Meter):
    """ Scale invariant logarithmic error """
    def __init__(self, use_log_depth):
        self.sid_loss = ScaleInvariantDepthLoss(input_logd=use_log_depth, target_logd=use_log_depth)
        self.reset()

    def reset(self):
        self.loss_history = []

    def add(self, pred, gt):
        sile = self.sid_loss(pred, gt).item()
        self.loss_history.append(sile)

        return {"sile": sile}

    def value(self):
        return {"average_sile": np.mean(self.loss_history)}


class PQStuffMeter(Meter):
    """ Calculate PQ (panoptic quality) for stuff classes """

    def __init__(self, n_classes, ignore_class=None, stuff_classes=None):
        """
        :param n_classes: int. Number of classes used for evaluation. Will calculate PQ stats for class 0..n_classes-1
        :param ignore_class: int. Class to ignore
        :param stuff_classes: (optional) list of ints. Only use these classes in calculation.
        """
        super().__init__()
        self.n_classes = n_classes
        self.ignore_class = ignore_class
        self.stuff_classes = stuff_classes
        self.reset()

    def reset(self):
        self.iou = torch.zeros(self.n_classes)
        self.tp = torch.zeros(self.n_classes)
        self.fp = torch.zeros(self.n_classes)
        self.fn = torch.zeros(self.n_classes)

    def add(self, pred_logit, gt):
        """ add new sample
            arguments:
            :param pred_logit: b x C x H x W
            :param gt: b x 1 x H x W
        """
        assert pred_logit.shape[0] == gt.shape[0] and pred_logit.shape[-2:] == gt.shape[-2:], 'Invalid dimensions found!'
        pred = pred_logit.argmax(dim=1)
        gt = gt.squeeze(dim=1)
        if self.ignore_class: # remove pixels in prediction that are labelled as void in groundtruth
            void_mask = (gt==self.ignore_class)
            pred[void_mask] = self.ignore_class

        for b in range(pred_logit.shape[0]):
            conf_mat = confusion_matrix(pred[b], gt[b])[:self.n_classes, :self.n_classes]
            tp_pixels = conf_mat.diag()
            fp_pixels = conf_mat.sum(dim=1) - tp_pixels
            fn_pixels = conf_mat.sum(dim=0) - tp_pixels
            denomiator = tp_pixels + fp_pixels + fn_pixels
            iou = tp_pixels / denomiator
            iou[denomiator==0] = 0

            # count statics for each class
            tp_mask = (iou >= 0.5)
            self.iou += (iou*tp_mask).cpu()
            self.tp += (torch.ones_like(tp_pixels)*tp_mask).cpu()
            fp_mask = (iou < 0.5) * (fp_pixels > 0)
            self.fp += (torch.ones_like(fp_pixels)*fp_mask).cpu()
            fn_mask = (iou < 0.5) * (fn_pixels > 0)
            self.fn += (torch.ones_like(fn_pixels)*fn_mask).cpu()


    def value(self):
        sq = self.iou / self.tp # segmentation quality
        rq = self.tp / (self.tp + 0.5*self.fp + 0.5*self.fn) # recognition quality
        pq = self.iou / (self.tp + 0.5*self.fp + 0.5*self.fn) # panoptic quality

        results = {}
        for id, stats in enumerate(zip(sq, rq, pq)):
            if self.stuff_classes is not None and id not in self.stuff_classes:
                continue
            results['{}_sq'.format(Cityscapes.train_id_to_name(id))] = stats[0].item()
            results['{}_rq'.format(Cityscapes.train_id_to_name(id))] = stats[1].item()
            results['{}_pq'.format(Cityscapes.train_id_to_name(id))] = stats[2].item()

        if self.stuff_classes:
            results['average_sq'] = sq[self.stuff_classes].mean().item()
            results['average_rq'] = rq[self.stuff_classes].mean().item()
            results['average_pq'] = pq[self.stuff_classes].mean().item()
        else:
            results['average_sq'] = sq.mean().item()
            results['average_rq'] = rq.mean().item()
            results['average_pq'] = pq.mean().item()

        return results
