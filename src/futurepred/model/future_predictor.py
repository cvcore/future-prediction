import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.cuda.amp import autocast
import numpy as np

from . import perception
from . import prediction
from .prediction import FutureDecoderUpConv, ConditioningDistributions, Generator
from .dynamics import build_dynamics_model_from_cfg

import futurepred.utils.model
import futurepred.utils.misc as misc

import math
import copy

import warnings


class FuturePredictor(nn.Module):
    """ Module for the whole predictor architecture """

    def __init__(self, cfg, criterions):
        """
        arguments:
        cfg (Config): config containing the following attributes:
            N_HISTORY: number of history frames to process
            N_FUTURE: number of future frames to process
                the two arguments above define the shape of the input tensor (image) as
                [batch x 3 x n_history+n_future+1 x H x W]
            N_SAMPLE_EVAL: number of future samples to generate during evaluation. (For training it's always 1)
            PREDICT_CURRENT: if the model should generate prediction for the current frame.
                If turned on, the output will predict n_future+1, otherwise n_future frames
            DETERMINISTIC: whether to predict future in deterministic or probabilistic mode.
            N_SEMANTIC_CLASS: number of classes for semantic segmentation
            BACKBONE: valid choices are: 'resnet-custom' (the one used in Hu2020), 'pantopic-deeplab' (the one from panoptic-deeplab)
            USE_FLOW: whether to concatenate flow features with backbone feature.
            CRITERIONS: dictionary of criterions for each task head. Possible keys should be 'prob', 'semantic', 'flow' and 'disparity'
            WEIGHT_LOSS_PROB: weighing factor of the probabilistic loss (KL divergence loss)
        criterions (dict):
            dictionary containining different criterions for loss calculation
        """
        super().__init__()

        self.predictor_state = None
        self.pred_gt = None

        self.n_history = cfg.N_FRAMES_HISTORY
        self.n_future = cfg.N_FRAMES_FUTURE
        self.n_sample_eval = cfg.N_SAMPLE_EVAL
        self.predict_current = cfg.PREDICT_CURRENT
        self.deterministic = cfg.DETERMINISTIC
        self.criterions = criterions
        self.weight_loss_prob = cfg.WEIGHT_LOSS_PROB
        self.weight_loss_pred = cfg.WEIGHT_LOSS_PRED
        self.weight_loss_f2m_flow = cfg.WEIGHT_LOSS_F2M_FLOW
        self.weight_loss_gan_g = cfg.WEIGHT_LOSS_GAN_G
        self.out_resize = cfg.OUT_RESIZE

        self.encoder, self.encoder_key_feature, DIM_ENCODER_FEATURE = perception.build_backbone_from_cfg(cfg.BACKBONE)
        self.backbone_type = cfg.BACKBONE.TYPE
        self.freeze_backbone = cfg.BACKBONE.FREEZE

        self.flow, DIM_FLOW_FEATURE = None, 0
        self.use_flow = cfg.FLOW.ENABLE
        DIM_PERCEPTION_OUT = DIM_ENCODER_FEATURE + DIM_FLOW_FEATURE

        self.bfp_module = None

        self.dynamics = build_dynamics_model_from_cfg(cfg.DYN)

        self.blender = prediction.build_blender_from_cfg(cfg.BLENDER)

        if not self.deterministic:
            self.cond_dist = prediction.build_conditioner_from_cfg(cfg.CONDITIONER)

        self.future_generator = prediction.build_generator_from_cfg(cfg.GENERATOR)

        self.task_heads = prediction.build_future_decoder_from_cfg(cfg.DECODER)

    def condition(self, x):
        """
        args
            x: [b x 3 x T x H x W] input image, we expect T = n_history + n_future + 1 for training or T = n_history+1 for validation
               the time index 0 .. T corresponds to t - n_history .. t .. t + n_future (inclusive)

        network input:
            [batch x 3 x T x H x W]: image

        network output:
            out_final: python array of shape [n_sample_future][n_task_head][b x n_future x C x H x W]
            pred_gt: pseudo groundtruth labels [n_task_head][b x n_future x C x H x W]
            dist_h (only training): history and current distribution, for calculating the KL divergence loss
            dist_f (only training): future distribution, used together with dist_h
        """
        assert len(x.shape) == 5, "Expected 5-dim input, got {}".format(x.shape)
        n_frames = x.shape[2]
        if self.training:
            assert n_frames == self.n_history+self.n_future + \
                1, "Invalid no. of frames for training, got {}".format(
                    n_frames)
        else:
            assert n_frames == self.n_history+self.n_future + \
                1, "Invalid no. of frames for validation, got {}".format(
                    n_frames)

        # process all input frames with perception and flow module
        out_perception = []
        skip_encoder = []

        for t in range(1, n_frames):

            # use current and previous frame to keep causality
            im0 = x[:, :, t-1, :, :]
            im1 = x[:, :, t, :, :]

            if self.bfp_module:
                with torch.no_grad():
                    fp_im1 = self.encoder(im1)
                    fp_im0 = self.encoder(im0)
                    # module involves cuda extension which doesn't support amp
                    with torch.cuda.amp.autocast(enabled=False):
                        flow = self.flow(im0.float(), im1.float())['flow']
                with torch.cuda.amp.autocast(enabled=False):
                    fp_im1 = {k: v.float() for k, v in fp_im1.items()}
                    fp_im0 = {k: v.float() for k, v in fp_im0.items()}
                    bsf = self.bsf_project(self.bfp_module(
                        fp_im1, fp_im0, flow.float()))
                out_e = fp_im1[self.encoder_key_feature]
                bsf = F.adaptive_avg_pool2d(bsf, out_e.shape[-2:])
                out_e += bsf
            else:
                with torch.no_grad():
                    out_backbone = self.encoder(im1)
                    out_e = out_backbone[self.encoder_key_feature]
                    del out_backbone
                    if self.use_flow:
                        out_flow_feature = self.flow(im0, im1)['feature'].to(out_e.dtype)
                        if out_e.shape[-2] != out_flow_feature.shape[-2]:
                            out_flow_feature = out_flow_feature[:, :, :out_e.shape[-2], :out_e.shape[-1]]
                        out_e = torch.cat([out_e, out_flow_feature], dim=1)
                        del out_flow_feature

            out_perception.append(out_e)  # b x C_out x H x W

        # b x C_out x n_frames-1 x H x W
        out_perception = torch.stack(out_perception, dim=2)

        # process perception output with dynamics module
        # time range: t - n_history + 1 ... t ... t + n_future

        n_frames_perception = self.n_history + self.n_future

        recep_field = self.n_history
        out_dynamics = []

        i_last = 0
        for i in range(0, n_frames_perception-recep_field, recep_field):
            in_dynamics = out_perception[:, :, i:i+recep_field, :, :]
            # TODO: check whether need to pass stacked features through separate dynamics modules or the same one suffice?
            out_d = self.dynamics(in_dynamics)
            out_dynamics.append(out_d)
            i_last = i
            if not self.training or self.deterministic:
                break

        # add last dynamics feature for full future coverage
        if i_last+recep_field < n_frames_perception and self.training:
            in_dynamics = out_perception[:, :, n_frames_perception-recep_field:, :, :]
            # TODO: check whether need to pass stacked features through separate dynamics modules or the same one suffice?
            out_d = self.dynamics(in_dynamics)
            out_dynamics.append(out_d)

        # future predictions

        if not self.deterministic:
            if self.training:
                dists = self.cond_dist(out_dynamics[0], out_dynamics)
            else:
                dists = self.cond_dist(out_dynamics[0])
            dist_h, dist_f = dists['present_dist'], dists['future_dist']
        else:
            dist_h = None
            dist_f = None

        self.predictor_state = {'dist_h': dist_h,
                                'dist_f': dist_f,
                                'skip_encoder': None,
                                'out_dynamics': out_dynamics}

        if self.blender is not None:
            self.predictor_state['out_perception'] = out_perception[:, :, :5, :, :]

    def sample_prediction(self):

        assert self.predictor_state is not None, 'You need to condition the distributions by running FuturePredictor.condition(x) once with the current dataset!'
        if not self.deterministic:
            dist_h = self.predictor_state['dist_h']
            dist_f = self.predictor_state['dist_f']
        skip_encoder_0 = self.predictor_state['skip_encoder']
        out_dynamics = self.predictor_state['out_dynamics']

        if self.deterministic:
            sample_noise = None
        else:
            if self.training:
                sample_noise = dist_f.rsample()
            else:
                sample_noise = dist_h.sample()

        z_pred = self.future_generator(
            {'dynamics': out_dynamics[0], 'noise': sample_noise})['dynamics']
        n_future = self.n_future
        if self.predict_current:
            # insert encoder feature to the front
            out_dynamics_cur = out_dynamics[0].unsqueeze(1)
            z_pred = torch.cat([out_dynamics_cur, z_pred], dim=1)
            n_future += 1

        if self.blender is not None:
            z_b, z_t, z_c, z_h, z_w = z_pred.shape
            out_perception = self.predictor_state['out_perception']
            out_perception = out_perception.repeat_interleave(n_future, dim=0)
            z_pred = z_pred.reshape(z_b*z_t, z_c, z_h, z_w)
            z_pred = self.blender({'dynamics': z_pred, 'context': out_perception})
            if isinstance(z_pred, dict): # calculate loss for intermediate values by stacking
                                         # f2f, f2m and f2mf predictions in the batch dimension

                z_pred_f2f = z_pred['f2f'].reshape(z_b, z_t, z_c, z_h, z_w)
                z_pred_f2m = z_pred['f2m'].reshape(z_b, z_t, z_c, z_h, z_w)
                # z_pred_flow = z_pred['f2m_flow'].reshape(z_b, z_t, 2, z_h, z_w) # uncomment to calculate loss on flow prediction
                z_pred = z_pred['blend'].reshape(z_b, z_t, z_c, z_h, z_w)
            else:
                z_pred = z_pred.reshape(z_b, z_t, z_c, z_h, z_w)

        # repeat each skip connection n_future times along time dimension, and combine with batch dim
        decoder_feature = {}
        decoder_feature[self.encoder_key_feature] = z_pred.view(
            -1, *z_pred.shape[2:])

        out_tasks = {}
        for task_type, task_head in self.task_heads.items():
            out_flatten = task_head(decoder_feature)
            if self.out_resize:
                out_flatten = F.interpolate(
                    out_flatten, size=self.out_resize, mode='bilinear', align_corners=True)
            task_out = out_flatten.view(
                *z_pred.shape[:2], *out_flatten.shape[1:])
            out_tasks[task_type] = task_out

        # use intermediate output for f2mf loss calculation
        if 'z_pred_f2f' in locals() and 'z_pred_f2m' in locals():
            z_shape = z_pred_f2f.shape

            z_pred_f2f = z_pred_f2f.view(-1, *z_shape[2:])
            out_f2f = self.task_heads['semantic'] \
                ({self.encoder_key_feature: z_pred_f2f})
            if self.out_resize:
                out_f2f = F.interpolate(out_f2f, size=self.out_resize, mode='bilinear', align_corners=True)
            out_f2f = out_f2f.view(*z_shape[:2], *out_f2f.shape[1:])
            out_tasks['semantic_f2f'] = out_f2f

            z_pred_f2m = z_pred_f2m.view(-1, *z_shape[2:])
            out_f2m = self.task_heads['semantic'] \
                ({self.encoder_key_feature: z_pred_f2m})
            if self.out_resize:
                out_f2m = F.interpolate(out_f2m, size=self.out_resize, mode='bilinear', align_corners=True)
            out_f2m = out_f2m.view(*z_shape[:2], *out_f2m.shape[1:])
            out_tasks['semantic_f2m'] = out_f2m


        return out_tasks

    def get_distributions(self):
        assert self.predictor_state is not None, 'You need to condition the distributions by running FuturePredictor.condition(x) once with the current dataset!'
        return self.predictor_state['dist_h'], self.predictor_state['dist_f']

    def forward(self, x):
        """
        Forward method of the predictor module

        arguments:
        :param x: input images of shape [b x 3 x T x H x W]

        output:
        A dictionary
            'loss': Dictionary of losses. Backprop only needs to be done once on loss_total
            'out_tasks': Dictionary of task head outputs
        """
        images_gpu = x['image_seq']
        gt_sem = x['semantic_seq']

        self.condition(images_gpu)

        dist_h, dist_f = self.get_distributions()

        loss_total = 0.
        out_dict = {'loss': {}, 'out_tasks': {}}

        if self.training:
            if not self.deterministic:
                loss_prob = self.criterions['prob'](dist_f, dist_h) * self.weight_loss_prob
                # misc.check_value(loss_prob)
                loss_total += loss_prob
                out_dict['loss']['loss_prob'] = loss_prob.item()

        loss_samples = []
        loss_samples_f2m_flow = []
        out_tasks = {'semantic': [], 'flow': [], 'disparity': []}

        if self.deterministic or self.training:
            n_sample = 1
        else:
            n_sample = self.n_sample_eval

        for _ in range(n_sample):
            loss_sample = 0.
            predictions = self.sample_prediction()
            pred_sem = predictions['semantic']
            out_tasks['semantic'].append(pred_sem)
            # misc.check_value(pred_disp)
            # misc.check_value(pred_flow)

            loss_sem = self.criterions['semantic'](pred_sem, gt_sem)
            # [DIRTY] in case of calculate intermediate loss:
            if 'semantic_f2f' in predictions and 'semantic_f2m' in predictions:
                loss_sem += self.criterions['semantic'](predictions['semantic_f2f'], gt_sem) + \
                    self.criterions['semantic'](predictions['semantic_f2m'], gt_sem)
            loss_sample += loss_sem * self.weight_loss_pred


            if not isinstance(loss_sample, float):
                loss_samples.append(loss_sample.item())

            loss_total += loss_sample

        out_dict['loss']['loss_total'] = loss_total
        out_dict['loss']['loss_samples'] = loss_samples
        out_dict['out_tasks'] = out_tasks

        return out_dict

    def train(self, mode=True):
        """ we need to customize this function so that the pretrained perception submodules will always remain in evaluation mode to avoid changing its BN stats """
        super().train(mode)

        if mode:
            if self.freeze_backbone:
                self.encoder.eval()
            if self.flow is not None:
                self.flow.eval()

        return self
