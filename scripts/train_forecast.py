from futurepred.model.future_predictor import FuturePredictor
from futurepred.model.prediction import build_discriminator_from_cfg
from futurepred.datasets.cityscapes import Cityscapes
from futurepred.datasets.cityscapes_sequence import CityscapesSequence
from futurepred.datasets.transforms import CityscapesSeqCombinedTransforms
from futurepred.datasets.sampler import TrainingSampler
from futurepred.utils import metrics, meters
import futurepred.utils.comm
import futurepred.utils.model
import futurepred.utils.misc
from futurepred.utils.model_dict import ModelDict
import futurepred.criterions as crit
from futurepred.utils.logger import Logger
import futurepred.utils.training as training
from futurepred.utils.misc import make_one_hot_encoding
import futurepred.utils as utils
import futurepred.config.forecast as cfg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import numpy as np

from tqdm import tqdm
import argparse
import wandb
import os
import warnings
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description='Parameters to train perception module', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('config', type=str, help='config file')
parser.add_argument('--opts', default=[], nargs=argparse.REMAINDER)

logger_global = Logger.default_logger()


def save_checkpoint_on_interval(state, interval, checkpoint_path):
    last_save = save_checkpoint_on_interval.last_save
    time_now = datetime.now()
    if last_save is None:
        last_save = time_now
    elif time_now > last_save + timedelta(seconds=interval):
        logger_global.info(f"=> Saving checkpoint every {interval} seconds.")
        state.save(checkpoint_path)
        last_save = time_now
    save_checkpoint_on_interval.last_save = last_save
save_checkpoint_on_interval.last_save = None


def run_epoch(model_dict, loader, device_id,
              optimizers, grad_scaler,
              logger,
              state,
              cfg,
              mode='train', show_indices=[], model_dir=""):

    torch.autograd.set_detect_anomaly(cfg.RUNTIME.DETECT_ANOMALY)

    loss_meter = meters.AverageValueMeter()
    n_class = cfg.PREDICTOR.N_SEMANTIC_CLASS
    seqlen = cfg.PREDICTOR.N_FRAMES_FUTURE
    miou_avg_only = not cfg.RUNTIME.VERBOSE_IOU
    miou_meter = meters.SequenceMeter(metrics.MIoUMeter(n_class, avg_only=miou_avg_only), seq_len=seqlen) # not considering train_id = 255 in eval
    miou_meter_rf = meters.SequenceMeter(metrics.MIoUMeter(n_class, avg_only=miou_avg_only), seq_len=seqlen, suffix='rf') # for repeat frames

    if mode == 'train':
        for name, opt in optimizers.items():
            logger.info(f"Optimizer({name}) LR={[param_group['lr'] for param_group in opt.param_groups]}")
        train = True
    else:
        train = False

    start_idx = state.data_index // (cfg.MODEL.BS * utils.comm.get_world_size())
    for batch_idx, sample in enumerate(tqdm(loader), start=start_idx):
        images = torch.stack(sample['image_seq'], dim=1) # batch x T x 3 x H x W
        if batch_idx in show_indices and cfg.RUNTIME.WATCH_BLENDER_WEIGHTS:
            logger.blender_input_image = images[0][1:cfg.PREDICTOR.N_FRAMES_HISTORY+1]
            logger.blender_frames = cfg.PREDICTOR.N_FRAMES_FUTURE
            logger_handle = \
                utils.model.get_module(
                    model_dict['predictor'],
                    cfg.MODEL.DISTRIBUTED
                ).blender.bld_weights.register_forward_hook(
                    logger.log_blender_weights_hook()
                )
        images = images.permute(0,2,1,3,4)
        images_gpu = images.cuda(device_id)
        n_future = seqlen+1 if cfg.PREDICTOR.PREDICT_CURRENT else seqlen
        gt_sem = torch.stack(sample['semantic_seq'][-n_future:], dim=1).squeeze(dim=2).to(device_id)
        # b x T x H x W

        with torch.set_grad_enabled(train):

            if train:
                for opt in optimizers.values():
                    opt.zero_grad()

            with torch.cuda.amp.autocast(cfg.RUNTIME.USE_AUTOCAST):
                out_dict = model_dict['predictor']({
                    'image_seq': images_gpu,
                    'semantic_seq': gt_sem,
                    })

            # if using discriminator
            if train and 'discriminator' in model_dict:
                # TODO: dirty / here we assume there is only 1 sample during training
                discriminator = model_dict['discriminator']
                if cfg.DISCRIMINATOR.TYPE == 'semantic_video':
                    gt_sem_d = torch.stack(
                        sample['semantic_seq'][-(cfg.PREDICTOR.N_FRAMES_FUTURE+cfg.PREDICTOR.N_FRAMES_HISTORY):],
                        dim=1
                    ).squeeze(dim=2).to(device_id)
                    mask = (gt_sem_d != 255).unsqueeze(2)
                    gt_sem_d = make_one_hot_encoding(gt_sem_d, dim=2, n_class=19).to(device_id)
                    # combine predicted frames with gt history frames
                    pred_sem = torch.cat([
                        gt_sem_d[:, :cfg.PREDICTOR.N_FRAMES_HISTORY, ...],
                        out_dict['out_tasks']['semantic'][0]
                    ], dim=1)
                elif cfg.DISCRIMINATOR.TYPE == 'semantic':
                    pred_sem = out_dict['out_tasks']['semantic'][0]
                    gt_sem_d = gt_sem
                    mask = (gt_sem_d != 255).unsqueeze(2)
                    gt_sem_d = make_one_hot_encoding(gt_sem_d, dim=2, n_class=19).to(pred_sem)
                else:
                    raise ValueError(f"Unsupported discriminator {cfg.DISCRIMINATOR.TYPE}")

                # 1. optimize discriminator
                ## 1.1 'real' labels from groundtruth
                with torch.cuda.amp.autocast(cfg.RUNTIME.USE_AUTOCAST):
                    out_d = discriminator(gt_sem_d, True, mask)
                    loss_d_real = out_d['loss']
                    out_d_real = out_d['out'].mean()
                grad_scaler.scale(loss_d_real).backward()

                ## 1.2 'fake' labels from prediction
                with torch.cuda.amp.autocast(cfg.RUNTIME.USE_AUTOCAST):
                    out_d = discriminator(pred_sem.detach(), False, mask)
                    loss_d_fake = out_d['loss']
                    out_d_fake = out_d['out'].mean()
                grad_scaler.scale(loss_d_fake).backward()

                ## 1.3 update discriminator weights
                grad_scaler.step(optimizers['discriminator'])

                # 2. optimize generator
                with torch.cuda.amp.autocast(cfg.RUNTIME.USE_AUTOCAST):
                    out_d = discriminator(pred_sem, True, mask)
                    loss_g = out_d['loss']

                out_dict['loss']['loss_total'] += loss_g * cfg.PREDICTOR.WEIGHT_LOSS_GAN_G

                # save for logging
                out_dict['loss']['gan_d_real'] = loss_d_real.item()
                out_dict['loss']['gan_d_fake'] = loss_d_fake.item()
                out_dict['loss']['gan_g'] = loss_g.item()
                out_dict['out_tasks']['gan_d_real'] = out_d_real.item()
                out_dict['out_tasks']['gan_d_fake'] = out_d_fake.item()

            if train:
                grad_scaler.scale(out_dict['loss']['loss_total']).backward()
                grad_scaler.step(optimizers['predictor'])
                grad_scaler.update()


        if utils.misc.check_value(out_dict):
            continue

        # logging
        with torch.no_grad():
            logger.log({'loss_total': out_dict['loss']['loss_total']})
            if train:
                if not cfg.PREDICTOR.DETERMINISTIC:
                    logger.log({'loss_prob': out_dict['loss']['loss_prob']})
                if cfg.DISCRIMINATOR.ENABLE:
                    logger.log({
                        'out_gan_d_real': out_dict['out_tasks']['gan_d_real'],
                        'out_gan_d_fake': out_dict['out_tasks']['gan_d_fake'],
                        'loss_gan_d_real': out_dict['loss']['gan_d_real'],
                        'loss_gan_d_fake': out_dict['loss']['gan_d_fake'],
                        'loss_gan_g': out_dict['loss']['gan_g'],
                    })
                logger.log({'grad_scaler_scale': grad_scaler.get_scale()})
            loss_meter.add(out_dict['loss']['loss_total'])

            n_samples = len(out_dict['out_tasks']['semantic'])
            for sample_idx, (pred_sem, loss_sample) in enumerate(zip(out_dict['out_tasks']['semantic'], out_dict['loss']['loss_samples'])):
                logger.log({'loss_sample': loss_sample})

                for meter, meter_rf, pred, gt in [(miou_meter, miou_meter_rf, pred_sem, gt_sem)]:
                    result = meter.add(pred, gt)
                    logger.log(result)

                    pred_rf = utils.misc.repeat_frames(pred[:,0,...], n_future)
                    result_rf = meter_rf.add(pred_rf, gt)
                    logger.log(result_rf)

                if batch_idx in show_indices:
                    image_log = images[0].permute(1,0,2,3)[-n_future:]
                    logger.log_prediction_seq(image_seq=image_log,
                                              sem_gt_seq=gt_sem[0],
                                              sem_logit_seq=pred_sem[0],
                                              vis_entropy=True,
                                              commit=(sample_idx==n_samples-1)
                                              )

                    if cfg.RUNTIME.WATCH_BLENDER_WEIGHTS:
                        logger.blender_input_image = None
                        logger.blender_frames = 0
                        logger_handle.remove()

                logger.commit()

        if train and cfg.MODEL.CHECKPOINT is not None \
            and utils.comm.is_main_process() \
            and batch_idx != len(loader)-1:
            state.data_index = (batch_idx+1) * cfg.MODEL.BS * utils.comm.get_world_size()
            save_checkpoint_on_interval(state, cfg.MODEL.CHECKPOINT_INTERVAL, cfg.MODEL.CHECKPOINT)

    result_dict = dict(
        loss=loss_meter.value(),
        miou=miou_meter.value(),
        miou_rf=miou_meter_rf.value()
    )

    return result_dict


def run(model_dict, optimizers, grad_scaler, train_loader, valid_loader, device_id, state, cfg):

    mute = False if utils.comm.is_main_process() and not cfg.RUNTIME.VERBOSE else True
    logger_train = Logger(f"train", mute)
    logger_valid = Logger(f"valid", mute)

    if utils.comm.is_main_process():
        if wandb.run.name:
            model_save_dir = os.path.join(cfg.MODEL.MODEL_DIR, wandb.run.name)
        else:
            model_save_dir = os.path.join(cfg.MODEL.MODEL_DIR, 'dryrun')

    try:
        for epoch in range(state.epoch, cfg.OPTIMIZER.MAX_EPOCH):
            logger_global.info("Beginning epoch {}".format(epoch))

            if not cfg.MODEL.EVAL:
                logger_train.info("training...")
                model_dict.train()
                result_train = run_epoch(model_dict, train_loader, device_id,
                                         optimizers, grad_scaler,
                                         logger_train, state,
                                         cfg,
                                         mode='train', show_indices=cfg.RUNTIME.SHOW_INDEX_TRAIN)
                logger_train.info("training stats: {}".format(result_train))

            logger_valid.info("evaluating...")
            model_dict.eval()
            state.data_index = 0
            result_eval = run_epoch(model_dict, valid_loader, device_id,
                                    optimizers, grad_scaler,
                                    logger_valid, state,
                                    cfg,
                                    mode='valid', show_indices=cfg.RUNTIME.SHOW_INDEX_VAL)
            logger_valid.info("validation stats: {}".format(result_eval))

            if cfg.MODEL.EVAL:
                # if run model for evaluation, one epoch is enough
                break

            # save checkpoint
            state.epoch = epoch + 1 # this epoch is done
            state.data_index = 0    # and reset data_idx
            if utils.comm.is_main_process():
                if result_eval['miou']['average_miou_9'] > state.best_score:
                    state.best_score = result_eval['miou']['average_miou_9']
                    model_save_path = os.path.join(model_save_dir, "best_model.pth")
                    logger_global.info("saving new best model to {}".format(model_save_path))
                    state.save(model_save_path)

    except Exception:
        warnings.warn(f"Exception during training:")
        raise
    finally:
        if utils.comm.is_main_process() and cfg.MODEL.CHECKPOINT is not None:
            logger_global.info(f"Saving emergency checkpoint file ..")
            state.save('checkpoint_exception.pth')
        warnings.warn("Training finished.")


def build_model(cfg, device_id, criterions):
    """instantiate the model and move it to corresponding devices as defined by cfg"""

    model_dict = ModelDict()

    predictor = FuturePredictor(
        cfg.PREDICTOR,
        criterions['predictor']
    ).cuda(device_id)
    model_dict['predictor'] = predictor

    if cfg.DISCRIMINATOR.ENABLE:
        discriminator = build_discriminator_from_cfg(
            cfg.DISCRIMINATOR,
            criterions['discriminator']
        )
        discriminator = discriminator.cuda(device_id)
        model_dict['discriminator'] = discriminator

    if cfg.MODEL.DISTRIBUTED:
        for key, model in model_dict.items():
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model_dict[key] = DistributedDataParallel(
                model,
                device_ids=[device_id],
                output_device=device_id,
                find_unused_parameters=True # TODO: exception raised for EfficientPS without this
                                            # argument. Why?
            )

    return model_dict


def build_optimizers(cfg, model_dict):

    optimizers = {}

    optimizers['predictor'] = optim.Adam(model_dict['predictor'].parameters(), cfg.OPTIMIZER.LR)

    if 'discriminator' in model_dict:
        optimizers['discriminator'] = optim.Adam(model_dict['discriminator'].parameters(), cfg.OPTIMIZER.LR)

    return optimizers


def build_grad_scaler(cfg):

    return torch.cuda.amp.GradScaler(
        init_scale=cfg.GRAD_SCALER.INIT_SCALE,
        enabled=cfg.GRAD_SCALER.ENABLE
    )


def build_dataloader(cfg):
    cfg_data = cfg.DATASET
    train_transforms = CityscapesSeqCombinedTransforms(flip_prob=cfg_data.HFLIP_PROB_TRAIN,
                                                 normalize_mean=cfg_data.IMAGE_NORM_MEAN,
                                                 normalize_std=cfg_data.IMAGE_NORM_STD,
                                                 scale_range=cfg_data.RAND_SCALE_TRAIN,
                                                 crop_size=cfg_data.CROP_SIZE,
                                                 image_resize_ratio=cfg_data.IMAGE_RESIZE_RATIO,
                                                 target_resize_ratio=cfg_data.TARGET_RESIZE_RATIO,
                                                 use_log_depth=cfg_data.USE_LOG_DEPTH,
                                                 image_pad_size=cfg_data.IMAGE_PAD_SIZE,
                                                 target_pad_size=cfg_data.TARGET_PAD_SIZE
                                                 )
    val_transforms = CityscapesSeqCombinedTransforms(flip_prob=cfg_data.HFLIP_PROB_VAL,
                                                 normalize_mean=cfg_data.IMAGE_NORM_MEAN,
                                                 normalize_std=cfg_data.IMAGE_NORM_STD,
                                                 scale_range=cfg_data.RAND_SCALE_VAL,
                                                 crop_size=cfg_data.CROP_SIZE,
                                                 image_resize_ratio=cfg_data.IMAGE_RESIZE_RATIO,
                                                 target_resize_ratio=cfg_data.TARGET_RESIZE_RATIO,
                                                 use_log_depth=cfg_data.USE_LOG_DEPTH,
                                                 image_pad_size=cfg_data.IMAGE_PAD_SIZE,
                                                 target_pad_size=cfg_data.TARGET_PAD_SIZE
                                                 )

    if cfg_data.TYPE in ['cityscapes-seq', 'cityscapes-seq-eps']:
        train_data = CityscapesSequence(cfg.DATASET.PATH, split='train', transforms=train_transforms, target_types=cfg_data.TARGETS, sequence_range=(19-cfg.PREDICTOR.N_FRAMES_HISTORY, 19+cfg.PREDICTOR.N_FRAMES_FUTURE+1))
        valid_data = CityscapesSequence(cfg.DATASET.PATH, split='val', transforms=val_transforms, target_types=cfg_data.TARGETS, sequence_range=(19-cfg.PREDICTOR.N_FRAMES_HISTORY, 19+cfg.PREDICTOR.N_FRAMES_FUTURE+1))

    train_sampler = TrainingSampler(len(train_data), shuffle=True, seed=cfg.RUNTIME.MANUAL_SEED)

    train_loader = DataLoader(train_data, batch_size=cfg.MODEL.BS, num_workers=cfg.RUNTIME.WORKERS, drop_last=True, sampler=train_sampler)
    valid_loader = DataLoader(valid_data, batch_size=cfg.MODEL.BS, shuffle=False, num_workers=cfg.RUNTIME.WORKERS, drop_last=True)

    return train_loader, valid_loader


def build_criterions(cfg, device_id):
    crit_dict = {}

    # predictor
    crit_pred = {}
    ## semantic
    class_weights = {0: 3.362088928135997, 1: 14.031521298730318, 2: 4.986657918172686, 3: 39.254403222891234, 4: 36.5125971773311, 5: 32.89620795239199, 6: 46.286660134462245, 7: 40.69042748040039, 8: 6.698241903441155, 9: 33.55545414377673, 10: 18.487832644189325, 11: 32.97431249303082, 12: 47.676506488107115, 13: 12.70028597336979, 14: 45.20542136324199, 15: 45.78372411642551, 16: 45.825290445040096, 17: 48.40614734589367, 18: 42.75592219573717}
    class_weights = torch.tensor(list(class_weights.values())).cuda(device_id)
    crit_pred['semantic'] = crit.DiscountedLoss(
        nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='none'),
        cfg.CRITERION.DISCOUNT_FACTOR
    )

    ## probabilistic
    crit_pred['prob'] = lambda p, q : torch.distributions.kl_divergence(p, q).mean()
    crit_dict['predictor'] = crit_pred

    # discriminator
    if cfg.DISCRIMINATOR.ENABLE:
        if cfg.DISCRIMINATOR.TYPE == 'semantic':
            crit_disc = crit.DiscountedLoss(
                nn.BCEWithLogitsLoss(reduction='none'),
                cfg.CRITERION.DISCOUNT_FACTOR
            )
            # crit_disc.dump_intermediate = True
        elif cfg.DISCRIMINATOR.TYPE == 'semantic_video':
            crit_disc = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported discriminator {cfg.DISCRIMINATOR.TYPE}")
        crit_dict['discriminator'] = crit_disc

    return crit_dict


def main():
    if cfg.RUNTIME.MANUAL_SEED:
        torch.manual_seed(cfg.RUNTIME.MANUAL_SEED)
        logger_global.info(f"Using manual seed: {cfg.RUNTIME.MANUAL_SEED}")

    # select cuda device
    if cfg.MODEL.DISTRIBUTED:
        device_id = int(os.environ['LOCAL_RANK'])
    else:
        device_id = cfg.RUNTIME.DEVICE_ID

    # initialize torch distributed if using multi-gpu training
    if cfg.MODEL.DISTRIBUTED:
        logger_global.info("Initializing parallel training...")
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://',
                                             timeout=timedelta(seconds=60))
        logger_global.info(f"Init done. RANK={dist.get_rank()} WORLD={dist.get_world_size()} LOCAL_RANK={device_id}")

    torch.cuda.set_device(device_id)

    if utils.comm.is_main_process():
        wandb.init(project='fr-panoptic-forecast-whole', name=cfg.EXPERIMENT_NAME)
        wandb.config.update(cfg)
    else:
        logger_global.mute()

    train_loader, valid_loader = build_dataloader(cfg)
    criterions = build_criterions(cfg, device_id)
    model_dict = build_model(cfg, device_id, criterions)
    optimizers = build_optimizers(cfg, model_dict)
    grad_scaler = build_grad_scaler(cfg)
    state = training.State(model_dict) # TODO
    if cfg.MODEL.LOAD:
        state.load(cfg.MODEL.LOAD, device_id, strict=cfg.MODEL.STRICT_LOAD)
        train_loader.batch_sampler.sampler._epoch = state.epoch
        train_loader.batch_sampler.sampler._last_index = state.data_index

    if utils.comm.is_main_process():
        wandb.watch(model_dict.get_submodules(), log='all', log_freq=5)
    run(model_dict, optimizers, grad_scaler, train_loader, valid_loader, device_id, state, cfg)

if __name__ == "__main__":
    args = parser.parse_args()
    cfg.update_from_args(args)
    main()
