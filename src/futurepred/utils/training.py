import torch
from torch.nn.parallel import DistributedDataParallel
import os
from .logger import Logger
import numpy as np
from . import model
from pathlib import Path
from argparse import Namespace
import yaml

logger = Logger.default_logger()

class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self, model_dict):
        self.epoch = 0
        self.data_index = 0
        self.best_score = 0.
        self.model_dict = model_dict

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::
        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            "data_index": self.data_index,
            "best_score": self.best_score,
            "state_dict": self.model_dict.state_dict(),
        }

    def apply_snapshot(self, obj, device_id, strict=True):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """

        if 'epoch' in obj and strict:
            self.epoch = obj["epoch"]
            logger.info('epoch loaded')
        else:
            logger.warning('epoch skipped or nonexist')

        if 'data_index' in obj and strict:
            self.data_index = obj["data_index"]
            logger.info('data_index loaded')
        else:
            logger.warning('data_index skipped or nonexist')

        if 'best_score' in obj and strict:
            self.best_score = obj["best_score"]
            logger.info('best_score loaded')
        else:
            logger.warning('best_score skipped or nonexist')

        if 'state_dict' in obj:
            state_model_dict = obj['state_dict']
            for sub_module, state_dict in state_model_dict.items():
                is_dist_ckp = 'module' in list(state_dict.keys())[0]
                model_dist = self.model_dict.is_distributed()
                if is_dist_ckp and not model_dist:
                    logger.info('Loaded snapshot from distributed training, converting to non-distributed..')
                    state_dict = model.get_state_submodule(state_dict, 'module', remove_prefix=True)
                    self.model_dict[sub_module].load_state_dict(state_dict, strict=strict)
                elif not is_dist_ckp and model_dist:
                    logger.info('Loaded snapshot from non-distributed training, converting to distributed..')
                    self.model_dict[sub_module].module.load_state_dict(state_dict, strict=strict)
                else:
                    self.model_dict[sub_module].load_state_dict(state_dict, strict=strict)
                    logger.info('state_dict loaded')

    def save(self, f):
        logger.info(f"=> saving checkpoint file: {f}")
        path = Path(f).resolve()
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        torch.save(self.capture_snapshot(), str(path))

    def load(self, file, device_id, remove_submodule=None, remove_key=None, strict=True):
        # Map model to be loaded to specified single gpu.
        if remove_submodule is None:
            remove_submodule = []
        if remove_key is None:
            remove_key = []

        logger.info(f"=> loading checkpoint file: {file}")
        if os.path.isfile(file):
            snapshot = torch.load(file, map_location=f"cuda:{device_id}")

            for sub_module, sub_snapshot in snapshot.items():
                for submodule in remove_submodule:
                    sub_snapshot['state_dict'] = model.remove_submodule(sub_snapshot['state_dict'], submodule)
                for key in remove_key:
                    logger.info(f"=> Removing {key} from checkpoint")
                    sub_snapshot.pop(key, None)
                snapshot[sub_module] = sub_snapshot

            self.apply_snapshot(snapshot, device_id, strict)
            logger.info(f"=> loaded checkpoint file: {file}")
        else:
            logger.warning(f"=> checkpoint file not found: {file}. Will train from scratch")


class Config(Namespace):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, str):
            self.parse_from_file(config)
        else:
            self.parse_config(config)

    def parse_config(self, config):
        for key, value in config.items():
            if isinstance(value, (list, tuple)):
                setattr(self, key, [Config(x) if isinstance(x, dict) else x for x in value])
            else:
                setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def parse_from_file(self, path):
        with open(path) as f:
            config = yaml.safe_load(f)
        self.parse_config(config)

    global_cfg_ = None
    @classmethod
    def get_global_config(cls):
        return cls.global_cfg_

    @classmethod
    def parse_global_config(cls, path):
        cls.global_cfg_ = Config(path)
