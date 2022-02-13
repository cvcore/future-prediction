import torch
import torch.nn as nn
import os

def total_parameters(model: nn.Module, trainable_only=False):
    assert isinstance(model, nn.Module)
    return sum(p.numel() for p in model.parameters()
                            if p.requires_grad or not trainable_only)


def save(checkpoint, path):
    """ save checkpoint with directory creation """
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(checkpoint, path)


def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/
    # add L2 weight decay regulation only to the weights

    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]


def freeze(net):
    for param in net.parameters():
        param.requires_grad = False


def get_state_submodule(state_dict, prefix, remove_prefix=True):
    if remove_prefix:
        res = {k.replace(prefix+'.', ''):v for k, v in state_dict.items() if k.startswith(prefix)}
    else:
        res = {k:v for k, v in state_dict.items() if k.startswith(prefix)}
    return res


def get_module(model, distributed):
    if distributed:
        return model.module
    else:
        return model


def remove_submodule(state_dict, prefix):
    """ return new state_dict with submodules in ``state_dict`` with ``prefix`` removed """
    return {k:v for k, v in state_dict.items() if not k.startswith(prefix)}
