import torch
import numpy as np
import warnings
from .logger import Logger

logger = Logger.default_logger()

def check_value(value, raise_exception=False):
    """ Check if value contains Inf or NaN, if found will return True """
    if isinstance(value, torch.Tensor):
        if torch.isnan(value).any():
            logger.error("Found NaN's in Tensor!")
            assert not raise_exception
            return True
        if torch.isinf(value).any():
            logger.error("Found Inf's in Tensor!")
            assert not raise_exception
            return True
    elif isinstance(value, dict):
        for val in value.values():
            if check_value(val, raise_exception):
                return True
        return False
    return False

def repeat_frames(input, n_frames, dim_t=1):
    """ repeat a input of shape [b x ... x H x W] n_frames times to generate a tensor of [b x n_frames x ... x H x W]
    """
    input = input.unsqueeze(dim_t)
    input = torch.repeat_interleave(input, n_frames, dim=dim_t)
    return input

def combine_dim(input, start_dim, end_dim):
    """ combine every dimension of input tensor between start_dim and end_dim into a one dimension
        e.g. input.shape = [3,4,5,6,7], start_dim=0, end_dim=2
             output.shape = [12,5,6,7]
    """
    shape = input.shape
    new_shape = [*shape[:start_dim], -1, *shape[end_dim:]]
    return torch.reshape(input, new_shape)


def make_one_hot_encoding(input, dim, n_class):
    """produce one-hot encoded tensor from input

    Args:
        input (Tensor): int tensor for conversion to one-hot encoding.
            For an input with shape [D0, D1, ..., Dn-1], the output
            tensor will have shape [D0, D1, ..., Dm, ..., Dn]
            where Dm == n_class and m == dim
        dim (int): dimension to generate one-hot encoding
        n_class: output classes for the one-hot tensor. Assumes each
            element from input is in range [0, n_class-1]

    Returns:
        nn.Tensor: one-hot encoded tensor placed on the same device
            as input
    """

    out_shape = list(input.shape)
    out_shape.insert(dim, n_class)
    out = torch.zeros(out_shape).to(input.device)

    index = input.unsqueeze(dim).detach().clone()
    index[index >= n_class] = 0
    src = torch.ones_like(index).to(out)
    out.scatter_(dim=dim, index=index, src=src)

    return out
