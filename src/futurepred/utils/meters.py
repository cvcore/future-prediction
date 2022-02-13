import math
import numpy as np
import torch
from copy import deepcopy


class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    '''

    def reset(self):
        raise NotImplementedError

    def add(self, *args, **kwargs):
        '''Log a new value to the meter, calls underlying criterion and returns the result as a dictionary for wandb
        Args:
            value: Next result to include.
        Returns:
            dictionary of values calculated over the current batch
        '''
        raise NotImplementedError

    def value(self):
        """ get the calculation result from last reset() till now """
        raise NotImplementedError

class SequenceMeter(object):
    """ Meter for calculation over sequential inputs. It calculates evaluation result for each time t
        expected input dim:
            pred / gt - [b x T x ... x H x W]. T, H, W should be same for pred and gt
    """

    def __init__(self, meter, seq_len, suffix=''):
        """ Arguments:
        meter (Meter): instance of a Meter class
        suffix (str): suffix for each key of the result dictionary
        """
        assert isinstance(meter, Meter)
        seq_meter = [deepcopy(meter) for _ in range(seq_len)]
        self.meter = seq_meter
        self.suffix = suffix
        self.seq_len = seq_len

    def add(self, pred, gt):
        assert isinstance(pred, torch.Tensor) and isinstance(gt, torch.Tensor)
        assert self.seq_len == pred.shape[1] == gt.shape[1] and pred.shape[-2:] == gt.shape[-2:], "pred and gt should be same sequential length!"

        results_final = {}
        for t in range(self.seq_len):
            results = self.meter[t].add(pred[:,t,...], gt[:,t,...])
            results = {k+self._get_suffix(t):v for k,v in results.items()}
            results_final.update(results)

        return results_final

    def value(self):
        results_final = {}
        for t in range(self.seq_len):
            results = self.meter[t].value()
            results = {k+self._get_suffix(t):v for k,v in results.items()}
            results_final.update(results)

        return results_final

    def _get_suffix(self, t):
        if self.suffix != '':
            return '_{}_{}'.format(t, self.suffix)
        else:
            return '_{}'.format(t)

class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        value = float(value)
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return {'mean':self.mean, 'std':self.std}

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
