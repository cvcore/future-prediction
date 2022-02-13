import torch
from torch.utils.data.sampler import Sampler
from futurepred.utils import comm
import itertools


class TrainingSampler(Sampler):
    """
    Returns an unified sampler for both distributed and non-distributed training.
    It supports loading previous sampling progress from checkpoint with the ``last_epoch`` and ``last_iteration`` arguments.
    This sampler will drop last batch by default.
    """

    def __init__(self, size, shuffle=True, seed=None, last_epoch=0, last_index=0):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
            last_epoch (int): number of epochs finished during last training session.
            last_index (int): number of indices trained during last session within an epoch,
                which equals the last training iteration times world size.
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self._epoch = last_epoch
        self._last_index = last_index
        self._index = 0

    def __iter__(self):
        start = self._rank + self._last_index
        indices = self._epoch_iterator()[start:None:self._world_size]
        for i, index in enumerate(indices):
            if i == len(indices) - 1:
                self._index = 0
                self._last_index = 0
                self._epoch += 1
            else:
                self._index += self._world_size
            yield index

    def __len__(self):
        return (self._size - self._last_index) // self._world_size

    def _epoch_iterator(self):
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch)
        if self._shuffle:
            return torch.randperm(self._size, generator=g).tolist()
        else:
            return list(torch.arange(self._size))

    # def get_epoch(self):
    #     return self._epoch

    # def get_index(self):
    #     # TODO: in case the number of workers is larger than batch size, more data will be sampled than those actually trained.
    #     #       and due to this reason, self._index can't record the number of data trained. So for creating the checkpoint,
    #     #       batch_index in the main training loop is used.
    #     return self._index
