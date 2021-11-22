import torch
from torch.utils.data.distributed import DistributedSampler
#import random
#import numpy as np
import math
import torch
import torch.distributed as dist
class ShuffleDistributedSampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset_len=None, num_replicas=None, rank=None, seed=0, drop_last=True,shuffle=True):
        #super(ShuffleDistributedSampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=True)
        #self.sampler = sampler
        self.dataset_len = dataset_len

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if dataset_len is not None:
            self.dataset_len = dataset_len
        elif self.dataset is not None:
            self.dataset_len = len(self.dataset)
        else:
            raise RuntimeError("Length of the dataset is unknown.")

        if self.drop_last and self.dataset_len % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (self.dataset_len - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(self.dataset_len / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_len, generator=g).tolist()  # type: ignore
            #seed = self.seed + self.epoch
            #random.seed(seed)
            #print('Epoch: ', self.epoch, 'Indices: ',indices)
            #print('Seed: ', seed, 'Indices before: ',indices)
            #random.shuffle(indices) # guarantee that use different images for each epoch 
            #print('Seed: ', seed, 'Indices after: ',indices)
        else:
            indices = list(range(self.dataset_len))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        #print('Epoch: ', self.epoch, ', Rank: ',self.rank, ' ', indices)
        return iter(indices)

    def get_indices(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            if self.dataset_len is None:
                indices = torch.randperm(self.dataset_len, generator=g).tolist()  # type: ignore
            else:
                indices = torch.randperm(self.dataset_len, generator=g).tolist()  # type: ignore
        else:
            indices = list(range(self.dataset_len))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        # print('Epoch: ', self.epoch, ', Rank: ',self.rank, ' ', indices)
        return indices
