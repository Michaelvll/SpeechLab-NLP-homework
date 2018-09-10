import numpy as np
import random
import torch
from .common import sort_data


class Batch(object):
    def __init__(self, size=0, batch_size=0):
        self.init_batch_idxs(size)
        self.now_idx = 0
        self.size = size
        self.batch_size = batch_size
        self.epoch = 0

    def init_batch_idxs(self, size):
        self.batch_idxs = list(range(size))
        random.shuffle(self.batch_idxs)

    def get_batch(self, data):
        start = self.now_idx * self.batch_size
        end = start + self.batch_size
        new_data = tuple()
        if isinstance(data, tuple):
            assert len(data) == 2
            x, y = data
            if isinstance(x, torch.Tensor):
                new_data += (x[self.batch_idxs[start:end]], )
                # print(x[self.batch_idxs[start:end]].type())
                data = (y, )
        else:
            if isinstance(data, torch.Tensor):
                new_data += (data[self.batch_idxs[start:end]],)
            data = tuple()
        for d in data:
            new_d = [d[x]
                     for x in self.batch_idxs[start:end]]
            new_data += (new_d, )
            # print(new_data)
        return new_data

    def sorted_batch(self, data, reverse=True):
        new_data = self.get_batch(data)
        return sort_data(new_data, reverse)

    def next(self):
        self.now_idx += 1
        if self.now_idx >= (self.size // self.batch_size):
            self.init_batch_idxs(self.size)
            self.now_idx = 0
            self.epoch += 1
