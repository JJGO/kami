import itertools

import numpy as np


class Batchify:

    def __init__(self, iterable, batch_size, labels=True):
        self.iterator = iter(iterable)
        self.batch_size = batch_size
        self.labels = labels

    def __iter__(self):
        return self

    def __next__(self):
        batch = itertools.islice(self.iterator, self.batch_size)
        if self.labels:
            X, Y = zip(*batch)
            return np.stack(X), np.stack(Y)
        else:
            return np.stack(batch)
