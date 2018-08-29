import numpy as np

from tqdm import tqdm

from .batch import Batchify


def _cycle_generator(array):
    while True:
        for i in array:
            yield i


def _double_cycle_generator(X, Y):
    while True:
        for x, y in zip(X, Y):
            yield x, y


def npz_reader(file):
        cache = np.load(file)
        train_generator = _double_cycle_generator(cache['X_train'], cache['y_train'])
        val_generator = _double_cycle_generator(cache['X_val'], cache['y_val'])
        test_generator = _double_cycle_generator(cache['X_test'], cache['y_test'])
        generators = [train_generator, val_generator, test_generator]
        return generators


def npz_writer(file,
               data,
               n_samples,
               proportional):
        sizes = n_samples * np.ones(3)
        if proportional:
            sizes *= np.array([1,
                               data.val_size/data.train_size,
                               data.test_size/data.train_size])
        sizes = sizes.astype(int)

        tosave = {}

        for i, group in enumerate(['train', 'val', 'test']):
            size = sizes[i]
            iterator = tqdm(data.getattr(f"{group}_iterator"), total=size)
            X, Y = next(Batchify(iterator, size))
            tosave[f'X_{group}'] = X
            tosave[f'y_{group}'] = Y

        tosave['params'] = data.params

        np.savez(file, *tosave)
