import numpy as np


def _hard_binary_metric(metric):

    def _metric(y_true, y_pred):
        y_pred = np.clip(np.round(y_pred), 0, 1)
        y_true = np.clip(np.round(y_true), 0, 1)
        # assert np.all(np.unique(y_true) == np.unique([0, 1]))
        # assert np.all(np.unique(y_pred) == np.unique([0, 1]))
        return metric(y_true, y_pred)

    return _metric


def soft_dice_coef(y_true, y_pred):
    intersection = np.sum(y_pred * y_true)
    total = np.sum(y_true) + np.sum(y_pred)
    return 2*intersection / total


@_hard_binary_metric
def dice_coef(y_true, y_pred):
    return soft_dice_coef(y_true, y_pred)


@_hard_binary_metric
def jaccard_coef(y_true, y_pred):
    intersection = np.sum(y_pred * y_true)
    union = np.sum(np.clip(y_true+y_pred, 0, 1))
    return intersection / union


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))


def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)


def binary_crossentropy(y_true, y_pred):
    y_true = np.clip(np.round(y_true), 0, 1)
    eps = np.finfo(y_pred.dtype).eps
    y_pred = np.clip(y_pred, eps, 1-eps)
    L = y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred)
    return -np.mean(L)


def mean_total_variation(x, norm=2, channels_last=False):
    if channels_last:
        return np.sum([mean_total_variation(x[..., i] for i in range(x.shape[-1]))])

    x = np.squeeze(x)
    N = len(x.shape)
    corner_slice = tuple(slice(None, -1, None) for _ in range(N))

    tvl = np.zeros((N,)+x[corner_slice].shape)
    for d in range(N):
        moving_slice = tuple(slice(1, None, None) if i == d else s for i, s in enumerate(corner_slice))
        tvl[d] += x[corner_slice] - x[moving_slice]

    tvl = np.linalg.norm(tvl, axis=0, ord=norm)
    return np.mean(tvl)
