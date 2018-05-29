import keras.backend as K

__EPS = K.epsilon()
__THRESHOLD = 0.5


# Decorator to make losses N-dimensional elementwise
def volume_loss(loss):

    def wrapped_loss(y_true, y_pred):
        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)
        return loss(y_true, y_pred)

    return wrapped_loss


@volume_loss
def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


@volume_loss
def sum_binary_crossentropy(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


def weighted_binary_crossentropy(one_weight, zero_weight=1):

    @volume_loss
    def _weighted_img_binary_crossentropy(y_true, y_pred):
        y_pred = K.clip(y_pred, __EPS, 1-__EPS)
        L = -(one_weight * y_true * K.log(y_pred) +
              zero_weight * (1-y_true) * K.log(1-y_pred))
        L = K.mean(L, axis=-1)
        return L

    return _weighted_img_binary_crossentropy


@volume_loss
def soft_dice_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=-1)
    dice = (2.0 * intersection + __EPS) / (K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) + __EPS)
    return dice


def soft_dice(y_true, y_pred):
    return 1-K.mean(soft_dice_coef(y_true, y_pred), axis=-1)


def dice_coef(y_true, y_pred):
    y_pred = K.round(y_pred + 0.5 - __THRESHOLD)
    return soft_dice_coef(y_true, y_pred)


@volume_loss
def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


@volume_loss
def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


@volume_loss
def se(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)


@volume_loss
def ae(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


@volume_loss
def L1Norm(y_true, y_pred):
    return K.sum(K.abs(y_pred), axis=-1)


def total_variation2d(_, y_pred):
    x = y_pred
    batch_size, img_nrows, img_ncols, img_channels = K.int_shape(x)
    assert K.ndim(x) == 4
    a = K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    a = K.batch_flatten(a)
    b = K.batch_flatten(b)
    c = K.mean(a, axis=-1) + K.mean(b, axis=-1)
    return c
