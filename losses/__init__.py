from .volume import *
from .vae import *
from .laplacian import *

_loss_dict = {
    'crossentropy': binary_crossentropy,
    'mce':      mean_binary_crossentropy,
    'sdice':    dice_coef_loss,
    'mse':      mse_loss,
    'mae':      mae_loss,
    'se':       se_loss,
    'ae':       ae_loss,
    'ce':       binary_crossentropy,
    'l2dse':    laplacian2d_se_loss,
}
