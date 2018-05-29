import numpy as np
import keras.backend as K


def combined_loss(losses, weights=None):

    if weights is None:
        weights = np.ones(len(losses))

    def _combined_loss(y_true, y_pred):
        total_loss = K.constant(0.)
        for loss, weight in zip(losses, weights):
            total_loss += K.constant(weight) * loss(y_true, y_pred)
        return total_loss

    return _combined_loss


def gated_loss(loss):

    def _gated_loss(y_true, y_pred):

        orig_loss = loss(y_true, y_pred)
        gates = np.ones(len(y_true))
        for i, y in enumerate(y_true): # across batch examples
            if np.any(np.isnan(y)):
                gates[i] = 0
        gates = K.constant(gates)
        return orig_loss * gates

    return _gated_loss
