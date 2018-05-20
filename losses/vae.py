from keras import backend as K


def KL_unit_spherical_gaussian(_, y_pred):

    z_mean, z_log_var = y_pred
    kl_loss = (KL_unit_spherical_gaussian_mean(_, z_mean) +
               KL_unit_spherical_gaussian_logvar(_, z_log_var))
    return kl_loss


def KL_unit_spherical_gaussian_mean(_, y_pred):

    z_mean = y_pred
    kl_loss_mean = 0.5 * K.sum(K.square(z_mean), axis=-1)
    return kl_loss_mean


def KL_unit_spherical_gaussian_logvar(_, y_pred):

    z_log_var = y_pred
    kl_loss_sigma = - 0.5 * K.sum(1 + z_log_var - K.exp(z_log_var), axis=-1)
    return kl_loss_sigma
