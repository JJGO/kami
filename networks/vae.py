from keras import backend as K


def gauss_sample(args):
    mu, log_var = args
    shape = K.shape(mu)
    eps = K.random_normal(shape=shape, mean=0., stddev=1.)
    return mu + K.exp(log_var / 2) * eps
