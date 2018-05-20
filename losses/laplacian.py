import numpy as np
from keras import backend as K


def laplacian2d(nx, ny):

    L = np.zeros((nx*ny, nx*ny))

    def Kord(i, j):
        return i*ny+j

    for i in range(nx):
        for j in range(ny):
            k = Kord(i, j)
            L[k, k] = 4
            if i == 0 or j == 0 or i == nx-1 or j == ny-1:
                L[k, k] = 3
            if k == 0 or k == ny-1 or k == (nx-1)*ny or k == nx*ny-1:
                L[k, k] = 2
            if i-1 >= 0:
                h = Kord(i-1, j)
                L[k, h] = -1
            if i+1 < nx:
                h = Kord(i+1, j)
                L[k, h] = -1
            if j-1 >= 0:
                h = Kord(i, j-1)
                L[k, h] = -1
            if j+1 < ny:
                h = Kord(i, j+1)
                L[k, h] = -1

    return L


def laplacian2d_se_loss(y_true, y_pred):

    if not hasattr(laplacian2d_se_loss, "L"):
        batch_size, img_nrows, img_ncols, img_channels = K.int_shape(y_pred)
        laplacian2d_se_loss.L = K.constant(laplacian2d(img_nrows, img_ncols))

    L = laplacian2d_se_loss.L
    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)

    # Diagonal of dy.T @ L @ dy
    loss = K.sum(K.dot(y_pred-y_true, L) * (y_pred-y_true), axis=-1)
    return loss
