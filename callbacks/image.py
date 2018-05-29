import os
import subprocess

from scipy.misc import imsave
import numpy as np

from keras.callbacks import Callback
from ..util.font import print_text

devnull = open(os.devnull, 'w')


class SavePredictImage(Callback):

    def __init__(self, logdir, X, Y, frequency=1):
        super().__init__()
        self.logdir = logdir
        self.X = X
        self.Y = Y
        self.frequency = frequency

    def on_train_begin(self, logs={}):
        os.makedirs(self.logdir, exist_ok=True)
        for j, y in enumerate(self.Y):
            filename = "{:02d}_gt.png".format(j)
            filename = os.path.join(self.logdir, filename)
            imsave(filename, y[..., 0])

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == 0:
            Y = self.model.predict(self.X)
            for j, y in enumerate(Y):
                filename = "{:02d}_epoch{:03d}.png".format(j, epoch)
                filename = os.path.join(self.logdir, filename)
                imsave(filename, y[..., 0])


class TrainTestMosaic(Callback):

    def __init__(self, logdir, X_train, Y_train, X_test, Y_test, frequency=1, inverse=False, threshold=None, index=None):
        super().__init__()
        self.logdir = logdir
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.frequency = frequency
        self.inverse = inverse
        self.threshold = threshold
        if index is None:
            self.predict_fn = lambda x: self.model.predict(x)
        else:
            self.predict_fn = lambda x: self.model.predict(x)[index]

        B1, H1, W1, C1 = self.X_train.shape
        B2, H2, W2, C2 = self.Y_train.shape

        assert B1 == B2
        assert H1 == H2
        assert W1 == W2
        self.W = W1
        self.H = H1
        self.C1, self.C2 = C1, C2
        self.C = max(self.C1, self.C2)

        if self.C1 > self.C2:
            self.Y_train = np.repeat(self.Y_train, self.C1, axis=-1)
            self.Y_test = np.repeat(self.Y_test, self.C1, axis=-1)
        elif self.C1 < self.C2:
            self.X_train = np.repeat(self.X_train, self.C2, axis=-1)
            self.X_test = np.repeat(self.X_test, self.C2, axis=-1)

    def on_train_begin(self, logs={}):
        os.makedirs(self.logdir, exist_ok=True)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == 0:

            if self.C1 > self.C2:
                Y_train_pred = self.predict_fn(self.X_train)
                Y_test_pred = self.predict_fn(self.X_test)
                Y_train_pred = np.repeat(Y_train_pred, self.C1, axis=-1)
                Y_test_pred = np.repeat(Y_test_pred, self.C1, axis=-1)
            elif self.C1 < self.C2:
                Y_train_pred = self.predict_fn(self.X_train[..., 0])
                Y_test_pred = self.predict_fn(self.X_test[..., 0])
            else:
                Y_train_pred = self.predict_fn(self.X_train)
                Y_test_pred = self.predict_fn(self.X_test)

            spacer = np.zeros_like(self.Y_train)
            spacer[0] = print_text(f'{epoch:03d}', self.W, self.H)[..., np.newaxis]

            if self.threshold is None:
                columns = [self.X_train, Y_train_pred, self.Y_train, spacer,
                           self.X_test, Y_test_pred, self.Y_test]
            else:
                Y_train_pred_th = np.round(Y_train_pred+0.5-self.threshold)
                Y_test_pred_th = np.round(Y_test_pred+0.5-self.threshold)
                columns = [self.X_train, Y_train_pred, Y_train_pred_th, self.Y_train, spacer,
                           self.X_test, Y_test_pred, Y_test_pred_th, self.Y_test]
            mosaic = np.hstack([c.reshape(-1, self.W, self.C) for c in columns])
            if self.inverse:
                mosaic = 1.-mosaic
            filename = "epoch{:03d}.png".format(epoch)
            filename = os.path.join(self.logdir, filename)
            if self.C == 1:
                mosaic = mosaic[..., 0]
            imsave(filename, mosaic)


class GenerateAnimation(Callback):

    def __init__(self, folder, pattern='epoch%03d.png', extension='mp4', fps=4, outfile='progress'):
        self.folder = folder
        self.pattern = os.path.join(self.folder, pattern)
        self.extension = extension
        self.fps = fps
        self.outfile = os.path.join(self.folder, "{}.{}".format(outfile, extension))
        self.command = ['ffmpeg', '-y', '-r', str(self.fps), '-i', self.pattern, '-vcodec', 'libx264', '-crf', '25', self.outfile]

    def on_epoch_end(self, epoch, logs={}):
        subprocess.Popen(self.command, stdout=devnull, stderr=devnull)
