from fnmatch import fnmatch
import pathlib
import shutil

import numpy as np

from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
    ReduceLROnPlateau,
    EarlyStopping,
    TerminateOnNaN,
    CSVLogger,
)

from keras.utils import plot_model

from ..callbacks import TrainTestMosaic, GenerateAnimation
from ..generators.lookahead import LookAhead
from .. import networks
from .. import losses
from .. import metrics as _metrics
from ..util.cli import color


def retrieve_output(model, search):
    matches = [layer for layer in model.layers if fnmatch(layer.name, f'*{search}')]
    return matches[0].output


class KerasModel:

    def __init__(self, model, **model_kwargs):
        self.compiled = False
        self.loss = {}

        self.params = {}
        if isinstance(model, str):
            self.params['model'] = model
            model = getattr(networks, model)(**model_kwargs)
            self.params.update(model.params)
        self.model = model

        self.hooks = {
            "input": model.input,
            "output": model.output,
        }

        self.outputs = ['output']

    def load_weights(self, init_weights):
        self.model.load_weights(init_weights, by_name=True)

    def loss_zipper(self, gen):
        for X, Y in gen:
            if len(self.outputs) == 1:
                Y = [Y]
            dictY = {output: y for output, y in zip(self.outputs, Y)}
            batch_size = X.shape[0]
            newY = [dictY[hook] if hook in dictY else np.zeros([batch_size]+self.hooks[hook].shape.as_list()[1:]) for hook in self.flat_hooks]

            yield X, newY

    def compile(self,
                epochs=1000,
                optim='Adam',
                lr=1e-4,
                loss='binary_crossentropy',
                lookahead=False,
                init_weights=None,
                metrics=None,
                reducelr=True,
                patience=30,
                min_lr=1e-6,
                earlystop=True,
                terminantenan=True,
                csv_logger=True,
                checkpoint=True,
                save_best_only=True,
                tensorboard=True,
                mosaic=False,
                ):

        params = dict(locals())
        del params['self']
        self.train_params = params

        self.epochs = epochs

        # Optimizer
        if optim == 'SGD':
            optim = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        elif optim == 'Adam':
            optim = Adam(lr=lr)

        # Losses
        if isinstance(loss, str):
            loss = {"output": [(loss, 1)]}

        self.loss.update(loss)

        flat_losses = [loss for hook in self.loss for loss, _ in self.loss[hook]]
        flat_losses = [getattr(losses, loss) if isinstance(loss, str) else loss for loss in flat_losses]
        flat_weights = [weight for hook in self.loss for _, weight in self.loss[hook]]
        flat_hooks = [hook for hook in self.loss for _ in self.loss[hook]]
        self.flat_hooks = flat_hooks
        flat_outputs = [self.hooks[hook] for hook in flat_hooks]

        self.first_output_index = flat_hooks.index('output')
        if len(flat_outputs) == 1:
            self.first_output_index = None

        # Model
        self.model = Model(self.hooks['input'], flat_outputs)

        if init_weights is not None:
            self.load_weights(init_weights)

        # Metrics
        if metrics is not None:
            metrics = [getattr(_metrics, m) if hasattr(_metrics, m) else m for m in metrics]

        self.model.compile(loss=flat_losses, loss_weights=flat_weights, optimizer=optim, metrics=metrics)
        self.compiled = True

        return params

    def fit(self, data, path=None):
        if not self.compiled:
            raise ValueError("Need to compile model first")

        if isinstance(path, str):
            path = pathlib.Path(path)

        model = self.model

        callbacks = []

        if self.train_params['reducelr']:
            patience = self.train_params['patience']
            min_lr = self.train_params['min_lr']
            reducelr = ReduceLROnPlateau(verbose=1, patience=patience//2, min_lr=min_lr)
            callbacks.append(reducelr)

        if self.train_params['earlystop']:
            patience = self.train_params['patience']
            earlystop = EarlyStopping(patience=patience)
            callbacks.append(earlystop)

        if self.train_params['terminantenan']:
            terminantenan = TerminateOnNaN()
            callbacks.append(terminantenan)

        if path is not None:

            # Save model properties for reproducibility
            with open(path / 'summary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))

            with open(path / 'model.json', 'w') as f:
                print(str(model.to_json()), file=f)

            if shutil.which("dot") is not None:
                plot_model(model, to_file=(path/'model.png').as_posix())

            # Output callbacks

            if self.train_params['csv_logger']:
                training_path = path / "training.csv"
                csv_logger = CSVLogger(training_path)
                callbacks.append(csv_logger)

            if self.train_params['checkpoint']:
                weight_path = path / "weights-best.hdf5"
                save_best_only = self.train_params['save_best_only']
                checkpoint = ModelCheckpoint(weight_path.as_posix(),
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=save_best_only,
                                             mode='min')
                callbacks.append(checkpoint)

            if self.train_params['tensorboard']:
                tensorboard = TensorBoard(log_dir=(path / 'tblogs').as_posix())
                callbacks.append(tensorboard)

            if self.train_params['mosaic']:
                imgpath = (path / 'images').as_posix()
                X_train, Y_train = next(data.train_generator)
                X_val, Y_val = next(data.val_generator)

                mosaic = TrainTestMosaic(imgpath, X_train, Y_train, X_val, Y_val, frequency=1, inverse=True, index=self.first_output_index)
                animation = GenerateAnimation(imgpath)
                callbacks.extend([mosaic, animation])

        train = self.loss_zipper(data.train_generator)
        val = self.loss_zipper(data.val_generator)

        if self.train_params['lookahead']:
            train = LookAhead(train)
            val = LookAhead(val)

        train_steps_per_epoch = data.steps_per_epoch['train']
        val_steps_per_epoch = data.steps_per_epoch['val']

        n_train = data.batch_size * train_steps_per_epoch
        n_val = data.batch_size * val_steps_per_epoch

        print("")
        print(f"{color.BOLD}Train: {n_train}\tSteps: {train_steps_per_epoch}\tBatch:{data.batch_size}{color.END}")
        print(f"{color.BOLD}Val:   {n_val}\tSteps: {val_steps_per_epoch}\tBatch:{data.batch_size}{color.END}")
        print("")

        model.fit_generator(train,
                            train_steps_per_epoch,
                            epochs=self.epochs,
                            callbacks=callbacks,
                            validation_data=val,
                            validation_steps=val_steps_per_epoch)

        if path is not None:

            model.save(path / 'final_model.hdf5')

        return model
