import numpy as np
import keras.layers as KL
from keras.models import Model
from fnmatch import fnmatch
from .vae import gauss_sample


def conv_layer_like(tensor):
    ndims = len(tensor.shape.as_list()[1:-1])  # all but batch and shape
    conv_layer = getattr(KL, f'Conv{ndims}D')
    return conv_layer, ndims


def conv_block(input,
               filters,
               convs_per_block=1,
               block_name="",
               activation="relu",
               kernel_size=3,
               padding='same',
               batch_norm=True,
               residual=False):

    conv_layer, ndims = conv_layer_like(input)

    if isinstance(filters, int):
        filters = (filters,) * convs_per_block

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * ndims

    tensor = input
    for i, f in enumerate(filters, start=1):
        tensor = conv_layer(f, kernel_size, padding=padding,
                            name=f"{block_name}_conv{ndims}D{i}")(tensor)
        if batch_norm:
            tensor = KL.BatchNormalization(name=f"{block_name}_bn{i}")(tensor)
        tensor = KL.Activation(activation, name=f"{block_name}_act{i}")(tensor)

    output = tensor

    if residual:
        input_channels = input.shape.as_list()[-1]
        output_channels = output.shape.as_list()[-1]
        shortcut = input
        if input_channels != output_channels:
            shortcut = conv_layer(output_channels, (1,)*ndims, padding=padding,
                                  name=f"{block_name}_conv{ndims}D_resconv")(input)
        output = KL.add([shortcut, output], name=f"{block_name}_conv{ndims}D_resmerge")
    return output


def _clean_inputs(input_shape, pool_size, kernel_size, filters, num_blocks, filter_mult):
    # Make everything independent of the num of dimensions
    ndims = len(input_shape) - 1  # Substract the channel dimension
    input_shape = tuple(input_shape)

    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * ndims

    if isinstance(filters, int):
        if num_blocks is not None:
            filters = tuple([filters*filter_mult**i for i in range(num_blocks)])
        else:
            raise ValueError("For int enc_filters a num_blocks must be provided")
    return ndims, pool_size, kernel_size, filters


def conv_encoder(input_shape,
                 filters,
                 num_blocks=None,
                 filter_mult=2,
                 convs_per_block=2,
                 batch_norm=True,
                 residual=False,
                 activation='relu',
                 kernel_size=3,
                 pool_size=2,
                 pool_freq=1,
                 padding='same',
                 prefix="conv_enc",
                 ):
    params = dict(locals())
    ndims, pool_size, kernel_size, filters = _clean_inputs(input_shape, pool_size,
                                                           kernel_size, filters,
                                                           num_blocks, filter_mult)

    # Describe model
    maxpool_layer = getattr(KL, f'MaxPooling{ndims}D')

    tensor = input = KL.Input(shape=input_shape, name=f'{prefix}_input')

    for i, f in enumerate(filters, start=1):
        tensor = conv_block(tensor,
                            f,
                            block_name=f'{prefix}_block{i}',
                            convs_per_block=convs_per_block,
                            activation=activation,
                            kernel_size=kernel_size,
                            padding=padding,
                            batch_norm=batch_norm,
                            residual=residual)

        # Pool except in last layer
        if i < len(filters) and i % pool_freq == 0:
            tensor = maxpool_layer(pool_size=pool_size, name=f'{prefix}_pool{i}')(tensor)

    output = tensor
    model = Model(input, output, prefix)
    model.params = params
    return model


def conv_classifier(input_shape,
                    n_outputs,
                    filters,
                    output_activation=None,
                    global_avgpool=False,
                    denses=None,
                    num_blocks=None,
                    filter_mult=2,
                    convs_per_block=2,
                    batch_norm=True,
                    residual=False,
                    activation='relu',
                    kernel_size=3,
                    pool_size=2,
                    pool_freq=1,
                    padding='same',
                    prefix="conv_clf",
                    ):
    params = dict(locals())
    if output_activation is None:
        if n_outputs == 1:
            output_activation = 'sigmoid'
        else:
            output_activation = 'softmax'

    encoder = conv_encoder(input_shape=input_shape,
                           filters=filters,
                           num_blocks=num_blocks,
                           filter_mult=filter_mult,
                           convs_per_block=convs_per_block,
                           batch_norm=batch_norm,
                           residual=residual,
                           activation=activation,
                           kernel_size=kernel_size,
                           pool_size=pool_size,
                           pool_freq=pool_freq,
                           prefix=f'{prefix}_enc')
    input = encoder.input
    middle = encoder.output
    if global_avgpool:
        ndims = len(input_shape) - 1    # We do not average across channels
        avgpool_layer = getattr(KL, f"AveragePooling{ndims}D")
        pool_size = middle.shape.as_list()[1:-1]
        strides = (1,)*ndims
        middle = avgpool_layer(pool_size=pool_size, strides=strides,
                               name=f'{prefix}_global_avgpool')(middle)

    middle = KL.Flatten(name=f'{prefix}_output_flatten')(middle)
    if denses is not None:
        for i, n_units in enumerate(denses, start=1):
            middle = KL.Dense(n_units, name=f'{prefix}_output_dense{i}')
            if batch_norm:
                middle = KL.BatchNormalization(name=f'{prefix}_output_bn{i}')(middle)
            middle = KL.Activation(activation, name=f'{prefix}_output_act{i}')(middle)

    output = KL.Dense(n_outputs, name=f'{prefix}_output_dense')(middle)
    output = KL.Activation(output_activation, name=f'{prefix}_output_act')(output)

    model = Model(input, output, prefix)
    model.params = params
    return model


def conv_decoder(input_shape,
                 filters,
                 num_blocks=None,
                 filter_mult=2,
                 convs_per_block=2,
                 batch_norm=True,
                 residual=False,
                 activation='relu',
                 kernel_size=3,
                 pool_size=2,
                 padding='same',
                 skip_connections=False,
                 input_model=None,
                 output_activation=None,
                 output_channels=None,
                 prefix="conv_dec",
                 ):
    params = dict(locals())

    if skip_connections:
        input = input_model.input
        tensor = input_model.output
        if input_model is None:
            raise ValueError("With skip_connections an input model must be provided")
        else:
            pools = [e for e in input_model.layers if fnmatch(e.name, '*_pool*')]
            pools = sorted(pools, reverse=True, key=lambda x: x.name)
            shortcuts = [p.input for p in pools]
    else:
        tensor = input = KL.Input(shape=input_shape, name=f'{prefix}_input')

    ndims, pool_size, kernel_size, filters = _clean_inputs(input_shape, pool_size,
                                                           kernel_size, filters,
                                                           num_blocks, filter_mult)
    if len(pool_size) == 1:
        pool_size = pool_size[0]
    # Describe model
    upsample_layer = getattr(KL, f'UpSampling{ndims}D')

    for i, f in enumerate(reversed(filters), start=1):
        tensor = upsample_layer(size=pool_size, name=f'{prefix}_up{i}')(tensor)
        if skip_connections:
            # i-1 Due to 1 indexing
            tensor = KL.concatenate([tensor, shortcuts[i-1]], -1, name=f'{prefix}_merge{i}')
        tensor = conv_block(tensor,
                            f,
                            block_name=f'{prefix}_block{i}',
                            convs_per_block=convs_per_block,
                            activation=activation,
                            kernel_size=kernel_size,
                            padding=padding,
                            batch_norm=batch_norm,
                            residual=residual)

    output = tensor

    if output_activation is not None:
        output = conv_block(output,
                            output_channels,
                            block_name=f'{prefix}_output',
                            convs_per_block=1,
                            activation=output_activation,
                            kernel_size=1,
                            padding=padding,
                            batch_norm=batch_norm,
                            residual=False)

    model = Model(input, output, prefix)
    model.params = params
    return model


def conv_autoencoder(input_shape,
                     enc_kwargs=None,
                     dec_kwargs=None,
                     latent_dim=None,
                     skip_connections=False,
                     sampling=False,
                     cond_input=None,
                     enc_cond=False,
                     return_parts=False,
                     prefix="conv_auto",
                     **kwargs,
                     ):

    params = dict(locals())

    if latent_dim is None and sampling:
        raise ValueError("Fully convolutional VAE not supported")

    if latent_dim is None and sampling:
        raise ValueError("Fully convolutional VAE not supported")

    # Preprocess inputs
    if enc_kwargs is None:
        enc_kwargs = {}

    if dec_kwargs is None:
        dec_kwargs = enc_kwargs

    enc_kwargs.update(kwargs)
    dec_kwargs.update(kwargs)

    # Define encoder model
    encoder = conv_encoder(input_shape, prefix=f'{prefix}_enc', **enc_kwargs)
    input = encoder.input
    middle = encoder.output
    # input = KL.Input(shape=input_shape,  name=f'{prefix}_input')
    # middle = encoder(input)
    last_shape = middle.shape.as_list()[1:]

    # If latent dimension flatten, dense and then reshape
    if latent_dim is not None:

        z = KL.Flatten(name=f'{prefix}_enc_flatten')(middle)

        # If sampling, do gauss diag sample
        if sampling:
            z_mean = KL.Dense(latent_dim, name=f'{prefix}_enc_mean')(z)
            z_logvar = KL.Dense(latent_dim, name=f'{prefix}_enc_logvar')(z)
            z = KL.Lambda(gauss_sample, output_shape=(latent_dim,), name=f'{prefix}_sampling')([z_mean, z_logvar])
            encoder = Model(input, [z_mean, z_logvar], f'{prefix}_enc')
        else:
            z = KL.Dense(latent_dim, name=f'{prefix}_enc_latent')(z)
            encoder = Model(input, z, f'{prefix}_enc')
    else:
        input_dec = middle

    # If we are conditioning on some input, concatenate it
    if cond_input is not None:
        # cond_input can be node or shape
        if isinstance(cond_input, (tuple, list)):
            cond_input = KL.Input(shape=cond_input, name=f'{prefix}_cond_input')

        z_cond = cond_input
        # We can apply the same encoder to the input
        if enc_cond:
            z_cond = encoder(z_cond)
        z = KL.concatenate([z, z_cond], -1, name=f'{prefix}_cond_merge')
        cond_channels = z_cond.shape.as_list()[-1]
        last_shape = last_shape[:-1] + [last_shape[-1] + cond_channels]
        input = [input, cond_input]

    if latent_dim is not None:
        input_dec = KL.Dense(np.prod(last_shape), name=f'{prefix}_dec_resize')(z)

    input_dec = KL.Reshape(last_shape, name=f'{prefix}_dec_reshape')(input_dec)
    decoder = conv_decoder(last_shape, prefix=f'{prefix}_dec',
                           skip_connections=skip_connections,
                           input_model=Model(input, input_dec),
                           **dec_kwargs)

    if not skip_connections:
        output = decoder(input_dec)
    else:
        output = decoder.output

    autoenc = Model(input, output, prefix)
    autoenc.params = params

    if return_parts:
        return autoenc, encoder, decoder
    else:
        return autoenc
