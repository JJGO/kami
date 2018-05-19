import keras.layers as KL
from keras.models import Model
from fnmatch import fnmatch


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
                 enc_filters,
                 num_blocks=None,
                 filter_mult=2,
                 convs_per_block=2,
                 batch_norm=True,
                 residual=False,
                 activation='relu',
                 kernel_size=3,
                 pool_size=2,
                 padding='same',
                 prefix=None,
                 ):

    if prefix is None:
        prefix = "conv_enc"

    ndims, pool_size, kernel_size, enc_filters = _clean_inputs(input_shape, pool_size,
                                                               kernel_size, enc_filters,
                                                               num_blocks, filter_mult)

    # Describe model
    maxpool_layer = getattr(KL, f'MaxPooling{ndims}D')

    tensor = input = KL.Input(shape=input_shape, name=f'{prefix}_input')

    for i, filters in enumerate(enc_filters, start=1):
        tensor = conv_block(tensor,
                            filters,
                            block_name=f'{prefix}_block{i}',
                            convs_per_block=convs_per_block,
                            activation=activation,
                            kernel_size=kernel_size,
                            padding=padding,
                            batch_norm=batch_norm,
                            residual=residual)

        # Pool except in last layer
        if i < len(enc_filters):
            tensor = maxpool_layer(pool_size=pool_size, name=f'{prefix}_pool{i}')(tensor)

    output = tensor
    model = Model(input, output, prefix)
    return model


def conv_decoder(input_shape,
                 dec_filters,
                 num_blocks=None,
                 filter_mult=2,
                 convs_per_block=2,
                 batch_norm=True,
                 residual=False,
                 activation='relu',
                 kernel_size=3,
                 up_size=2,
                 skip_connections=False,
                 input_model=None,
                 padding='same',
                 prefix=None,
                 ):

    if prefix is None:
        prefix = "conv_dec"

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

    ndims, up_size, kernel_size, dec_filters = _clean_inputs(input_shape, up_size,
                                                             kernel_size, dec_filters,
                                                             num_blocks, filter_mult)
    if len(up_size) == 1:
        up_size = up_size[0]
    # Describe model
    upsample_layer = getattr(KL, f'UpSampling{ndims}D')

    for i, filters in enumerate(reversed(dec_filters), start=1):
        tensor = upsample_layer(size=up_size, name=f'{prefix}_up{i}')(tensor)
        tensor = conv_block(tensor,
                            filters,
                            block_name=f'{prefix}_block{i}',
                            convs_per_block=convs_per_block,
                            activation=activation,
                            kernel_size=kernel_size,
                            padding=padding,
                            batch_norm=batch_norm,
                            residual=residual)
        if skip_connections:
            # i-1 Due to 1 indexing
            tensor = KL.concatenate([tensor, shortcuts[i-1]], -1, name=f'{prefix}_merge{i}')

    output = tensor
    model = Model(input, output, prefix)
    return model
