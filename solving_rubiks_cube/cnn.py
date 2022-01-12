import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.stax import Conv, Relu, Flatten, Dense, Dropout, serial
from jax.nn.initializers import glorot_normal, normal
from jax.lax import conv_general_dilated
from jax.nn import relu


def conv_net(mode="train"):
    out_dim = 1
    dim_nums = ("NHWC", "HWIO", "NHWC")
    unit_stride = (1,1)
    zero_pad = ((0,0), (0,0))

    # Primary convolutional layer.
    conv_channels = 32
    conv_init, conv_apply = Conv(out_chan=conv_channels, filter_shape=(3,3),
                                 strides=(1,3), padding=zero_pad)
    # Group all possible pairs.
    pair_channels, filter_shape = 256, (1, 2)

    # Convolutional block with the same number of channels.
    block_channels = pair_channels
    conv_block_init, conv_block_apply = serial(Conv(block_channels, (1,3), unit_stride, "SAME"), Relu,  # One block of convolutions.
                                               Conv(block_channels, (1,3), unit_stride, "SAME"), Relu,          
                                               Conv(block_channels, (1,3), unit_stride, "SAME"))
    # Forward pass.
    hidden_size = 2048
    dropout_rate = 0.25
    serial_init, serial_apply = serial(Conv(block_channels, (1,3), (1, 3), zero_pad), Relu,     # Using convolution with strides
                                       Flatten, Dense(hidden_size),                             # instead of pooling for downsampling.
                                    #    Dropout(dropout_rate, mode),
                                       Relu, Dense(out_dim))

    def init_fun(rng, input_shape):
        rng, conv_rng, block_rng, serial_rng = jax.random.split(rng, num=4)

        # Primary convolutional layer.
        conv_shape, conv_params = conv_init(conv_rng, (-1,) + input_shape)

        # Grouping all possible pairs.
        kernel_shape = [filter_shape[0], filter_shape[1], conv_channels, pair_channels]
        bias_shape = [1, 1, 1, pair_channels]
        W_init = glorot_normal(in_axis=2, out_axis=3)
        b_init = normal(1e-6)
        k1, k2 = jax.random.split(rng)
        W, b = W_init(k1, kernel_shape), b_init(k2, bias_shape)
        pair_shape = conv_shape[:2] + (15,) + (pair_channels,)
        pair_params = (W, b)

        # Convolutional block.
        conv_block_shape, conv_block_params = conv_block_init(block_rng, pair_shape)

        # Forward pass.
        serial_shape, serial_params = serial_init(serial_rng, conv_block_shape)
        params = [conv_params, pair_params, conv_block_params, serial_params]
        return serial_shape, params

    def apply_fun(params, inputs):
        conv_params, pair_params, conv_block_params, serial_params = params

        # Apply the primary convolutional layer.
        conv_out = conv_apply(conv_params, inputs)
        conv_out = relu(conv_out)

        # Group all possible pairs.
        W, b = pair_params
        pair_1 = conv_general_dilated(conv_out, W, unit_stride, zero_pad, (1,1), (1,1), dim_nums) + b
        pair_2 = conv_general_dilated(conv_out, W, unit_stride, zero_pad, (1,1), (1,2), dim_nums) + b
        pair_3 = conv_general_dilated(conv_out, W, unit_stride, zero_pad, (1,1), (1,3), dim_nums) + b
        pair_4 = conv_general_dilated(conv_out, W, unit_stride, zero_pad, (1,1), (1,4), dim_nums) + b
        pair_5 = conv_general_dilated(conv_out, W, unit_stride, zero_pad, (1,1), (1,5), dim_nums) + b
        pair_out = jnp.dstack([pair_1, pair_2, pair_3, pair_4, pair_5])
        pair_out = relu(pair_out)

        # Convolutional block.
        conv_block_out = conv_block_apply(conv_block_params, pair_out)

        # Residual connection.
        res_out = conv_block_out + pair_out
        res_out = relu(res_out)

        # Forward pass.
        out = serial_apply(serial_params, res_out)
        return out

    return init_fun, apply_fun

#