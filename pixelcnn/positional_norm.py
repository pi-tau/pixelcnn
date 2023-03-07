import torch.nn as nn

class PositionalNorm(nn.LayerNorm):
    """PositionalNorm is a normalization layer used for 3D inputs that
    normalizes exclusively across the channels dimension. This layer supports
    two input data formats: `channels_last` (default) or `channels_first`. These
    specify the ordering of the dimensions in the inputs. Using `channels_last`
    corresponds to inputs with shape (B, H, W, C) while `channels_first`
    corresponds to inputs with shape (B, C, H, W).

    Layer normalization was introduced in the paper "Layer Normalization" by
    Jimmy Lei Ba et al., 2016. It was proposed in order to solve some of the
    problems that batch normalization was facing. Namely, training with small
    batch sizes and using recurrent neural network models. However, the authors
    of the paper explicitly state that layer normalization does not work well
    in convolutional networks:
        `With fully connected layers, all the hidden units in a layer tend to
         make similar contributions to the final prediction and re-centering and
         rescaling the summed inputs to a layer works well. However, the
         assumption of similar contributions is no longer true for convolutional
         neural networks.`

    Usually, what is done in practice is to normalize only along the channels
    dimension. This operation is discussed in the paper "Positional normalization"
    by Boyi Li et al., 2019.

    The standard implementation of LayerNorm allows for normalizing along a
    specified number of dimensions, but they have to come last. Thus, we can
    easily do normalization over the channel dimension if the input is of shape
    (B, H, W, C). However, usually this is not the case, as convolutional layers
    expect the input to be of shape (B, C, H, W).

    This layer will be used as a building block for the PixelCNN model. In order
    to respect the auto-regressive property of the channel values we need to
    perform the normalization across three separate groups (corresponding to R,G,B),
    instead of across the entire channels dimension.
    """
    def __init__(self, data_format="channels_last", *args, **kwargs):
        """Init a positional normalization layer.

        Args:
            data_format: str
                Specifies the format of the input data. Must be one of
                ["channels_last", "channels_first"].
            *args, **kwargs: Additional configuration parameters used for
                initializing a standard `nn.LayerNorm` layer.
        """
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError

        super().__init__(*args, **kwargs)
        self.data_format = data_format

    def forward(self, x):
        """Perform a forward pass through the positional normalization layer.

        Args:
            x: torch.Tensor
                Input tensor of shape (B, C, H, W) or (B, H, W, C).
        """
        if self.data_format == "channels_last":
            # The input is of shape (B, H, W, C). Separate the channels into
            # three groups and perform the standard layer norm across each group.
            x_shape = x.shape
            x = x.reshape(*x.shape[:-1]+(3, -1))
            return super().forward(x).view(*x_shape)

        # The input is of shape (B, C, H, W). Transpose the input so that the
        # channels are pushed to the last dimension, separate them into groups,
        # and then run the standard LayerNorm layer.
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        x = x.reshape(*x.shape[:-1]+(3, -1))
        out = super().forward(x).view(*x_shape)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out

#