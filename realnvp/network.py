import torch.nn as nn


class WeightNormConv2d(nn.Module):
    """Same as Conv2d but with weight normalization."""

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        """Init a weight normalized Conv2d layer.

        Args:
            in_channels: int
                Number of channels in the input tensor.
            out_channels: int
                Number of channels in the output tensor.
            kernel_size: int, tuple(int, int)
                The size of the convolving kernel.
            kwargs: dict, optional
                Dictionary with additional configuration parameters used for
                initializing a standard `nn.Conv2d` layer.
        """
        super(WeightNormConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(
            in_channels, out_channels, kernel_size, **kwargs,
        ))

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """ResBlock is a residual block of convolutional layers.

    The input to the block is a tensor of shape `(C, H, W)`, and the output is a
    tensor of the same shape.
    The block consists of three layers:
        * The first layer is a `1 x 1` convolutional layer and it reduces the
            number of channels to `C // 2`.
        * The second layer is a `k x k` convolutional layer that preserves the
            number of channels.
        * The third layer is a `1 x 1` convolutional layer that up-samples back
            the number of channels to `C`.
    """

    def __init__(self, filters, kernel_size, **kwargs):
        """Init a residual block of convolutional layers.

        Args:
            filters: int
                Number of filters in the input image.
            kernel_size: int or tuple(int, int)
                The size of the convolving kernel for the middle layer.
            kwargs: dict, optional
                Dictionary with additional configuration parameters used for
                initializing a standard `nn.Conv2d` layer. Used for the mid layer.
        """
        super().__init__()
        kwargs["padding"] = "same" # ignore user specified padding.

        self.block = nn.Sequential(
            WeightNormConv2d(filters, filters // 2, kernel_size=1, padding=0),
            nn.ReLU(),
            WeightNormConv2d(filters // 2, filters // 2, kernel_size, **kwargs),
            nn.ReLU(),
            WeightNormConv2d(filters // 2, filters, kernel_size=1, padding=0),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNet(nn.Module):
    """Simple ResNet created by stacking together multiple residual blocks."""

    def __init__(self, in_channels, out_channels, n_blocks=8, filters=128, kernel_size=3):
        """Init a Residual network.

        Args:
            in_channels: int
                Number of channels in the input image.
            out_channels: int
                Number of channels in the output tensor.
            n_blocks: int, optional
                Number of residual blocks. Default: 8.
            filters: int, optional
                Number of filters for the convolutional layers. Default: 128.
            kernel_size: int or tuple(int, int), optional
                The size of the convolving kernel for the middle layer. Default: 3.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.filters = filters
        self.kernel_size = kernel_size

        layers = []
        layers.extend([
            WeightNormConv2d(in_channels, filters, kernel_size, padding="same"),
            nn.ReLU(),
        ])
        for _ in range(n_blocks):
            layers.append(ResBlock(filters, kernel_size))
        layers.extend([
            nn.ReLU(),
            WeightNormConv2d(filters, out_channels, kernel_size, padding="same")
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

#