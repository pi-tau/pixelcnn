import torch.nn as nn

from conv2d_mask import MaskConv2d


class ResBlock(nn.Module):
    """ResBlock is a residual block of convolutional layers used in the PixelCNN
    model.

    The input to the block is a tensor of shape (C, H, W), and the output is a
    tensor of the same shape.
    The block consists of three layers:
        * The first layer is a `1 x 1` convolutional layer and it reduces the
            number of channels to `C // 2`.
        * The second layer is a `k x k` convolutional layer that preserves the
            number of channels and uses mask B.
        * The third layer is a `1 x 1` convolutional layer that up-samples back
            the number of channels to `C`.

    Note that the `1 x 1` convolutional layers also need to be masked layer in
    order to impose the auto-regressive property along the channel dimension.
    ```
        |-------------|
        |    ReLU + Conv 1 x 1 mask B
        |             |
        |    ReLU + Conv k x k mask B
        |             |
             ReLU + Conv 1 x 1 mask B
        X  -----------|
    ```
    """

    def __init__(self, in_channels, kernel_size, **kwargs):
        """Init a residual block of convolutional layers.

        Args:
            in_channels (int): Number of channels in the input image.
            kernel_size (int): The size of a square convolving kernel.
            kwargs (dict): A dictionary with additional configuration parameters
                used for initializing a standard `nn.Conv2d` layer.
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.ReLU(),
            MaskConv2d("B", in_channels, in_channels // 2, 1),
            nn.ReLU(),
            MaskConv2d("B", in_channels // 2, in_channels // 2, kernel_size, **kwargs),
            nn.ReLU(),
            MaskConv2d("B", in_channels // 2, in_channels, 1),
        )

    def forward(self, x):
        return x + self.block(x)

#