import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskConv2d(nn.Conv2d):
    """MaskConv2d is the same as the conv2d layer but applies a mask on the
    kernel weights. We will assume an ordering row-by-row from top to bottom and
    we will mask all the weights of the kernel that come after the mid point.

    Given a square kernel of size `k x k`, this layer will apply the following
    mask to the kernel (example `k=5`):
    ```
        1  1  1  1  1
        1  1  1  1  1
        1  1 *?* 0  0
        0  0  0  0  0
        0  0  0  0  0
    ```

    Depending on whether we want to mask the mid point or not we have two types
    of masked conv2d layers:
        * type A will mask the mid point
        * type B will not mask the mid point

    Using a masked kernel ensures that the activation value for a given pixel
    depends only on the values of the previous pixels, without looking ahead.
    Note that for each pixel we have multiple channels. We would also like to
    make sure that the activation at a particular channel of a given pixel
    depends only on previous channels for the same pixel, without looking at the
    future channels (assuming a natural ordering of the channels).

    Input images for the entire model **usually** have 3 color channels (R,G,B).
    Thus, we will spilt the feature maps of the convolving kernel into three
    equal groups, each group corresponding to one of the color channels.
    """

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, **kwargs):
        """Init a masked Conv 2D layer.

        Args:
            mask_type (str): The type of the mask (type "A" or type "B").
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple(int, int)): The size of the convolving kernel.
            kwargs (dict): A dictionary with additional configuration parameters
                used for initializing a standard `nn.Conv2d` layer. If padding
                is provided it will be ignored.
        """
        kwargs["padding"] = "same" # ignore user specified padding
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        assert mask_type == "A" or mask_type == "B", "unknown mask type"
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kv, kh = kernel_size

        # Mask the contribution of future pixels. Allow only previous pixels
        # to be used as context.
        mask = torch.ones_like(self.weight) # (out_channels, in_channels, kv, kh)
        mask[:, :, kv // 2, kh // 2 + 1:] = 0 # mask the weights at mid row after mid point
        mask[:, :, kv // 2 + 1:, :] = 0       # mask the weights below mid row

        # Mask the context of future channels at the current pixel (p. [kv//2, kh//2]).
        # We will split the input channels and the output channels into three
        # separate groups of equal size, each group corresponding to one of the
        # colors (R,G,B). The activation value for a particular channel for the
        # middle pixel should only depend on input channels that correspond to
        # previous colors of the same pixel.
        assert in_channels % 3 == 0 and out_channels % 3 == 0, "number of filters must be divisible by 3"
        one_third_in, one_third_out = in_channels // 3, out_channels // 3
        if mask_type == "A":
            # Mask the channels for colors R, G, B when computing R
            mask[:one_third_out, :, kv // 2, kh // 2] = 0
            # Mask the channels for colors G, B when computing G
            mask[one_third_out:2*one_third_out, one_third_in:, kv // 2, kh // 2] = 0
            # Mask the channels for color B, when computing B
            mask[2*one_third_out:, 2*one_third_in:, kv // 2, kh // 2] = 0
        else: # mask_type == "B"
            # Mask the channels for colors G, B when computing R
            mask[:one_third_out, one_third_in:, kv // 2, kh // 2] = 0
            # Mask the channels for colors B when computing G
            mask[one_third_out:2*one_third_out, 2*one_third_in:, kv // 2, kh // 2] = 0
            # No masking is imposed when computing B

        # Instead of `self.mask = mask` we will use `register_buffer`.
        # This provides the benefit of pushing both the buffer and the model
        # parameters to the same device when calling `model.to(device)`.
        self.register_buffer("mask", mask.type(dtype=torch.bool))

    def forward(self, x):
        return F.conv2d(
            x, self.mask * self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )

#