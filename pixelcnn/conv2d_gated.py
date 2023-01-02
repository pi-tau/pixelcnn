import torch
import torch.nn as nn

from conv2d_mask import MaskConv2d


class GatedConv2d(nn.Module):
    """GatedConv2d implements the gated convolutional layer as described in
    "Conditional image generation with PixelCNN decoders" by van en Oord et al., 2016.

    The layer combines two convolutional filters: a horizontal filter and a
    vertical filter. The vertical filter is a 2D filter masked so that only
    pixels above the current pixel are used in the calculation:

    `vertical convolving filter`
    ```
        1  1  1  1  1
        1  1  1  1  1
        0  0  0  0  0
        0  0  0  0  0
        0  0  0  0  0
    ```

    The horizontal filter is a 1D filter that slides along the rows and is
    masked so that only pixels coming before the current pixel are considered:

    `horizontal convolving filter`
    ```
        1  1  *?* 0  0
    ```

    Depending on whether we want to mask the mid point of the horizontal filter
    or not we have two types of gated conv2d layers:
        * type A will mask the mid point
        * type B will not mask the mid point

    The output of the two filters is then combined and a gated unit is applied
    as a non-linearity function.
    """

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, **kwargs):
        """Init a gated Conv 2D layer.

        Args:
            mask_type (str): The type of the mask (type "A" or type "B").
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple(int, int)): The size of the convolving kernel.
            kwargs (dict): A dictionary with additional configuration parameters
                used for initializing a standard `nn.Conv2d` layer. If padding
                is provided it will be ignored.
        """
        super().__init__()
        assert mask_type == "A" or mask_type == "B", "unknown mask type"
        k = kernel_size

        # Initialize the convolutional layer for the vertical convolution.
        # Instead of using a `k x k` masked filter, we will use a `[k//2] x k`
        # non-masked rectangular filter and we will apply padding along the top
        # and the left edges of the image. Note that this operation will produce
        # an output of shape (H + k//2, W + k//2), so we need to crop the output
        # afterwards. No masking is needed as we are convolving the rows above.
        #
        # We need two separate convolutions in order to pass the results through
        # the gated unit at the end. In order to increase parallelization, these
        # two convolutions are combined in a single forward pass where we double
        # the number of filters. After forwarding through this layer, the output
        # is split into two chunks and each is passed to the gate unit.
        k_up = (k + 1) // 2 # round-up integer division
        self.v_conv = nn.Conv2d(
            in_channels, 2*out_channels, kernel_size=(k_up, k), padding=k_up, **kwargs)

        # Initialize the convolutional layer for the horizontal convolution.
        # Instead of using a `1 x k` masked filter, we could use a `1 x [k//2]`
        # non-masked filter combined with padding and cropping. However, we need
        # to mask the channels of the current pixel appropriately.
        #
        # Similarly to the vertical convolution, we need two separate convolving
        # kernels which we will model by simply doubling the number of filters.
        # However, due to the masking we need to be careful when splitting the
        # output tensor. The masked convolutional layer will associate the first
        # one third of the channels with color R, the second third with color G,
        # and the last third with color B. We will split the channels into odds
        # and evens in order to preserve the color mapping.
        self.h_conv = MaskConv2d(
            mask_type, in_channels, 2*out_channels, kernel_size=(1, k), **kwargs)

        # NOTE: Not sure why we need to have this layer here, when we could simply
        # add the outputs of the vertical and the horizontal convolution as they
        # have the same shapes. Probably the authors where experimenting with
        # settings where the vertical convolution had a different number channels
        # from the horizontal convolution.
        #
        # The output of the horizontal convolutions needs to be combined with
        # the output of the vertical convolution before passing the result
        # through the gated unit. In order not to violate the auto-regressive
        # property of the channels of each pixel, we need to use masked
        # convolutional layers.
        # Note that we are using two separate layers in order not to mix the
        # channels from the two separate vertical convolutions.
        # We do not need to have a bias term in these layers as the horizontal
        # convolution already adds a bias.
        self.v1toh = MaskConv2d("B", out_channels, out_channels, 1, bias=False)
        self.v2toh = MaskConv2d("B", out_channels, out_channels, 1, bias=False)

        # Finally, the horizontal convolution stack uses a residual connection
        # to skip-connect the input. We need to equalize the number of channels
        # before summing the two tensors and for this reason we use a `1 x 1`
        # convolution applied on the output of the horizontal convolutional layer.
        # Note that again we need to use a masked convolution in order not to
        # violate the auto-regressive property of the channels of each pixel.
        # Note that being a residual layer, this layer does not need to have a
        # bias term. Any bias added here could simply be learned by the bias
        # term of the previous layer.
        self.htoh = MaskConv2d("B", out_channels, in_channels, 1, bias=False)

    def forward(self, x):# xv, xh):
        """Perform a forward pass through the gated convolutional layer. This
        layer accepts two tensors with the same number of channels and returns
        two tensors corresponding to the outputs of the vertical and the
        horizontal convolution.

        Args:
            x (tuple(Tensor, Tensor)): A tuple of two tensors of the same shape
                (B, C, H, W). The first tensor will be the input to the vertical
                convolution and the second - to the horizontal convolution.

        Returns:
            out (tuple(Tensor, Tensor)): A tuple of two tensors. The first tensor
                is of shape (B, Cv, H, W), giving the output of the vertical
                convolution. The second tensor is of shape (B, C, H, W), giving
                the output of the horizontal convolution.
        """
        xv, xh = x
        _, _, H, W = xv.shape

        # Vertical convolution stack.
        vc = self.v_conv(xv)   # shape (B, C, H + k/2, W + k/2) due to padding
        vc = vc[:, :, :H, :W]  # crop the spatial dimensions
        vc_1, vc_2 = torch.chunk(vc, chunks=2, dim=1)  # split the feature maps
        v_out = torch.tanh(vc_1) * torch.sigmoid(vc_2) # gated activation

        # Horizontal convolution stack.
        hc = self.h_conv(xh)
        hc_1, hc_2 = hc[:, ::2], hc[:, 1::2] # split the feature maps into odds and evens
        hc_1 = self.v1toh(vc_1) + hc_1       # connect vertical to horizontal
        hc_2 = self.v2toh(vc_2) + hc_2
        h_out = torch.tanh(hc_1) * torch.sigmoid(hc_2) # gated activation
        h_out = self.htoh(h_out) + xh                  # residual connection

        # Note that because of the residual connection, the output of the
        # horizontal convolution has the same number of channels as the input.
        # Usually the residual connection is implemented as:
        # `h_out = h_out + conv(xh)`.
        # This way we would modify the number of channels of the input instead
        # of the output.

        return (v_out, h_out)

#