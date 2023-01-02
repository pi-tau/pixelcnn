import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv2d_gated import GatedConv2d
from conv2d_mask import MaskConv2d
from positional_norm import PositionalNorm


class GatedPixelCNN(nn.Module):
    """GatedPixelCNN is an implementation of the PixelCNN decoder network
    described in the paper "Conditional image generation with PixelCNN decoders"
    by van en Oord et al., 2016.

    The original architecture of the PixelCNN model has a "blind spot" in the
    receptive field. This model uses a modified `Gated` convolutional layer in
    order to fix this problem.
    """

    def __init__(self, input_shape, color_depth, n_blocks, filters, kernel_size):
        """Init GatedPixelCNN network.

        Args:
            input_shape (tuple[int]): The shape of the input tensors. Images
                should be reshaped channels first, i.e. input_shape = (C, H ,W).
            color_depth (int): The color depth per channel of the image.
                Given a 24-bit RGB image with 8 bits per color means that we
                have a color depth of `256` for every one of the three colors.
            n_blocks (int): Number of gated convolutional blocks in the network.
            filters (int): All convolutional layers of the model are initialized
                with the same number filters.
            kernel_size (int): The size of a the kernel to be used for all
                convolutional layers.
        """
        super().__init__()
        self.input_shape = input_shape
        self.n_colors = color_depth
        self.n_blocks = n_blocks

        # Initialize the architecture of the model.
        # As with the original PixelCNN, here we also use two types of masks:
        # **Mask A** and **Mask B**, where Mask A will be applied only to the
        # first layer, and all other layers will have Mask B.
        # Note that the first layer is a standard MaskConv2d layer instead of a
        # gated convolutional layer. The reason for this is that the architecture
        # of the gated layer does not allow the horizontal convolution to change
        # the number of channels and the output has the same number of channels
        # as the input.
        # The kernel size for the first layer is hard-coded to 7, which seems to
        # work fine for images of size `32 x 32`.
        in_channels = input_shape[0]
        self.in_conv = nn.Sequential(
            MaskConv2d("A", in_channels, filters, kernel_size=7),
            nn.ReLU(),
        )

        # Initialize a block of gated convolutional layers.
        self.gated_layers = nn.ModuleList()
        self.v_norm_layers = nn.ModuleList()
        self.h_norm_layers = nn.ModuleList()
        for _ in range(n_blocks):
            # The gated convolutional layers replace the residual blocks from
            # the original PixelCNN. Note that the horizontal convolution stack
            # has a residual connection and, thus, we do not need to create a
            # residual block.
            # No need of a non-linear activation neither.
            # After every gated layer we will apply a normalization layer that
            # normalizes specifically along the channel dimension. The channels
            # are split into three groups, corresponding to each color, and are
            # normalized separately.
            # Normalization is applied separately for the vertical convolutional
            # stack and for the horizontal convolutional stack.
            # The gated convolutional layers are initialized with no bias, as
            # the normalization layer will re-center the data anyway, removing
            # the bias.
            self.gated_layers.append(
                GatedConv2d("B", filters, filters, kernel_size, bias=False))
            self.v_norm_layers.append(PositionalNorm("channels_first", filters // 3))
            self.h_norm_layers.append(PositionalNorm("channels_first", filters // 3))

        # The final pair of layers are `1 x 1` convolutional layers that gradually
        # increase the channel dimension in order to produce a set of logits for
        # every (C, H, W) position. Again, we use standard MaskConv2d layers in
        # order to change the number of channels.
        self.out_conv = nn.Sequential(
            MaskConv2d("B", filters, out_channels=768, kernel_size=1),
            nn.ReLU(),
            MaskConv2d("B", 768, self.n_colors * in_channels, 1),
        )

    def forward(self, x):
        """Perform a forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            logits (Tensor): Output tensor of shape (B, color_depth, C, H, W)
                giving the un-normalized logits for each dimension of the input.
        """
        x = x.to(self.device).contiguous().float()
        batch_size = x.shape[0]

        x_in = self.in_conv(x)
        xv, xh = x_in, x_in
        for i in range(self.n_blocks):
            # Note that the gated convolutional layer accepts a tuple of two
            # tensors. One will be used for the vertical stack and the other for
            # the horizontal stack. The output is also a tuple of two tensors.
            xv, xh = self.gated_layers[i]((xv, xh))
            xv = self.v_norm_layers[i](xv)
            xh = self.h_norm_layers[i](xh)

        # We need to redirect the output of the horizontal stack to the output
        # layers.
        logits = self.out_conv(xh)
        logits = logits.view(batch_size, self.n_colors, *self.input_shape)
        return logits

    @torch.no_grad()
    def sample(self, n=1):
        """Generate samples using the network model.
        In order to generate samples the model has to perform one forward pass
        per input dimension, i.e. we need to perform `C x H x W` forward passes.
        We will use three nested for-loops in order to generate the image
        pixel-by-pixel and channel-by-channel.

        Args:
            n (int): Number of samples to be generated. Default values is 1.

        Returns:
            samples (Tensor): A tensor of shape (n, C, H, W), giving the
                sampled data points generated by the model.
        """
        x_in = torch.zeros(size=(n,)+self.input_shape, device=self.device)

        C, H, W = self.input_shape
        # We are generating the image row-by-row from top to bottom.
        for h in range(H):
            for w in range(W):
                # Note that we need to generate the image pixel-by-pixel. Thus,
                # we need to generate all three color channels of a pixel before
                # moving to the next one. Therefore, looping over the channels
                # has to be the inner-most loop.
                for c in range(C):
                    logits = self(x_in)              # shape (n, d, C, H, W)
                    logits = logits[:, :, c, h, w]   # we are interested only in these
                    probs = F.softmax(logits, dim=1) # along d dim
                    vals = torch.multinomial(probs, 1).squeeze(dim=-1)
                    x_in[:, c, h, w] = vals

        return x_in

    @property
    def device(self):
        """str: Determine on which device is the model placed upon, CPU or GPU."""
        return next(self.parameters()).device

#