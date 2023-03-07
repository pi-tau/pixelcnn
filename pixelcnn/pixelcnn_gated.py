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

    The model takes a `C x H x W` image as input and produces `d x C x H x W`
    predictions as output. The model scans the image row-by-row pixel-by-pixel
    and predicts the conditional distribution over the `d` possible pixel values
    given the scanned context.

    """

    def __init__(self, input_shape, color_depth, n_blocks, filters, kernel_size):
        """Init GatedPixelCNN network.

        Args:
            input_shape: list[int]
                The shape of the input tensors. Images should be reshaped
                channels first, i.e. input_shape = (C, H ,W).
            color_depth: int
                The color depth per channel of the image. Given a 24-bit RGB
                image with 8 bits per color means that we have a color depth of
                `256` for every one of the three colors.
            n_blocks: int
                Number of gated convolutional blocks in the network.
            filters: int
                All convolutional layers of the model are initialized with the
                same number filters.
            kernel_size: int
                The size of a the kernel to be used for all convolutional layers.
        """
        super().__init__()
        self.input_shape = input_shape
        self.n_colors = color_depth
        self.n_blocks = n_blocks
        self.filters = filters
        self.kernel_size = kernel_size

        # Initialize the architecture of the model.
        # Two types of masks are used: **Mask A** and **Mask B**. Mask A will
        # be applied only to the first layer, and all other layers will have
        # Mask B.
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
        for i in range(n_blocks):
            # The gated convolutional layers replace the residual blocks from
            # the original PixelCNN. Note that the horizontal convolution stack
            # has a residual connection and, thus, we do not need to create a
            # residual block.
            # No need of a non-linear activation either.
            # After every gated layer we will apply a normalization layer that
            # normalizes specifically along the channel dimension. The channels
            # are split into three groups, corresponding to each color, and are
            # normalized separately.
            # Normalization is applied separately for the vertical convolutional
            # stack and for the horizontal convolutional stack.
            # The gated convolutional layers are initialized with no bias, as
            # the normalization layer will re-center the data anyway, removing
            # the bias.
            mask_type = "A" if i == 0 else "B"
            self.gated_layers.append(
                GatedConv2d(mask_type, filters, filters, kernel_size, bias=False))
            self.v_norm_layers.append(PositionalNorm("channels_first", filters // 3))
            self.h_norm_layers.append(PositionalNorm("channels_first", filters // 3))

        # The final pair of layers are `1 x 1` convolutional layers that gradually
        # increase the channel dimension in order to produce a set of logits for
        # every (C, H, W) position. Again, we use standard MaskConv2d layers in
        # order to change the number of channels. The paper states that the number
        # of filters should be 1024, but we will use a value that is divisible by 3.
        self.out_conv = nn.Sequential(
            MaskConv2d("B", filters, out_channels=768, kernel_size=1),
            nn.ReLU(),
            MaskConv2d("B", 768, self.n_colors * in_channels, 1),
        )

    def forward(self, x):
        """Perform a forward pass through the network.

        Args:
            x: torch.Tensor
                Tensor of shape (B, C, H, W). Note that the input must be the
                raw pixel values of the image.

        Returns:
            logits: torch.Tensor
                Tensor of shape (B, color_depth, C, H, W) giving the un-normalized
                logits for each dimension of the input.
        """
        x = x.to(self.device).contiguous().float()
        B, C, H, W = x.shape
        d = self.n_colors

        # Normalize the input.
        # Note that we are not normalizing the input using the training data
        # statistics. Rather we are using fixed values for the mean and the std.
        # This is ok, given that we are working with natural images. We expect
        # the mean color value of the data to be approximately equal to the
        # mean color.
        # We could also preprocess the data the usual way, but then the
        # calculated mean and std would need to be passed to the `sample` method
        # as well. Passing the generated raw pixel values would be incorrect as
        # they need to be normalized before forwarding.
        mean, std = (d-1)/2, (d-1)/2
        x = (x - mean) / std

        x_in = self.in_conv(x)
        xv, xh = x_in, x_in
        for i in range(self.n_blocks):
            # Note that the gated convolutional layer accepts a tuple of two
            # tensors. One will be used for the vertical stack and the other for
            # the horizontal stack. The output is also a tuple of two tensors.
            xv, xh = self.gated_layers[i]((xv, xh))
            xv = self.v_norm_layers[i](xv)
            xh = self.h_norm_layers[i](xh)

        # We need to redirect only the output of the horizontal stack to the
        # output layers.
        logits = self.out_conv(xh)

        # Instead of doing `logits.view(B, d, C, H, W)`, we are first splitting
        # along the channels. This needs to be done because the last masked
        # convolution has grouped the logits into `C` groups, depending on their
        # color channel. Reshaping in the wrong order splits the groups correctly
        # taking into account the masking.
        logits = logits.view(B, C, d, H, W).permute(0, 2, 1, 3, 4)
        return logits

    @torch.no_grad()
    def sample(self, n=1):
        """Generate samples using the network model.
        In order to generate samples the model has to perform one forward pass
        per input dimension, i.e. we need to perform `C x H x W` forward passes.
        We will use three nested for-loops in order to generate the image
        pixel-by-pixel and channel-by-channel.

        Args:
            n: int, optional
                Number of samples to be generated. Default: 1.

        Returns:
            samples: torch.Tensor
                Tensor of shape (n, C, H, W), giving the sampled data points.
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

    @classmethod
    def load(cls, path):
        """Load the model from a file."""
        params = torch.load(path, map_location=lambda storage, loc: storage)
        kwargs = params["kwargs"]
        model = cls(**kwargs)
        model.load_state_dict(params["state_dict"])
        return model

    def save(self, path):
        """Save the model to a file."""
        params = {
            "kwargs": {
                "input_shape": self.input_shape,
                "color_depth": self.n_colors,
                "n_blocks": self.n_blocks,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
            },
            "state_dict": self.state_dict(),
        }
        torch.save(params, path)

#