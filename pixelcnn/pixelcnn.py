import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_norm import PositionalNorm
from conv2d_mask import MaskConv2d
from residual_block import ResBlock


class PixelCNN(nn.Module):
    """PixelCNN is an implementation of the generative pixel convolutional network
    described in "Pixel Recurrent Neural Networks" by van den Oord et al., 2016.

    The model takes a `C x H x W` image as input and produces `d x C x H x W`
    predictions as output. The model scans the image row-by-row pixel-by-pixel
    and predicts the conditional distribution over the `d` possible pixel values
    given the scanned context.

    The PixelCNN uses multiple convolutional layers that preserve the spatial
    resolution. Down-sampling is not applied. The convolving kernel for every
    layer is masked in order to avoid seeing the future context.

    Note that all convolutional layers of the model are initialized with the
    same number of filters. One reason for this might be the fact that the input
    is never down-sampled. Usually, in deep convolutional networks the resolution
    (i.e., spatial dimensions) decreases while the number of channel increases
    up until the point where a fully connected layer aggregates all the features.
    In this model, however the spatial dimensions remain unchanged throughout
    and, thus, it might be reasonable to also keep the channels constant.
    """

    def __init__(self, input_shape, color_depth, n_blocks, filters, kernel_size):
        """Init PixelCNN network.

        Args:
            input_shape (tuple[int]): The shape of the input tensors. Images
                should be reshaped channels first, i.e. input_shape = (C, H ,W).
            color_depth (int): The color depth per channel of the image.
                Given a 24-bit RGB image with 8 bits per color means that we
                have a color depth of `256` for every one of the three colors.
            n_blocks (int): Number of residual blocks in the network.
            filters (int): All convolutional layers of the model are initialized
                with the same number filters.
            kernel_size (int): The size of a square kernel to be used for all
                convolutional layers.
        """
        super().__init__()
        self.input_shape = input_shape
        self.n_colors = color_depth
        self.n_blocks = n_blocks
        self.filters = filters
        self.kernel_size = kernel_size

        # Initialize the architecture of the network.
        # Two types of masks are used: **Mask A** and **Mask B**. Mask A will
        # mask the current pixel and all the following pixels, and is applied
        # only to the first layer of the model. Mask B relaxes the restriction
        # of Mask A by not masking the current pixel and is applied to all the
        # subsequent layers. The reason for having two types of masks is the
        # following: every neuron `i_1` from the first layer will be independent
        # of pixels (i, i+1, ...), thus at the second layer we are allowed to
        # connect neuron `i_2` to neuron `i_1`.
        #
        #         neuron `i_2`    *   Layer 2
        #                         |
        #         neuron `i_1`    *   Layer 1
        #                      /  x                   x = mask, i.e. no connection
        #                     *   *   Input layer
        #         pixels   `i-1` `i`
        #
        layers = []
        in_channels = input_shape[0]
        # The kernel size for the first layer is hard-coded to 7, which seems to
        # work fine for images of size `32 x 32`.
        layers.append(MaskConv2d("A", in_channels, filters, kernel_size=7))
        for _ in range(n_blocks):
            # The model uses residual blocks of masked convolutional layers.
            # Before every residual block apply normalization along the channel
            # dimension. The original paper does not use normalization layers,
            # however adding them greatly improves the performance.
            # We use a custom positional normalization layer that normalizes
            # specifically along the channel dimension. In addition, the channels
            # are divided into three separate groups and each group is normalized
            # separately in order to respect the auto-regressive property.
            layers.extend([
                PositionalNorm("channels_first", filters // 3),
                ResBlock(filters, kernel_size),
            ])
        # At the end the model uses two consecutive `1 x 1` convolutional layers
        # with a fixed size. The paper states that the number of filters should
        # be 1024, however we will use a value that is divisible by 3.
        layers.extend([nn.ReLU(), MaskConv2d("B", filters, out_channels=768, kernel_size=1)])
        layers.extend([nn.ReLU(), MaskConv2d("B", 768, self.n_colors * in_channels, 1)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Perform a forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W). Note that the input
                must be the raw pixel values of the image.


        Returns:
            logits (Tensor): Output tensor of shape (B, color_depth, C, H, W)
                giving the un-normalized logits for each dimension of the input.
        """
        x = x.to(self.device).contiguous().float()
        batch_size = x.shape[0]

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
        mean, std = (self.n_colors - 1) / 2, (self.n_colors - 1) / 2
        x = (x - mean) / std

        logits = self.net(x)
        logits = logits.view(batch_size, self.n_colors, *self.input_shape)
        return logits

    @torch.no_grad()
    def sample(self, n=1):
        """Generate samples using the network model.
        In order to generate samples the model has to perform one forward pass
        per input dimension, i.e. we need to perform `C*H*W` forward passes.
        We will use three nested for-loops in order to generate the image
        pixel-by-pixel and channel-by-channel.

        Args:
            n (int): Number of samples to be generated. Default value is 1.

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
        params = {"kwargs": {
                "input_shape": self.input_shape,
                "color_depth": self.n_colors,
                "n_blocks": self.n_blocks,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
            },
            "state_dict": self.state_dict()
        }
        torch.save(params, path)

#