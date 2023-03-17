import torch
import torch.nn as nn

from flows import (
    AffineCoupling, LogitTransform, Dequantize,
    SqueezeFlow, SplitFlow, CompositeFlow,
)
from network import ResNet


class RealNVP(nn.Module):
    """RealNVP is an implementation of the realNVP model described here:
    https://arxiv.org/pdf/1605.08803.pdf

    The is an implementation of the model architecture used for training on the
    CIFAR-10 dataset. Although the details of the architecture are not given in
    the paper, they can be found here:
    https://github.com/tensorflow/models/tree/1345ec9bb110ec1173f7558d3d700a5a42ae2b2f/research/real_nvp
    """

    def __init__(self, in_shape, n_colors):
        """Init RealNVP flow model.

        Args:
            in_shape: tuple(int)
                The shape of the input tensors. Images should be reshaped
                channels first, i.e. in_shape = (C, H ,W).
            n_colors: int
                Number of colors per channel of the image. Used for
                de-quantizing and quantizing the data.
        """
        super().__init__()

        # Affine flows will use a ResNet to compute the scale and translation.
        # Combine both computations by doubling the output channels.
        # Define the masking patterns for images.
        make_net = lambda C: ResNet(in_channels=C, out_channels=2*C)
        checkerboard = lambda H, W: ((torch.arange(H).reshape(-1,1)+torch.arange(W)) % 2).to(torch.bool).reshape(1, H, W)
        channelwise = lambda C: (torch.arange(C) % 2).to(torch.bool).reshape(C, 1, 1)

        # Define the base distribution to be used by the flow model.
        prior = torch.distributions.Normal(0., 1.)

        # Create the composite flow by stacking together affine transformations
        # with checkerboard and channelwise maskings. Transformations are repeated
        # with inverted maskings in order for all inputs to be altered. Following
        # such an alternating pattern, the set of units which remain identical
        # in one transformation layer are always modified in the next.
        C, H, W = in_shape
        self.flow = CompositeFlow(
                # Deal with discrete input data.
            Dequantize(n_colors),
            LogitTransform(alpha=0.1),

            AffineCoupling(make_net(C),  checkerboard(H, W)),
            AffineCoupling(make_net(C), ~checkerboard(H, W)),
            AffineCoupling(make_net(C),  checkerboard(H, W)),
            SqueezeFlow(),
                # Converting space to channels and then performing "channelwise"
                # transform does not seem like a very logical thing to do?
                # It looks like we are performing row-by-row coupling, and
                # only at the third flow we have a "true" channelwise coupling.
            AffineCoupling(make_net(4*C),  channelwise(4*C)),
            AffineCoupling(make_net(4*C), ~channelwise(4*C)),
            AffineCoupling(make_net(4*C),  channelwise(4*C)),

                # At this point the original implementation performs unsqueeze
                # and then a so-called "factor_out" which is basically the same
                # as squeeze but arranges the spatial dimensions differently.
                # https://github.com/tensorflow/models/blob/36101ab4095065a4196ff4f6437e94f0d91df4e9/research/real_nvp/real_nvp_multiscale_dataset.py#L734
                # Maybe it's not a big deal if we skip both...
            SplitFlow(prior),
            AffineCoupling(make_net(2*C),  checkerboard(H//2, W//2)),
            AffineCoupling(make_net(2*C), ~checkerboard(H//2, W//2)),
            AffineCoupling(make_net(2*C),  checkerboard(H//2, W//2)),
            AffineCoupling(make_net(2*C), ~checkerboard(H//2, W//2)),
        )

        # Store the final output shape. When sampling, we need to start
        # from this shape in order to build the initial image.
        self.out_shape = (2*C, H//2, W//2)
        self.in_shape = in_shape
        self.n_colors = n_colors
        self.prior = prior

    def log_prob(self, x):
        """Compute the log probabilities for each pixel.

        Args:
            x: torch.Tensor
                Tensor of shape (B, C, H, W). Note that the input must be the
                raw pixel values of the image.

        Returns:
            log_prob: torch.Tensor
                Tensor of the same shape as the input giving the non-reduced
                log probabilities for each of the dimensions of the input.
                Summing over the input dimensions gives the log probability of x.
        """
        x = x.to(self.device).contiguous().float()
        z, log_det = self.flow(x)
        log_pz = self.prior.log_prob(z)
        return log_pz.sum(dim=(1, 2, 3)) + log_det

    @torch.no_grad()
    def sample(self, n=1):
        """Generate samples using the model.

        Args:
            n: int, optional
                Number of samples to be generated. Default: 1.

        Returns:
            samples: torch.Tensor
                Int tensor of shape (n, C, H, W), giving the sampled images.
        """
        # Generate samples from the base distribution and invert the flow.
        z = self.prior.sample(sample_shape=(n,)+self.out_shape)
        z = z.to(self.device).contiguous().float()
        imgs, _ = self.flow(z, invert=True)
        return imgs.int().cpu()

    @property
    def device(self):
        """str: Determine on which device is the model placed upon, CPU or GPU."""
        return next(self.parameters()).device

#