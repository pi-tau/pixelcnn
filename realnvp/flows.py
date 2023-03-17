import numpy as np
import torch
import torch.nn as nn


class AffineCoupling(nn.Module):
    """The affine coupling flow splits the input into two parts, `x[1:k]` and
    `x[k+1:d]`. The first part remains unchanged by the flow, and the second part
    is transformed by an affine transformation.
    ```
        z[1:k] = x[1:k]
        z[k+1:d] = s * x[k+1:d] + t
    ```
    Here `s` and `t` are the scale and the translation for the affine
    transformation and are functions of the first part `x[1:k]`.
    """

    def __init__(self, net, mask):
        """Init an affine coupling flow.

        Args:
            net: nn.Module
                PyTorch neural network module used for computing the scale and
                the translation of the affine transformation.
            mask: torch.Tensor
                Tensor of the same size as the input of the network, denoting
                the elements of the input that are to be masked.
        """
        super().__init__()
        self.register_buffer("mask", mask.type(dtype=torch.bool))
        self.register_module("net", net)
        self.rescale = nn.Parameter(torch.zeros(size=(1,)), requires_grad=True) # (1, c_in, 1, 1)
        self.reshift = nn.Parameter(torch.zeros(size=(1,)), requires_grad=True)

    def forward(self, x, invert=False):
        """Transform the input using the affine coupling flow.

        Args:
            x: torch.Tensor
                Input tensor of the same shape as the input to the network.
            invert: bool, optional
                If True, the inverted flow will be applied. Default: False.

        Returns:
            z: torch.Tensor
                Tensor of the same shape as the input giving the transformed variables.
            log_det: float
                The log determinant of the Jacobian.
        """
        # Apply the network to the masked input.
        B = x.shape[0]
        mask = self.mask.unsqueeze(dim=0)   # add batch dimension
        x_in = x * mask
        out = self.net(x_in)

        # We will be learning the log of the scale in order for the flow to be
        # invertible (scale must never be zero!). However, due to the exp() we
        # could have a sudden large change of the scale that would destabilize
        # the training. Thus, we will apply `tanh` on the log scales and we will
        # rescale them back using learnable rescaling parameters.
        log_s, t = torch.chunk(out, chunks=2, dim=1)
        log_s = self.rescale * torch.tanh(log_s) + self.reshift

        # Use the inverted mask to mask the log scale and the translation.
        log_s = ~mask * log_s
        t = ~mask * t
        if invert:
            z = (x - t) * (-log_s).exp()
            log_det = -log_s.reshape(B, -1).sum(dim=-1) # negative sum over non-batch dims
        else:
            z = x * log_s.exp() + t
            log_det = log_s.reshape(B, -1).sum(dim=-1) # sum over non-batch dims
        return z, log_det


class LogitTransform(nn.Module):
    """Logit transformation flow transforms real-valued data from the unit
    interval [0, 1] spreading it back across the real numbers. The flow uses the
    logit function -- the inverse of the sigmoid.
    ```
        z = logit(a/2 + (1-a)x)
    ```
    The coefficient `alpha` controls the size of the spread, with `alpha=0` leading
    to F: [0, 1] -> R.
    """

    def __init__(self, alpha=0.1):
        """Init a logit transformation flow.

        Args:
            alpha: float, optional
                Coefficient controlling the size of the spread. Default: 0.1
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, x, invert=False):
        """Transform the input using the logit transform flow.

        Args:
            x: torch.Tensor
                Input tensor of any shape.
            invert: bool, optional
                If True, the inverted flow will be applied. Default: False.

        Returns:
            z: torch.Tensor
                Tensor of the same shape as the input giving the transformed variables.
            log_det: float
                The log determinant of the Jacobian.
        """
        B = x.shape[0]
        if invert:
            # Because of the `alpha` coefficient, the reverse does not map to [0, 1]
            x_in = torch.sigmoid(x)
            z = (x_in - self.alpha / 2) / (1 - self.alpha)
            # dsigm(x)/dx = sigm(x)(1 - sigm(x))
            log_det = -np.log(1-self.alpha) + torch.log(x_in) + torch.log(1-x_in)
            log_det = log_det.reshape(B, -1).sum(dim=-1) # sum over non-batch dims
        else:
            x_in = self.alpha / 2 + (1 - self.alpha) * x
            z = torch.log(x_in) - torch.log(1-x_in)         # logit(x_in)
            # Note that this is an element-wise flow, thus, the Jacobian is diagonal.
            # dz/dx = dx_in/dx * 1 / (x_in (1 - x_in))
            # log dz/dx = log dx_in/dx - log(x_in (1-x_in))
            log_det = np.log(1-self.alpha) - torch.log(x_in) - torch.log(1-x_in)
            log_det = log_det.reshape(B, -1).sum(dim=-1) # sum over non-batch dims
        return z, log_det


class Dequantize(nn.Module):
    """De-quantization transforms discrete data into continuous data by adding
    noise and mapping to [0,1].
    """

    def __init__(self, n_bins):
        """Init a de-quantization transform.

        Args:
            n_bins: int
                Number of bins of the discrete data.
        """
        super().__init__()
        self.n_bins = float(n_bins)

    def forward(self, x, invert=False):
        """Transform the input using the de-quantize transform flow.

        Args:
            x: torch.Tensor
                Input tensor of any shape.
            invert: bool, optional
                If True, the inverted flow will be applied. Default: False.

        Returns:
            z: torch.Tensor
                Tensor of the same shape as the input giving the transformed variables.
            log_det: float
                The log determinant of the Jacobian.
        """
        size = np.prod(x.shape[1:])
        x = x.float()
        if invert:
            z = x * self.n_bins
            log_det = np.log(self.n_bins) * size # sum over non-batch dims
        else:
            # de-quantization: add noise and map to [0, 1]
            z = (x + torch.rand_like(x)) / self.n_bins
            log_det = -np.log(self.n_bins) * size # negative sum over non-batch dims
        return z, log_det


class SqueezeFlow(nn.Module):
    """Squeeze flow transforms images by reducing the spatial dimensions in half
    while scaling the number of channels by 4.
    """

    def forward(self, x_in, invert=False):
        """Transform the input using the squeeze flow.

        Args:
            x: torch.Tensor
                Input tensor of shape (B, C, H, W).
            invert: bool, optional
                If True, the inverted flow will be applied. Default: False.

        Returns:
            z: torch.Tensor
                Tensor of shape (B, 4*C, H//2, W//2) (or (B, C//4, 2*H, 2*W) if
                invert is True), giving the squeezed input.
            log_det: float
                The log determinant of the Jacobian.
        """
        B, C, H, W = x_in.shape
        if invert:
            # unsqueeze 4C x H/2 x W/2 -> C x H x W
            z = x_in.reshape(B, 2, 2, C//4, H, W)
            z = z.permute(0, 3, 4, 1, 5, 2)
            z = z.reshape(B, C//4, 2*H, 2*W)
        else:
            # squeeze C x H x W -> 4C x H/2 x W/2
            #
            # NOTE: Most implementations that I've seen on the internet use the
            # permutation [0, 1, 3, 5, 2, 4] instead, but I think that is not
            # what the original implementation does. They reshape by pushing the
            # spatial dimensions after the channel dimension, thus achieving an
            # intermixing between the two. Official implementation here:
            # https://github.com/tensorflow/models/blob/36101ab4095065a4196ff4f6437e94f0d91df4e9/research/real_nvp/real_nvp_utils.py#L213
            # (link might not work tho, official impl was removed and I dug it up in the commit history)
            z = x_in.reshape(B, C, H//2, 2, W//2, 2)
            z = z.permute(0, 3, 5, 1, 2, 4)
            z = z.reshape(B, 4*C, H//2, W//2)
        log_det = 0. # The log determinant is 0 as there is no computation involved.
        return z, log_det


class SplitFlow(nn.Module):
    """Split flow transforms images by dividing in half along the channels
    dimension and evaluating one part directly on the prior.
    """

    def __init__(self, prior):
        """Init a Split flow.

        Args:
            prior: torch.distribution
                Prior distribution used for evaluating the latents.
        """
        super().__init__()
        self.prior = prior

    def forward(self, x, invert=False):
        """Transform the input using the squeeze flow.

        Args:
            x: torch.Tensor
                Input tensor of shape (B, C, H, W).
            invert: bool, optional
                If True, the inverted flow will be applied. Default: False.

        Returns:
            z: torch.Tensor
                Tensor of shape (B, C//2, H, W) (or (B, 2*C, H, W) if invert is
                True), giving the split input.
            log_det: float
                The log determinant of the Jacobian.
        """
        # Note that log probs from the evaluated split should be added to the
        # latent log probs evaluated at the end of the composite flow and not
        # to the Jacobian. However, it has the same effect as adding it here.
        if invert:
            z_split = self.prior.sample(x.shape).to(x.device)
            z = torch.cat((x, z_split), dim=1)
            log_prob = self.prior.log_prob(z_split)
            log_det = -log_prob.sum(dim=(1,2,3))
        else:
            z, z_split = torch.chunk(x, chunks=2, dim=1)
            log_prob = self.prior.log_prob(z_split)
            log_det = log_prob.sum(dim=(1,2,3))
        return z, log_det


class CompositeFlow(nn.Module):
    """CompositeFlow implements a normalizing flow model by stacking a sequence
    of invertible flows.
    """

    def __init__(self, *flows):
        """Init a composite flow.

        Args:
            flows: list(Flow Model)
                A list of flow models used to create the composite flow.
                Each flow model must implement the `flow` method with an optional
                `invert` parameter for reversing the flow.
        """
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x, invert=False):
        """Transform the input through the composite flow.

        Args:
            x: torch.Tensor
                Input tensor of any shape compatible with the flow transformations.
            invert: bool, optional
                If True, the inverted flow will be applied. Default: False.

        Returns:
            z: torch.Tensor
                Tensor of the same shape as the input, giving the transformed variables.
            log_det: float
                The log determinant of the Jacobian.
        """
        z, log_det = x, 0.
        flows = reversed(self.flows) if invert else self.flows
        for f in flows:
            z, delta_log_det = f(z, invert)
            log_det += delta_log_det
        return z, log_det

#