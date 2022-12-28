import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    """MaskedLinear is the same as the linear layer but applies a mask on the
    weights.
    """

    def __init__(self, fan_in, fan_out, mask):
        """Init a Masked Linear layer.

        Args:
            fan_in (int): Number of input units.
            fan_out (int): Number of output units.
            mask (Tensor): A boolean tensor of shape (fan_in, fan_out) defining
                the mask of the layer.
        """
        # Initialize the masked linear module the same way as a normal linear module.
        super().__init__(fan_in, fan_out, bias=True)

        # Instead of `self.mask = mask` we will use `register_buffer`.
        # This provides the benefit of pushing both the buffer and the model
        # parameters to the same device when calling `model.to(device)`.
        # Checkout:
        # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/8
        #
        # Also, note that we are transposing the mask. The reason for this is that
        # the weight matrix is also stored in transposed form by PyTorch.
        assert mask.shape == (fan_in, fan_out), "mask dimension must match layer dimensions"
        self.register_buffer("mask", mask.type(dtype=torch.bool).T)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)

#