from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskLinear(nn.Linear):
    """MaskLinear is the same as Linear but applies a mask on the weights."""

    def __init__(self, fan_in, fan_out, mask):
        """Init a masked Linear layer.

        Args:
            fan_in: int
                Number of input units.
            fan_out: int
                Number of output units.
            mask: torch.Tensor
                Boolean tensor of shape (fan_in, fan_out) defining the mask of the layer.
        """
        super().__init__(fan_in, fan_out, bias=True)

        # Note that we are transposing the mask. The reason for this is that
        # the weight matrix is also stored in transposed form by PyTorch.
        assert mask.shape == (fan_in, fan_out), "mask dimension must match layer dimensions"
        self.register_buffer("mask", mask.type(dtype=torch.bool).T)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    """MADE is an implementation of the "Masked-Autoencoder for Distribution
    Estimation" by Germain et al., 2015.
    https://arxiv.org/pdf/1502.03509.pdf

    This model assumes that every dimension of the input takes discrete values
    from a set of `d` numbers, i.e. `x_i` belongs in `{n_1, n_2, ..., n_d}` for
    every `i`. Thus for a D-dimensional input, the output of the model will be a
    tensor of shape `d x D`, where every slice `i` of `D` elements gives the
    un-normalized logits for every dimension corresponding to the number `n_i`.
    This shape of the output is useful for directly plugging in for computing
    the cross entropy loss.

    This model also allows for training on categorical data by converting the
    input into one-hot vector encodings.

    We will impose an ordering on the dimensions of the input and then the output
    of the model will be interpreted as a set of conditional probabilities:
    ```
        x = [x_0, x_1, ..., x_(D-1)]
        ordering = [0, 1, 2, ..., (D-1)] # natural ordering
        out = [p(x_0), p(x_1 | x_0), ..., p(x_(D-1) | x_(D-2), ..., x_1, x_0)]
    ```
    The product of the conditional probabilities yields the joint probability of
    the input:
    ```
        p(x) = p(x_0, x_1, ..., x_(D-1)) = prod p(x_i | x_<i)
    ```

    To impose the auto-regressive property (ordering) the model uses masking:
    each output is reconstructed only from previous inputs given the ordering.
    Since output `out_i = p(x_i | x_<i)` must depend only on the preceding
    inputs `x_<i`, then there must be no computational path between the output
    unit `out_i` and any of the input units `x_i, x_(i+1), ..., x_(D-1)`. Thus,
    we need to zero these connections by multiplying the weights matrix of every
    layer by a binary mask matrix.

    This code is inspired by Andrej Karpathy's implementation given here:
    https://github.com/karpathy/pytorch-made
    """

    def __init__(self, input_shape, d, hidden_sizes, ordering=None, one_hot=False):
        """Init MADE network.

        Args:
            input_shape: list[int]
                The shape of the input tensors.
            d: int
                The number of possible discrete values for each random variable.
            hidden_sizes: list[int]
                List of sizes for the hidden layers.
            ordering: list[int], optional
                List of integers giving an ordering for the dimensions of the
                input. Default value is None, corresponding to a natural
                ordering, i.e. [0, 1, ...., (D-1)].
            one_hot: bool, optional
                Boolean flag indicating weather the input should be converted to
                one-hot encoding.
        """
        super().__init__()
        self.d = d
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        assert self.num_layers > 0, "cannot use a linear model"

        # We will consider every dimension D of the input as a random variable.
        # Each input dimension is a discrete random variable that takes on one
        # of `d` possible values.
        # `nout` will be `d` x `nin` dimensional, the first `nin` numbers represent
        # the probability mass function for the first dimension, and so on...
        self.nin = prod(input_shape)
        self.nout = d * self.nin

        # Create the masks for the Masked Linear layers.
        # The masks for every layer of the model are constructed in the `_create_masks`
        # method. Afterwards, we initialize the layers of the network as MaskedLinear
        # layers.
        self.ordering = torch.arange(self.nin) if ordering is None else torch.IntTensor(ordering)
        self.one_hot = one_hot
        masks = self._create_masks()

        # Initialize the architecture of the network.
        layers = []
        nin = self.nin if not one_hot else self.nin * self.d
        sizes = [nin] + hidden_sizes
        for i, fan_in, fan_out in zip(range(len(sizes)-1), sizes[:-1], sizes[1:]):
            layers.extend([
                MaskLinear(fan_in, fan_out, masks[i]),
                nn.ReLU(),
            ])
        layers.append(MaskLinear(fan_out, self.nout, masks[self.num_layers]))
        self.net = nn.Sequential(*layers)

    def _create_masks(self):
        """This function creates the masks for the weights of the neural network.
        The masks are created in such a way so that the auto-regressive property
        is respected.

        Returns:
            masks: dict
                A dictionary of Tensors representing the mask for each layer of
                the neural network. Calling `mask[i]` will return the mask
                Tensor for the i-th layer.
        """
        L = self.num_layers
        masks = [None] * (L + 1)
        m = {-1: self.ordering}

        for l in range(L):
            # To impose the auto-regressive property we will assign to each
            # unit `k` of the hidden layer `l` a value `m_l(k)`. This number
            # gives the maximum value of the units from the previous layer to
            # which this unit can be connected. Thus `k` is only connected to
            # units of the previous layer `k'` for which `m_l(k) >= m_(l-1)(k')`.
            # Thus, `mask[k, k'] = 1` if `m_l(k) >= m_(l-1)(k')`.
            #
            # In the case of the first hidden layer (l=0), the value `m_0(k)`
            # gives the highest index of the units from the input layer to which
            # the current unit can be connected. This is achieved by taking `l=-1`
            # to mean the input layer and defining `m_(-1)(i) = i`.
            #
            # To avoid unconnected units. the value `m_l(k)` should be greater
            # than or equal to the minimum connectivity at the previous layer,
            # i.e., min_k' m_(l-1)(k').
            # Suppose we choose `m_l(k) = min_k' m_(l-1)(k') - c`. Then the unit
            # `k` has to be connected to units from the previous layer `k'` for
            # which `m_(l-1)(k') <= m_l(k)`, but since `m_l(k)` is smaller than
            # the minimum, then there are no connections for unit `k`.
            # Note that `m_l(k) = nin-1` is disallowed, since no unit should
            # depend on `x_(nin-1)`. (nin is the dimension of the input D)
            K, K_prime = self.hidden_sizes[l], len(m[l-1])
            m[l] = torch.randint(m[l - 1].min(), self.nin - 1, size=(K,))   # shape = (K,)
            masks[l] = (m[l - 1].reshape(K_prime, 1) <= m[l].reshape(1, K)) # shape = (K_prime, K)

        # For the mask of the output layer (l=L) we need to encode the constraint
        # that the i-th output depends only the preceding inputs `x_<i`.
        # Therefore, the output weights can only connect i-th output to units `k`
        # from the last hidden layer with `m_(L-1)(k) < i`.
        # Note that the i-th output is actually represented by `d` consecutive
        # units encoding the pmf of the i-th dimension. Thus, the mask for the
        # output layer is resized to take this into account.
        K_1, K_2 = len(m[L-1]), len(m[-1])
        masks[L] = (m[L-1].reshape(K_1, 1) < m[-1].reshape(1, K_2)) # shape (K, nin)
        masks[L] = masks[L].repeat_interleave(self.d, dim=1)        # shape (K, nout)

        # In case the input needs to be one-hot encoded, then resize the mask
        # of the first hidden layer accordingly.
        if self.one_hot:
            masks[0] = masks[0].repeat_interleave(self.d, dim=0) # shape (K, d*nin)

        return masks

    def forward(self, x):
        """Perform a forward pass through the network.

        Args:
            x: torch.Tensor
                Tensor of shape (B, d1, d2, ...). Every value of the input
                tensor must take discrete values from a set of `d` numbers.

        Returns:
            logits: torch.Tensor
                Tensor of shape (B, d, d1, d2, ...) giving the un-normalized
                logits for each dimension of the input.
        """
        x = x.to(self.device).contiguous()
        B = x.shape[0]

        if self.one_hot:
            x = x.long()
            x_oh = F.one_hot(x, num_classes=self.d).float()
            inp = x_oh.reshape(B, -1)
        else:
            x = x.float()
            inp = x.view(B, self.nin) # runtime error if shapes do not match

        logits = self.net(inp)
        logits = logits.reshape(B, self.nin, self.d)
        logits = logits.permute(0, 2, 1).contiguous().view(B, self.d, *self.input_shape)

        return logits

    def sample(self, n=1):
        """Generate samples using the network model.
        In order to generate samples the model has to perform one forward pass
        per input dimension, i.e. we need to perform `self.nin` forward passes.
        On the i-th forward pass the model calculates the conditional probability
        `p(x_i | x_<i)` and samples from it.

        Args:
            n: int, optional
                Number of samples to be generated. Default: 1.

        Returns:
            samples: torch.Tensor
                Tensor of shape (n, *input_shape) giving the sampled data points.
        """
        self.inv_ordering = {x.item(): i for i, x in enumerate(self.ordering)}
        samples = torch.zeros(size=(n, self.nin), device=self.device)
        with torch.no_grad():
            for i in range(self.nin):
                logits = self(samples).view(n, self.d, self.nin)[:, :, self.inv_ordering[i]]
                probs = F.softmax(logits, dim=1)
                samples[:, self.inv_ordering[i]] = torch.multinomial(probs, 1).squeeze(-1)
        samples = samples.view(n, *self.input_shape)
        return samples

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
                "d": self.d,
                "hidden_sizes": self.hidden_sizes,
                "ordering": self.ordering.tolist(),
                "one_hot": self.one_hot,
            },
            "state_dict": self.state_dict(),
        }
        torch.save(params, path)

#