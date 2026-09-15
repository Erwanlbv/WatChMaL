"""Shared node-embedding block for the graph models, and the activation registry."""

import torch.nn as nn


_ACTIVATIONS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
}


def get_activation(name: str) -> nn.Module:
    """Return a fresh activation module by name.

    A new instance is returned on every call. Activation modules are stateless, but
    sharing one instance across a network makes the module tree misreport its
    structure, so instances are not cached.
    """
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}. Choose from {sorted(_ACTIVATIONS)}.")
    return _ACTIVATIONS[key]()


# ============================================================
# Node encoder
# ============================================================
class NodeEncoder(nn.Module):
    """Embed per-node input features into the hidden dimension.

    The block is ``Linear -> activation -> LayerNorm -> Linear``, optionally preceded by
    a normalisation of the raw input channels.

    Parameters
    ----------
    in_channels : int
        Width of the input feature vector per node.
    hidden_channels : int
        Width of the embedding.
    activation : str
        One of ``relu``, ``gelu``, ``silu``.
    input_norm : bool
        Apply ``nn.LayerNorm(in_channels)`` to the raw features before the first linear
        layer. **Default False, and changing it is rarely correct.** ``LayerNorm``
        normalises across the channel axis, and the channels of a PMT hit are physically
        heterogeneous quantities — charge in photoelectrons, time in nanoseconds,
        coordinates in centimetres. Normalising across them removes each node's scale and
        offset. The degenerate case is exact: for ``in_channels=2`` the output of
        ``nn.LayerNorm(2)`` is ``(+1, -1)`` or ``(-1, +1)`` regardless of the input, so
        that ``[0.9, 0.2]`` and ``[5.0, 1.0]`` both map to ``[+1.000, -1.000]`` and only
        the sign of the difference survives. For larger ``in_channels`` the loss is
        partial rather than total, but the invariance to a common offset remains exact.
        Per-channel scaling belongs in the dataset transforms
        (:class:`watchmal.dataset.graph.pyg_transformations.Normalize`), where the
        statistics are per channel and computed over the dataset.
    hidden_norm : bool
        Insert ``nn.LayerNorm(hidden_channels)`` between the activation and the second
        linear layer.
    """

    def __init__(self, in_channels, hidden_channels, activation='relu',
                 input_norm=False, hidden_norm=True):
        super().__init__()

        layers = []
        if input_norm:
            layers.append(nn.LayerNorm(in_channels))
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(get_activation(activation))
        if hidden_norm:
            layers.append(nn.LayerNorm(hidden_channels))
        layers.append(nn.Linear(hidden_channels, hidden_channels))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
