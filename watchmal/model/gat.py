"""
Graph attention network for water-Cherenkov event reconstruction.

Two readouts are available and are selected by ``num_cls_tokens``:

``num_cls_tokens = 0``
    Global mean pooling over the real nodes. Every hit contributes equally to the event
    representation.

``num_cls_tokens > 0``
    Virtual classification (CLS) nodes are appended to every graph and connected to all
    of that graph's real nodes in both directions; the readout is the concatenation of
    their final representations. See Gilmer et al. (ICML 2017) for more details regarding
    the CLS implementation for GNNs.

Edges are obtained one of two ways, selected by ``knn_k``:

``knn_k = None``
    ``data.edge_index`` is read off the batch. This is the path used by the stored PyG
    datasets, whose edges were computed offline.

``knn_k = int``
    The k-nearest-neighbour graph is built from ``data.pos`` inside the forward pass, on
    the device the batch already occupies, and any incoming ``edge_index`` is ignored.
    See :mod:`watchmal.model.knn_edges`.

Provenance. This class merges the graph attention network of the CAVERN framework with
the CLS-token and forward-pass-kNN variant developed downstream in GhostHunter. Two
variants present in those earlier versions are not carried over: the choice between a
concatenated residual, ``MLP(cat([x, agg])) + x``, and the additive transformer-style
residual retained here; and the projection of node features to a wider dimension before
mean pooling.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax, scatter

# WatChMaL imports
from watchmal.model.node_encoder import NodeEncoder, get_activation
from watchmal.model.knn_edges import build_knn_edge_index
from watchmal.utils.logging_utils_caverns import setup_logging

log = setup_logging(__name__)


class AttentionBlock(nn.Module):
    """One message-passing block: multi-head attention over neighbours, then a
    position-wise feed-forward network, each wrapped in a residual connection and a
    layer normalisation.

    The attention is the scaled dot product of Vaswani et al. (NeurIPS 2017) restricted
    to the edges of the graph: for every edge ``(row, col)`` the score is
    ``Q[row] . K[col] / sqrt(head_dim)``, the softmax normalises over all edges sharing
    the same ``row``, and the weighted values are accumulated into ``row``. The node in
    ``edge_index[0]`` is therefore the one that aggregates.
    """

    def __init__(self, hidden_channels: int, num_heads: int, activation: str,
                 mlp_expansion_factor: int):
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)

        self.norm_attention = nn.LayerNorm(hidden_channels)
        self.norm_ffn = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * mlp_expansion_factor),
            get_activation(activation),
            nn.Linear(hidden_channels * mlp_expansion_factor, hidden_channels),
        )

    def attend(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        query = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        key = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        value = self.v_proj(x).view(-1, self.num_heads, self.head_dim)

        scores = (query[row] * key[col]).sum(-1) / self.head_dim ** 0.5
        # num_nodes is passed explicitly: inferring it from the edge index would drop
        # trailing nodes that have no incoming edge.
        weights = softmax(scores, index=row, num_nodes=x.size(0))
        aggregated = scatter(weights.unsqueeze(-1) * value[col], row, dim=0,
                             dim_size=x.size(0), reduce='sum')
        return aggregated.view(-1, self.num_heads * self.head_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.norm_attention(x + self.attend(x, edge_index))
        x = self.norm_ffn(x + self.ffn(x))
        return x


class GraphAttentionNetwork(nn.Module):
    """Graph attention network with an optional CLS-token readout.

    Parameters
    ----------
    in_channels : int
        Number of node feature columns in ``data.x``. It must equal the width the
        dataset and its transforms produce; nothing checks this at construction, and a
        mismatch surfaces as a shape error in the first forward pass.
    hidden_channels : int
        Width of the node representations. Must be divisible by ``num_heads``.
    out_channels : int
        Task output width: the number of classes for classification, or the number of
        components for regression. It must equal ``len(data.dataset.target_names)`` in
        the configuration, which the engine uses to reshape the evaluation output.
    num_layers : int
        Number of attention blocks.
    num_heads : int
        Number of attention heads per block.
    num_cls_tokens : int
        Virtual CLS nodes appended to every graph. ``0`` selects global mean pooling.
    mlp_expansion_factor : int
        Width multiplier of the feed-forward network inside each block. At
        ``hidden_channels=128``, ``num_layers=4``, ``num_heads=4`` and the default value
        of 4, the blocks hold 727 040 of the model's 744 710 parameters (97.6 %).
    activation : str
        One of ``relu``, ``gelu``, ``silu``.
    node_encoder_input_norm : bool
        Normalise the raw node features across channels before the node encoder. Off by
        default; see :class:`watchmal.model.node_encoder.NodeEncoder` for why enabling it
        discards per-channel scale.
    node_encoder_hidden_norm : bool
        Normalise inside the node encoder, between its activation and its second linear
        layer.
    position_dim : int
        Number of leading columns of ``data.pos`` concatenated onto ``data.x`` before the
        node encoder. ``0`` leaves the features untouched, which is correct when the
        coordinates are already columns of ``data.x`` or when the task does not need
        them. **Setting it above 0 appends ``data.pos`` unscaled.** No transform in this
        package normalises ``pos``, since the neighbour search needs it in detector units,
        so the encoder would receive centimetres beside features scaled to [0, 1]: on one
        Hyper-K event the charge and time columns then carry 0.00 % of the input variance
        (1.0e-02 and 1.3e-02 against 3.0e+06, 3.2e+06 and 4.6e+06), and the model does not
        learn. Prefer listing the coordinates among the dataset's feature columns, where
        they are scaled with everything else. Energy and PID can be learned from the graph topology alone; vertex and
        direction regression cannot, since without coordinates in the features the model
        has no way to express a location.
    knn_k : int, optional
        Neighbours per node when the edges are built in the forward pass from
        ``data.pos``. ``None`` reads ``data.edge_index`` off the batch instead.
    knn_loop : bool
        Retain self-loops in the constructed graph. Used only when ``knn_k`` is set.
    knn_undirected : bool
        Add the reversed edges to the constructed graph. Used only when ``knn_k`` is set.
    use_nhits : bool
        Concatenate the per-event hit count ``data.n_hits`` onto the readout vector.
    use_event_total_charge : bool
        Concatenate the per-event total charge ``data.event_total_charge`` onto the
        readout vector. Both quantities are produced by
        :class:`watchmal.dataset.graph.pyg_transformations.AddFeaturesInData` and are
        already scaled by that transform.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        num_heads: int = 4,
        num_cls_tokens: int = 0,
        mlp_expansion_factor: int = 4,
        activation: str = 'relu',
        node_encoder_input_norm: bool = False,
        node_encoder_hidden_norm: bool = True,
        position_dim: int = 0,
        knn_k: Optional[int] = None,
        knn_loop: bool = False,
        knn_undirected: bool = True,
        use_nhits: bool = False,
        use_event_total_charge: bool = False,
    ):
        super().__init__()

        self.num_cls_tokens = num_cls_tokens
        self.use_cls = num_cls_tokens > 0
        self.position_dim = position_dim
        self.knn_k = knn_k
        self.knn_loop = knn_loop
        self.knn_undirected = knn_undirected
        self.use_nhits = use_nhits
        self.use_event_total_charge = use_event_total_charge

        self.encoder = NodeEncoder(
            in_channels + position_dim,
            hidden_channels,
            activation=activation,
            input_norm=node_encoder_input_norm,
            hidden_norm=node_encoder_hidden_norm,
        )

        self.attn_layers = nn.ModuleList([
            AttentionBlock(hidden_channels, num_heads, activation, mlp_expansion_factor)
            for _ in range(num_layers)
        ])

        if self.use_cls:
            self.cls_tokens = nn.Parameter(torch.randn(num_cls_tokens, hidden_channels))
            readout_channels = hidden_channels * num_cls_tokens
        else:
            self.cls_tokens = None
            readout_channels = hidden_channels

        # Normalisation then a single linear map. The readout already carries the whole
        # event: with the CLS readout it is the concatenation of the token
        # representations, each the output of a full attention stack. A hidden layer here
        # adds capacity where the representation is already learned, and the depth belongs
        # in the attention blocks instead.
        classifier_channels = readout_channels + int(use_nhits) + int(use_event_total_charge)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_channels),
            nn.Linear(classifier_channels, out_channels),
        )

    # ------------------------------------------------------------------ #
    # Input assembly
    # ------------------------------------------------------------------ #

    def encoder_input(self, data) -> torch.Tensor:
        """``data.x``, with the leading ``position_dim`` columns of ``data.pos`` appended."""
        if self.position_dim == 0:
            return data.x

        # getattr with a default, not hasattr: on PyG 2.8 a Data object declares `pos`
        # as a property, so hasattr is True even when no positions were stored.
        pos = getattr(data, 'pos', None)
        available = 0 if pos is None else pos.size(-1)
        if available < self.position_dim:
            raise ValueError(
                f"position_dim={self.position_dim} but data.pos has {available} "
                "column(s). Provide hit coordinates in data.pos, or set position_dim=0."
            )
        return torch.cat([data.x, pos[..., :self.position_dim]], dim=-1)

    def edges(self, data, batch: torch.Tensor) -> torch.Tensor:
        """The batch's own ``edge_index``, or a k-nearest-neighbour graph built here."""
        if self.knn_k is None:
            edge_index = getattr(data, 'edge_index', None)
            if edge_index is None:
                raise ValueError(
                    "No edge_index on the batch. Either use a dataset that stores "
                    "edges, or set the model's knn_k so the graph is built in the "
                    "forward pass from data.pos."
                )
            return edge_index

        pos = getattr(data, 'pos', None)
        if pos is None:
            raise ValueError(
                "knn_k is set but the batch has no data.pos. The dataset must place the "
                "hit coordinates in data.pos for the graph to be built here."
            )
        return build_knn_edge_index(
            pos, batch, k=self.knn_k,
            loop=self.knn_loop, undirected=self.knn_undirected,
        )

    def add_cls_nodes(self, x, edge_index, batch, num_graphs):
        """Append ``num_cls_tokens`` virtual nodes per graph, connected to every real
        node of that graph in both directions.

        Returns the augmented ``(x, edge_index, batch)`` and the number of real nodes,
        which is the offset at which the CLS block starts.
        """
        num_real = x.size(0)
        device = x.device

        # cls_tokens is [T, H]; repeating it num_graphs times lays the CLS block out
        # graph-major, so the readout view below recovers graphs in order.
        cls_features = self.cls_tokens.repeat(num_graphs, 1)
        x = torch.cat([x, cls_features], dim=0)

        cls_batch = torch.arange(num_graphs, device=device).repeat_interleave(self.num_cls_tokens)
        batch = torch.cat([batch, cls_batch])

        real_index = torch.arange(num_real, device=device)
        cls_starts = num_real + batch[:num_real] * self.num_cls_tokens
        destination = (
            cls_starts.repeat_interleave(self.num_cls_tokens)
            + torch.arange(self.num_cls_tokens, device=device).repeat(num_real)
        )
        source = real_index.repeat_interleave(self.num_cls_tokens)
        cls_edges = torch.stack([source, destination])
        cls_edges = torch.cat([cls_edges, cls_edges.flip(0)], dim=1)

        return x, torch.cat([edge_index, cls_edges], dim=1), batch, num_real

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def append_event_feature(self, x, data, name):
        if not hasattr(data, name) or getattr(data, name) is None:
            raise ValueError(
                f"use_{name} is set but the batch carries no '{name}'. It is produced by "
                "the AddFeaturesInData transform; add it to the data/transforms config."
            )
        value = getattr(data, name).to(x.dtype).to(x.device).reshape(x.size(0), 1)
        return torch.cat([x, value], dim=1)

    def forward(self, data):
        # num_graphs is read from the batch rather than from batch.max()+1: the two
        # disagree when the last graph of a batch has no hits, which makes the CLS
        # readout view fail and the mean-pool readout return too few rows.
        num_graphs = data.num_graphs
        batch = data.batch

        edge_index = self.edges(data, batch)
        x = self.encoder(self.encoder_input(data))

        if self.use_cls:
            x, edge_index, batch, num_real = self.add_cls_nodes(
                x, edge_index, batch, num_graphs
            )

        for layer in self.attn_layers:
            x = layer(x, edge_index)

        if self.use_cls:
            graph_representation = x[num_real:].view(num_graphs, -1)
        else:
            graph_representation = global_mean_pool(x, batch, size=num_graphs)

        if self.use_nhits:
            graph_representation = self.append_event_feature(
                graph_representation, data, 'n_hits')
        if self.use_event_total_charge:
            graph_representation = self.append_event_feature(
                graph_representation, data, 'event_total_charge')

        return self.classifier(graph_representation)
