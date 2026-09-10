"""
k-nearest-neighbour graph construction, performed inside a model's forward pass.

A graph neural network over photomultiplier (PMT) hits requires an edge set. Two
placements are possible. Building the edges upstream — in a dataset's ``__getitem__``,
or offline into a stored ``edge_index`` — fixes the neighbour search on the CPU and
transports 2*k*N int64 entries per event through the collate and the host-to-device
copy; for a 3752-hit Hyper-K event at k=5 that is 37520 edges, and ``edge_index``
accounts for 80% of a stored PyG dataset's bytes (measured on a 200-event subset:
40.4 MB of 50.5 MB). Building the edges here instead runs the search on the device the
batch already occupies and keeps the result transient.

Two backends are selected automatically at the first call:

``torch_cluster.knn_graph``
    A compiled CUDA/C++ kernel. Used whenever ``torch_cluster`` imports, which is the
    case in every cluster container listed in ``docs/`` but in none of the CPU-only
    environments described by ``requirements-ci.txt``.

a chunked ``torch.cdist`` + ``topk`` fallback
    Pure torch, no compiled extension. Peak memory is bounded by
    ``chunk_size * N_graph`` rather than ``N_total**2``; time is O(N**2) per graph,
    which is acceptable at the ~2500-hit scale of the smoke datasets and is the
    bottleneck at full detector occupancy.

``torch_geometric.nn.knn_graph`` is not called directly because it dispatches to
``torch_cluster`` or ``pyg-lib`` and raises ``ImportError`` when neither is installed,
which is precisely the environment the fallback exists to serve.

Direct and inverse neighbourhoods
---------------------------------
k-nearest-neighbour is a **directed** and **asymmetric** relation: that u is among the k
nearest PMTs to v does not imply that v is among the k nearest to u. Two edge sets follow
from it, for a given PMT u:

*direct*
    edges from u to the PMTs nearest to u. The convolution on u then aggregates over
    distances measured from u, and u has exactly k edges.

*inverse*
    edges from u to the PMTs v for which u is among the k nearest. Nothing constrains how
    many such v exist, so the number of edges per PMT varies with the local hit density.

An undirected graph makes each edge an edge of both endpoints, and direct and inverse
then coincide; the degree still varies, since it is the size of the union of the two.
``undirected=True`` is the default and the only setting exercised so far.

Duplicates. Symmetrising produces two copies of every reciprocated edge: a mutually
nearest pair (u, v) yields (u, v) twice. Within one convolution that would give v twice
the softmax mass of a neighbour reached by a single edge, which is not the intent — an
edge should serve both endpoints once each, not one endpoint twice. Reciprocation is the
common case rather than a corner: on a 3170-hit Hyper-K event, 41.0 % of the symmetrised
edges at k=4 and 42.7 % at k=8 are duplicates, and the aggregation degree they inflate is
that of 99.5 % and 100 % of the nodes respectively. The edge set is therefore
deduplicated, which also removes that fraction of the message-passing work.

Backends do not agree edge for edge. The detector is a regular lattice of PMTs, so exact
distance ties at the k-th neighbour are common rather than exceptional, and the two
backends resolve them differently: 7.3 % of the edges differ at k=4 and 5.6 % at k=8,
measured over five Hyper-K events. See WatChMaL/WatChMaL#123.

Still to establish: whether a directed graph should use the direct or the inverse
neighbourhood. The stored PyG datasets and this module disagree on that, and only the
undirected case has been tested.
"""

from typing import Optional

import torch

_backend = None          # 'torch_cluster' | 'cdist'
_torch_cluster_knn = None
_announced = False


def _resolve_backend() -> str:
    """Import torch_cluster once, and fall back to the pure-torch implementation."""
    global _backend, _torch_cluster_knn
    if _backend is not None:
        return _backend
    try:
        from torch_cluster import knn_graph as torch_cluster_knn
        _torch_cluster_knn = torch_cluster_knn
        _backend = 'torch_cluster'
    except ImportError:
        _backend = 'cdist'
    return _backend


def _announce(backend: str, k: int) -> None:
    """State the selected backend once per process.

    The backend changes between a laptop and a cluster container without any
    configuration change, and the two differ in cost by orders of magnitude, so the
    choice is reported rather than left to be inferred from the run time.
    """
    global _announced
    if _announced:
        return
    _announced = True
    detail = ("torch_cluster.knn_graph"
              if backend == 'torch_cluster'
              else "chunked torch.cdist + topk (torch_cluster not available)")
    print(f"[knn] edge_index built in the forward pass with {detail} (k={k})", flush=True)


def _knn_cdist(pos: torch.Tensor, batch: Optional[torch.Tensor], k: int,
               chunk_size: int) -> torch.Tensor:
    """Per-graph chunked neighbour search in pure torch."""
    num_nodes = pos.size(0)
    device = pos.device

    if batch is None:
        bounds = [(0, num_nodes)]
    else:
        # A PyG Batch orders nodes by graph, so per-graph counts give contiguous slices.
        counts = torch.bincount(batch)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                             counts.cumsum(0)])
        bounds = [(int(offsets[i]), int(offsets[i + 1])) for i in range(counts.numel())]

    rows, cols = [], []
    for start, end in bounds:
        n = end - start
        if n < 2:
            continue
        graph_pos = pos[start:end]
        # k neighbours excluding self: request k+1 and drop the self column afterwards.
        kk = min(k + 1, n)
        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n)
            dist = torch.cdist(graph_pos[chunk_start:chunk_end], graph_pos)
            idx = dist.topk(kk, dim=1, largest=False).indices
            query = torch.arange(chunk_start, chunk_end, device=device).repeat_interleave(kk)
            rows.append(query + start)
            cols.append(idx.reshape(-1) + start)

    if not rows:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)


def _unique_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Remove repeated (source, target) pairs.

    The pair is encoded as a single integer rather than sorted lexicographically, which
    is what ``torch.unique(..., dim=1)`` would do: one pass over a 1-D tensor is cheaper
    than a row-wise sort, and the edge order carries no meaning here.
    """
    key = edge_index[0] * num_nodes + edge_index[1]
    keep = torch.unique(key, sorted=False, return_inverse=False)
    return torch.stack([keep // num_nodes, keep % num_nodes], dim=0)


def build_knn_edge_index(
    pos: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    k: int = 16,
    loop: bool = False,
    undirected: bool = True,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Build a k-nearest-neighbour ``edge_index`` from node positions.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions, shape ``[N, D]``. ``D`` is 3 for PMT hit coordinates.
    batch : torch.Tensor, optional
        Graph assignment per node, shape ``[N]``. Nodes are never connected across
        graphs. ``None`` treats the whole tensor as a single graph.
    k : int
        Neighbours per node, self excluded unless ``loop`` is set.
    loop : bool
        Retain self-loops.
    undirected : bool
        Append the reversed edges, so that a node both queries its own neighbours and
        answers the nodes that selected it. The result is deduplicated, so a reciprocated
        pair contributes one edge to each endpoint rather than two to one; see the module
        docstring. The degree is then no longer k but the size of the union of the direct
        and inverse neighbourhoods.
    chunk_size : int
        Query nodes per distance-matrix chunk in the fallback backend.

    Returns
    -------
    torch.Tensor
        ``edge_index`` of shape ``[2, E]`` and dtype ``torch.long``, on the device of
        ``pos``.
    """
    backend = _resolve_backend()
    _announce(backend, k)

    if pos.size(0) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)

    # Graph construction is a discrete lookup: no gradient flows through it, and the
    # distance matrices are large enough that keeping them off the autograd tape matters.
    with torch.no_grad():
        if backend == 'torch_cluster':
            edge_index = _torch_cluster_knn(pos, k=k, batch=batch, loop=loop,
                                            flow='target_to_source')
        else:
            edge_index = _knn_cdist(pos, batch, k, chunk_size)
            if not loop:
                edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        if undirected:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_index = _unique_edges(edge_index, pos.size(0))

    return edge_index
