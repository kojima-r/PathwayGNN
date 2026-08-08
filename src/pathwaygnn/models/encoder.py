"""The relational graph encoder and its edge-prediction pre-training head.

Tensor shapes below use these symbols:

``N``
    graph nodes (``num_nodes``), or ``n`` for the nodes of one partition batch
``R``
    relation types (``num_relations``)
``E``
    directed edges of the graph being convolved
``H``
    ``hidden_dim``, the width of the node embedding and of every layer
``L``
    ``num_layers``
``K``
    scored edges in one training step (``training.batch_size``)
``D``
    the width of one group of external node vectors (2560 for proteins, 512 for
    chemicals in ``embedding_pc``), and ``m`` the nodes that group covers
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch_geometric.nn import GINConv

from pathwaygnn.data.node_embeddings import (
    NodeEmbeddingTable,
    check_spec,
    load_node_embeddings,
)


class ExternalNodeEmbedding(nn.Module):
    """Adapters that bring pre-computed node vectors to the encoder's width ``H``.

    One ``nn.Linear(D, H)`` per group, because the groups come from different
    source models and therefore have different widths. A node the table covers
    takes its adapted vector — added to its learned ``nn.Embedding`` row, or
    replacing it under ``combine: replace``; a node the table does not name keeps
    that learned row untouched, exactly as before external vectors existed.

    The vectors are frozen **non-persistent** buffers: only the adapters learn, so
    the checkpoint carries a few thousand parameters rather than a copy of a
    200 MB table. ``model_config["node_embeddings"]`` records where to read them
    from, and :func:`load_encoder` rebuilds this module from that.
    """

    def __init__(self, table: NodeEmbeddingTable, hidden_dim: int):
        super().__init__()
        self.combine = table.combine
        self.init_std = table.init_std
        self.group_names = tuple(group.name for group in table.groups)
        self.dims = {group.name: group.dim for group in table.groups}
        # The input scale of each group, so the init below is independent of how
        # `normalize` happened to scale the vectors.
        self.input_rms = {group.name: group.rms for group in table.groups}
        self.num_covered = table.num_covered
        self.adapters = nn.ModuleDict()
        for group in table.groups:
            self.adapters[group.name] = nn.Linear(group.dim, hidden_dim, bias=table.bias)
            # float32 [m,D] and int64 [m]; not persisted, see the class docstring.
            self.register_buffer(
                f"vectors_{group.name}",
                torch.from_numpy(group.vectors.astype("float32", copy=False)),
                persistent=False,
            )
            self.register_buffer(
                f"nodes_{group.name}", torch.from_numpy(group.nodes), persistent=False
            )

    def reset_parameters(self) -> None:
        """Start every adapted row at ``init_std`` per coordinate, whatever its width.

        ``nn.Linear``'s default init is calibrated for a generic input, not for
        rows that sit beside ``nn.Embedding`` weights of std ``0.1`` and are then
        multiplied together three at a time by the DistMult head — it starts the
        pre-training loss an order of magnitude too high. Dividing by the group's
        own input RMS makes this independent of ``normalize``, so ``l2`` and
        ``standardize`` differ in what they *keep*, not in how loud they start.
        """
        for name, adapter in self.adapters.items():
            rms = max(self.input_rms[name], 1e-12)
            nn.init.normal_(
                adapter.weight, std=self.init_std / (rms * math.sqrt(adapter.in_features))
            )
            if adapter.bias is not None:
                nn.init.zeros_(adapter.bias)

    def forward(self, base: Tensor) -> Tensor:
        """Fold the adapted vectors into the covered rows of a ``[N, H]`` matrix.

        ``replace`` overwrites those rows, ``add`` leaves the learned row in place
        and adds the adapted vector to it. Both are out of place, so ``base`` — the
        learned embedding table — still receives a gradient (zero on the replaced
        rows under ``replace``). That keeps ``nn.Embedding.weight`` a used
        parameter, which DDP requires.
        """
        out = base
        for name in self.group_names:
            nodes = self.get_buffer(f"nodes_{name}")
            # [m,D] -> [m,H]
            adapted = self.adapters[name](self.get_buffer(f"vectors_{name}").to(base.dtype))
            out = (
                out.index_add(0, nodes, adapted)
                if self.combine == "add"
                else out.index_copy(0, nodes, adapted)
            )
        return out


class RelationalGIN(nn.Module):
    """Relation-wise GIN encoder compatible with the original SLGCN GraphNet."""

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        external: NodeEmbeddingTable | None = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        self.convs = nn.ModuleList()
        self.projections = nn.ModuleList()
        for _ in range(num_layers):
            relation_convs, relation_projections = nn.ModuleList(), nn.ModuleList()
            for _ in range(num_relations):
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                relation_convs.append(GINConv(mlp, train_eps=True))
                relation_projections.append(nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(relation_convs)
            self.projections.append(relation_projections)
        self.readout = nn.Linear(hidden_dim * num_layers, hidden_dim)
        self.external: ExternalNodeEmbedding | None = None
        self.external_spec: dict[str, Any] | None = None
        self.reset_parameters()
        # Built *after* the reset, and only when configured: a run without external
        # node embeddings therefore draws exactly the random numbers it drew before
        # they existed, and a run with them starts from the same encoder weights as
        # the run without — only the adapters are extra.
        if external is not None:
            self.external = ExternalNodeEmbedding(external, hidden_dim)
            self.external.reset_parameters()
            self.external_spec = external.spec

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.readout.reset_parameters()
        for layer in self.convs:
            for conv in layer:
                conv.reset_parameters()
        for layer in self.projections:
            for projection in layer:
                projection.reset_parameters()
        if self.external is not None:
            self.external.reset_parameters()

    def node_embedding_matrix(self) -> Tensor:
        """The encoder's input representation: float32 ``[N, H]``, one row per node.

        Without external vectors this *is* the learned embedding table. With them,
        the covered nodes carry their adapted vector and the rest keep their
        learned row — which is why every consumer (the forward pass, partition
        mode, Integrated Gradients) goes through this instead of ``embedding.weight``.
        """
        weight = self.embedding.weight
        return weight if self.external is None else self.external(weight)

    def embed_nodes(self, nodes: Tensor) -> Tensor:
        """The ``[n, H]`` rows of :meth:`node_embedding_matrix` for one node subset.

        Args:
            nodes: int64 ``[n]`` — graph node ids, e.g. one partition batch.
        """
        if self.external is None:
            return self.embedding(nodes)
        return self.node_embedding_matrix().index_select(0, nodes)

    def forward_from_embedding(
        self, x: Tensor, edge_index: Tensor, edge_type: Tensor
    ) -> Tensor:
        """Convolve a given embedding matrix over the graph.

        Args:
            x: float32 ``[N, H]`` — one row per node of the graph being convolved.
                Integrated Gradients passes a scaled copy, and partition mode passes
                the gathered ``[n, H]`` rows of one batch, so this is not always the
                embedding table itself.
            edge_index: int64 ``[2, E]`` — row 0 source, row 1 destination, indexing
                into ``x``'s rows (so *local* indices in partition mode).
            edge_type: int64 ``[E]`` — values in ``[0, R)``, aligned with the columns
                of ``edge_index``.

        Returns:
            float32 ``[N, H]`` (``[n, H]`` in partition mode) — the readout over all
            ``L`` layers concatenated, one row per input row of ``x``.
        """
        outputs = []
        for relation_convs, relation_projections in zip(self.convs, self.projections):
            relation_outputs = []
            for relation, (conv, projection) in enumerate(
                zip(relation_convs, relation_projections)
            ):
                mask = edge_type == relation  # bool [E]
                relation_outputs.append(
                    # conv: [N,H] -> [N,H] over this relation's edges only
                    torch.nn.functional.elu(projection(conv(x, edge_index[:, mask])))
                )
            # stack: [R,N,H] -> sum over relations -> [N,H]. This is the tensor whose
            # size drives peak memory, hence the partition mode.
            x = self.dropout(torch.stack(relation_outputs).sum(dim=0))
            outputs.append(x)
        # cat: L x [N,H] -> [N, L*H] -> readout -> [N,H]
        return self.readout(torch.cat(outputs, dim=-1))

    def forward(self, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        """Convolve the node embedding matrix: ``[2,E]``/``[E]`` in, ``[N,H]`` out."""
        return self.forward_from_embedding(self.node_embedding_matrix(), edge_index, edge_type)


class GraphPretrainer(nn.Module):
    def __init__(self, encoder: RelationalGIN):
        super().__init__()
        self.encoder = encoder
        self.relation = nn.Embedding(encoder.num_relations, encoder.hidden_dim)
        nn.init.normal_(self.relation.weight, std=0.1)

    def score(self, node_embeddings: Tensor, edges: Tensor, edge_types: Tensor) -> Tensor:
        """DistMult score per edge.

        Args:
            node_embeddings: float32 ``[N, H]`` (``[n, H]`` in partition mode).
            edges: int64 ``[2, K]`` — indices into ``node_embeddings``'s rows.
            edge_types: int64 ``[K]`` — values in ``[0, R)``.

        Returns:
            float32 ``[K]`` — one logit per edge.
        """
        src, dst = edges  # int64 [K] each
        return (
            # [K,H] * [K,H] * [K,H] -> sum over H -> [K]
            node_embeddings[src] * self.relation(edge_types) * node_embeddings[dst]
        ).sum(dim=-1)

    def forward(
        self,
        graph_edge_index: Tensor,
        graph_edge_type: Tensor,
        positive_edges: Tensor,
        positive_types: Tensor,
        negative_edges: Tensor,
        nodes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Score positive against corrupted edges over the graph it is given.

        ``nodes`` switches from full-graph to *partition* mode: the edges are then
        indexed locally within that node set, and only its rows of the embedding
        table are gathered, which is what bounds a step's activation memory to the
        subgraph (see :mod:`pathwaygnn.data.partition`). Every parameter still
        takes part in the step, so DDP needs no unused-parameter search.

        Args:
            graph_edge_index: int64 ``[2, E]`` — the graph to convolve.
            graph_edge_type: int64 ``[E]``.
            positive_edges: int64 ``[2, K]`` — the edges to score as present.
            positive_types: int64 ``[K]`` — used for **both** score calls, since a
                corrupted edge keeps its relation.
            negative_edges: int64 ``[2, K]`` — ``positive_edges`` with row 1 replaced.
            nodes: int64 ``[n]`` or ``None`` — original node ids of a partition batch.
                Given, every index above is local to it and ``E``/``N`` become the
                subgraph's; ``None`` means the whole graph.

        Returns:
            two float32 ``[K]`` tensors: the positive and the negative logits.
        """
        embeddings = (
            self.encoder(graph_edge_index, graph_edge_type)
            if nodes is None
            else self.encoder.forward_from_embedding(
                self.encoder.embed_nodes(nodes), graph_edge_index, graph_edge_type
            )
        )
        return (
            self.score(embeddings, positive_edges, positive_types),
            self.score(embeddings, negative_edges, positive_types),
        )


def encoder_config(encoder: RelationalGIN, dropout: float) -> dict[str, Any]:
    """The metadata every checkpoint carries so consumers can rebuild the encoder."""
    config = {
        "num_nodes": encoder.num_nodes,
        "num_relations": encoder.num_relations,
        "hidden_dim": encoder.hidden_dim,
        "num_layers": encoder.num_layers,
        "dropout": float(dropout),
    }
    if encoder.external_spec is not None:
        # Only written when external vectors are in use, so a checkpoint from a
        # plain run keeps exactly the keys it always had.
        config["node_embeddings"] = encoder.external_spec
    return config


def load_encoder(
    checkpoint_path: str | Path,
    num_nodes: int,
    num_relations: int,
    device: torch.device | str = "cpu",
    node_names: Sequence[str] | None = None,
    node_embeddings: Any = None,
) -> tuple[RelationalGIN, dict[str, Any]]:
    """Rebuild a pre-trained encoder and check it against the dataset's graph.

    ``checkpoint_path`` may be a local path or an ``hf://`` reference, so every
    command that takes a ``pretrained_checkpoint`` accepts a published encoder.

    Args:
        node_names: the dataset's ``nodes.json``. Required only when the checkpoint
            was pre-trained with external node vectors, which are matched by name.
        node_embeddings: an optional ``model.node_embeddings:`` block overriding the
            one recorded in the checkpoint — the way to point at the same table
            after it has moved. The adapter shapes still have to agree.
    """
    from pathwaygnn.hub import resolve_checkpoint

    checkpoint_path = resolve_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    if int(config["num_nodes"]) != num_nodes or int(config["num_relations"]) != num_relations:
        raise ValueError(
            f"{checkpoint_path} was pre-trained on a graph with {config['num_nodes']} nodes and "
            f"{config['num_relations']} relations, but the dataset has {num_nodes} and "
            f"{num_relations}; re-run pre-training on this dataset"
        )
    spec = config.get("node_embeddings")
    if spec is None and node_embeddings:
        raise ValueError(
            f"{checkpoint_path} was pre-trained without external node embeddings, so its "
            "encoder has no adapters to feed; re-run pre-training with "
            "`model.node_embeddings:` to use them"
        )
    external = None
    if spec is not None:
        if node_names is None:
            raise ValueError(
                f"{checkpoint_path} was pre-trained with external node embeddings "
                f"({spec.get('path')}), which are matched by node name; this command must "
                "pass the dataset's nodes.json"
            )
        external = load_node_embeddings(node_embeddings or spec, node_names)
        check_spec(external, spec)  # type: ignore[arg-type]
    encoder = RelationalGIN(
        num_nodes,
        num_relations,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config.get("dropout", 0.0)),
        external=external,
    )
    encoder.load_state_dict(checkpoint["encoder"])
    return encoder.to(device), checkpoint

