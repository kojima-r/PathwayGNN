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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.nn import GINConv


class RelationalGIN(nn.Module):
    """Relation-wise GIN encoder compatible with the original SLGCN GraphNet."""

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.readout.reset_parameters()
        for layer in self.convs:
            for conv in layer:
                conv.reset_parameters()
        for layer in self.projections:
            for projection in layer:
                projection.reset_parameters()

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
        """Convolve the learned embedding table: ``[2,E]``/``[E]`` in, ``[N,H]`` out."""
        return self.forward_from_embedding(self.embedding.weight, edge_index, edge_type)


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
                self.encoder.embedding(nodes), graph_edge_index, graph_edge_type
            )
        )
        return (
            self.score(embeddings, positive_edges, positive_types),
            self.score(embeddings, negative_edges, positive_types),
        )


def encoder_config(encoder: RelationalGIN, dropout: float) -> dict[str, Any]:
    """The metadata every checkpoint carries so consumers can rebuild the encoder."""
    return {
        "num_nodes": encoder.num_nodes,
        "num_relations": encoder.num_relations,
        "hidden_dim": encoder.hidden_dim,
        "num_layers": encoder.num_layers,
        "dropout": float(dropout),
    }


def load_encoder(
    checkpoint_path: str | Path,
    num_nodes: int,
    num_relations: int,
    device: torch.device | str = "cpu",
) -> tuple[RelationalGIN, dict[str, Any]]:
    """Rebuild a pre-trained encoder and check it against the dataset's graph.

    ``checkpoint_path`` may be a local path or an ``hf://`` reference, so every
    command that takes a ``pretrained_checkpoint`` accepts a published encoder.
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
    encoder = RelationalGIN(
        num_nodes,
        num_relations,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config.get("dropout", 0.0)),
    )
    encoder.load_state_dict(checkpoint["encoder"])
    return encoder.to(device), checkpoint

