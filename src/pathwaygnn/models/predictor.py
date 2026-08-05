"""Sample-level head that turns node-level features of gene values into one logit.

The head is dataset-agnostic: it aggregates every node_feature of a
:class:`~pathwaygnn.data.samples.SampleBatch` over genes, concatenates the
per-node_feature summaries with the optional sample_feature branch, and scores the result.
``block="paper"`` reproduces the block used by the cancer-survival architecture
(Linear-ELU-[BN]-Linear-ELU-[BN]); ``block="plain"`` reproduces the
target-repositioning block (Linear-ELU-[Dropout]-Linear).

Tensor shapes below use these symbols:

``B``
    samples in the batch (``batch.size``)
``G``
    genes of a *dense* node-level feature — the same set for every sample, so its
    values form a rectangle
``V``
    stored values of a *sparse* node-level feature across the whole batch — variable
    per batch, because each sample stores only its own non-zero genes
``N``
    graph nodes, i.e. the row count of the encoder's embedding matrix
``E``
    ``embedding_dim`` (the width the encoder's embedding is added at)
``H``
    ``hidden_dim`` (the width of every block's output)
``S``
    ``sample_feature_dim``
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn

from pathwaygnn.data.samples import SampleBatch

BLOCKS = ("plain", "paper")


def _scatter_sum(values: Tensor, index: Tensor, size: int) -> Tensor:
    """Sum rows of ``values`` into the sample they belong to.

    The sparse counterpart of a ``reshape(B, G, -1).sum(1)``: those two are
    mathematically equal but accumulate in a different order, which is why each kind
    keeps its own path (see CLAUDE.md).

    Args:
        values: float32 ``[V, H]`` — one row per stored value in the batch.
        index: int64 ``[V]`` — values in ``[0, size)``, the sample each row belongs to.
        size: ``B``.

    Returns:
        float32 ``[B, H]``.
    """
    output = values.new_zeros((size, values.size(-1)))  # [B,H]
    output.index_add_(0, index, values)
    return output


def _block(
    in_dim: int, out_dim: int, dropout: float, batch_norm: bool, activate_output: bool
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim), nn.ELU()]
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_dim))
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(out_dim, out_dim))
    if activate_output:
        layers.append(nn.ELU())
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
    return nn.Sequential(*layers)


class SampleLevelModel(nn.Module):
    def __init__(
        self,
        node_features: Sequence[str],
        embedding_dim: int,
        hidden_dim: int | None = None,
        sample_feature_dim: int = 0,
        use_graph: bool = True,
        use_sample_features: bool = False,
        dropout: float = 0.0,
        batch_norm: bool = False,
        block: str = "plain",
    ):
        super().__init__()
        if block not in BLOCKS:
            raise ValueError(f"block must be one of {BLOCKS}, got {block!r}")
        if use_sample_features and sample_feature_dim <= 0:
            raise ValueError("use_sample_features requires a task that has sample-level features")
        self.node_feature_names = tuple(node_features)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else embedding_dim)
        self.sample_feature_dim = int(sample_feature_dim)
        self.use_graph = bool(use_graph)
        self.use_sample_features = bool(use_sample_features)
        self.dropout = float(dropout)
        self.batch_norm = bool(batch_norm)
        self.block = block
        activate = block == "paper"
        # Module creation order fixes the parameter initialisation draws; keep it.
        self.value_projection = _block(1, self.embedding_dim, self.dropout, self.batch_norm, activate)
        self.gene_blocks = nn.ModuleList(
            _block(self.embedding_dim, self.hidden_dim, self.dropout, self.batch_norm, activate)
            for _ in self.node_feature_names
        )
        self.aggregate_blocks = nn.ModuleList(
            _block(self.hidden_dim, self.hidden_dim, self.dropout, self.batch_norm, activate)
            for _ in self.node_feature_names
        )
        if self.use_sample_features:
            self.sample_feature_block = _block(
                self.sample_feature_dim, self.hidden_dim, self.dropout, self.batch_norm, activate
            )
        parts = len(self.node_feature_names) + (1 if self.use_sample_features else 0)
        output: list[nn.Module] = [nn.Linear(self.hidden_dim * parts, self.hidden_dim), nn.ELU()]
        if self.batch_norm:
            output.append(nn.BatchNorm1d(self.hidden_dim))
        if self.dropout:
            output.append(nn.Dropout(self.dropout))
        output.append(nn.Linear(self.hidden_dim, 1))
        self.output = nn.Sequential(*output)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "node_features": list(self.node_feature_names),
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "sample_feature_dim": self.sample_feature_dim,
            "use_graph": self.use_graph,
            "use_sample_features": self.use_sample_features,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "block": self.block,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SampleLevelModel":
        return cls(**config)

    def forward(self, batch: SampleBatch, node_embeddings: Tensor | None = None) -> Tensor:
        """One logit per sample.

        Args:
            batch: the sample-level batch; see
                :class:`~pathwaygnn.data.samples.SampleBatch` for its own shapes.
            node_embeddings: float32 ``[N, E]`` or ``None`` — the encoder's output,
                indexed by *graph node id*, so it must cover every gene the batch
                references. Required when ``use_graph``; ignored otherwise.

        Returns:
            float32 ``[B]`` — logits, not probabilities.
        """
        if self.use_graph and node_embeddings is None:
            raise ValueError("node_embeddings are required when use_graph=True")
        size = batch.size  # B
        parts = []  # each [B,H]
        for position, name in enumerate(self.node_feature_names):
            node_feature = batch.node_features[name]
            if node_feature.dense:
                genes = node_feature.value.size(1)  # G
                # value [B,G] -> [B*G,1] -> project each scalar -> [B*G,E]
                values = self.value_projection(node_feature.value.reshape(-1, 1))
                if self.use_graph:
                    # gene [G] -> embeddings [G,E] -> tiled per sample [B*G,E]. This
                    # addition is what gives the head gene *identity*.
                    values = values + node_embeddings[node_feature.gene].repeat(size, 1)  # type: ignore[index]
                # [B*G,E] -> block -> [B*G,H] -> [B,G,H] -> sum over genes -> [B,H]
                summary = self.gene_blocks[position](values).reshape(size, genes, -1).sum(dim=1)
            else:
                # value [V,1] -> [V,E]; one row per stored value, not per gene
                values = self.value_projection(node_feature.value)
                if self.use_graph:
                    values = values + node_embeddings[node_feature.gene]  # [V,E]
                # [V,E] -> block -> [V,H] -> scatter into samples -> [B,H]
                summary = _scatter_sum(
                    self.gene_blocks[position](values), node_feature.sample, size  # type: ignore[arg-type]
                )
            parts.append(self.aggregate_blocks[position](summary))  # [B,H]
        if self.use_sample_features:
            if batch.sample_feature is None:
                raise ValueError("the batch carries no sample-level features")
            parts.append(self.sample_feature_block(batch.sample_feature))  # [B,S] -> [B,H]
        # cat: parts x [B,H] -> [B, parts*H] -> [B,1] -> [B]
        return self.output(torch.cat(parts, dim=-1)).squeeze(-1)


def build_model(
    task_node_features: Sequence[str],
    sample_feature_dim: int,
    embedding_dim: int,
    model_cfg: dict[str, Any],
    use_graph: bool,
    use_sample_features: bool,
) -> SampleLevelModel:
    """Instantiate the head from a config ``model:`` block and a variant."""
    return SampleLevelModel(
        node_features=task_node_features,
        embedding_dim=embedding_dim,
        hidden_dim=model_cfg.get("hidden_dim"),
        sample_feature_dim=sample_feature_dim,
        use_graph=use_graph,
        use_sample_features=use_sample_features,
        dropout=float(model_cfg.get("dropout", 0.0)),
        batch_norm=bool(model_cfg.get("batch_norm", False)),
        block=str(model_cfg.get("block", "plain")),
    )
