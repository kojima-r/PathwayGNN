"""Sample-level head that turns channels of gene values into one logit.

The head is dataset-agnostic: it aggregates every channel of a
:class:`~pathwaygnn.data.samples.SampleBatch` over genes, concatenates the
per-channel summaries with the optional covariate branch, and scores the result.
``block="paper"`` reproduces the block used by the cancer-survival architecture
(Linear-ELU-[BN]-Linear-ELU-[BN]); ``block="plain"`` reproduces the
target-repositioning block (Linear-ELU-[Dropout]-Linear).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn

from pathwaygnn.data.samples import SampleBatch

BLOCKS = ("plain", "paper")


def _scatter_sum(values: Tensor, index: Tensor, size: int) -> Tensor:
    output = values.new_zeros((size, values.size(-1)))
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
        channels: Sequence[str],
        embedding_dim: int,
        hidden_dim: int | None = None,
        covariate_dim: int = 0,
        use_graph: bool = True,
        use_covariates: bool = False,
        dropout: float = 0.0,
        batch_norm: bool = False,
        block: str = "plain",
    ):
        super().__init__()
        if block not in BLOCKS:
            raise ValueError(f"block must be one of {BLOCKS}, got {block!r}")
        if use_covariates and covariate_dim <= 0:
            raise ValueError("use_covariates requires a task with covariates")
        self.channel_names = tuple(channels)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else embedding_dim)
        self.covariate_dim = int(covariate_dim)
        self.use_graph = bool(use_graph)
        self.use_covariates = bool(use_covariates)
        self.dropout = float(dropout)
        self.batch_norm = bool(batch_norm)
        self.block = block
        activate = block == "paper"
        # Module creation order fixes the parameter initialisation draws; keep it.
        self.value_projection = _block(1, self.embedding_dim, self.dropout, self.batch_norm, activate)
        self.gene_blocks = nn.ModuleList(
            _block(self.embedding_dim, self.hidden_dim, self.dropout, self.batch_norm, activate)
            for _ in self.channel_names
        )
        self.aggregate_blocks = nn.ModuleList(
            _block(self.hidden_dim, self.hidden_dim, self.dropout, self.batch_norm, activate)
            for _ in self.channel_names
        )
        if self.use_covariates:
            self.covariate_block = _block(
                self.covariate_dim, self.hidden_dim, self.dropout, self.batch_norm, activate
            )
        parts = len(self.channel_names) + (1 if self.use_covariates else 0)
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
            "channels": list(self.channel_names),
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "covariate_dim": self.covariate_dim,
            "use_graph": self.use_graph,
            "use_covariates": self.use_covariates,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "block": self.block,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SampleLevelModel":
        return cls(**config)

    def forward(self, batch: SampleBatch, node_embeddings: Tensor | None = None) -> Tensor:
        if self.use_graph and node_embeddings is None:
            raise ValueError("node_embeddings are required when use_graph=True")
        size = batch.size
        parts = []
        for position, name in enumerate(self.channel_names):
            channel = batch.channels[name]
            if channel.dense:
                genes = channel.value.size(1)
                values = self.value_projection(channel.value.reshape(-1, 1))
                if self.use_graph:
                    values = values + node_embeddings[channel.gene].repeat(size, 1)  # type: ignore[index]
                summary = self.gene_blocks[position](values).reshape(size, genes, -1).sum(dim=1)
            else:
                values = self.value_projection(channel.value)
                if self.use_graph:
                    values = values + node_embeddings[channel.gene]  # type: ignore[index]
                summary = _scatter_sum(
                    self.gene_blocks[position](values), channel.sample, size  # type: ignore[arg-type]
                )
            parts.append(self.aggregate_blocks[position](summary))
        if self.use_covariates:
            if batch.covariate is None:
                raise ValueError("the batch carries no covariates")
            parts.append(self.covariate_block(batch.covariate))
        return self.output(torch.cat(parts, dim=-1)).squeeze(-1)


def build_model(
    task_channels: Sequence[str],
    covariate_dim: int,
    embedding_dim: int,
    model_cfg: dict[str, Any],
    use_graph: bool,
    use_covariates: bool,
) -> SampleLevelModel:
    """Instantiate the head from a config ``model:`` block and a variant."""
    return SampleLevelModel(
        channels=task_channels,
        embedding_dim=embedding_dim,
        hidden_dim=model_cfg.get("hidden_dim"),
        covariate_dim=covariate_dim,
        use_graph=use_graph,
        use_covariates=use_covariates,
        dropout=float(model_cfg.get("dropout", 0.0)),
        batch_norm=bool(model_cfg.get("batch_norm", False)),
        block=str(model_cfg.get("block", "plain")),
    )
