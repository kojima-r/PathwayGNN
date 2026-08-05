"""Sample-level batching over a prepared task, independent of the dataset.

This is the other half of the head's tensor contract; the shape symbols are the ones
:mod:`pathwaygnn.models.predictor` documents: ``B`` samples per batch, ``G`` genes of
a dense node-level feature, ``V`` stored values of a sparse one across the batch,
``S`` sample-level features.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from pathwaygnn.data.format import Task


@dataclass
class NodeFeatureBatch:
    """One node-level feature of a batch.

    The two kinds carry the same information in different layouts, because a dense
    table gives every sample the same gene set while a sparse one does not:

    ================  ==========================  =============================
    field             ``kind == "dense"``          ``kind == "sparse"``
    ================  ==========================  =============================
    ``value``         float32 ``[B, G]``           float32 ``[V, 1]``
    ``gene``          int64 ``[G]``, shared        int64 ``[V]``, per value
    ``sample``        ``None``                     int64 ``[V]``, values in ``[0,B)``
    ================  ==========================  =============================

    ``gene`` holds **graph node ids**, so it indexes the encoder's embedding rows
    directly.
    """

    kind: str
    gene: Tensor
    value: Tensor
    sample: Tensor | None = None

    @property
    def dense(self) -> bool:
        return self.kind == "dense"

    def to(self, device: torch.device | str) -> "NodeFeatureBatch":
        return NodeFeatureBatch(
            self.kind,
            self.gene.to(device),
            self.value.to(device),
            None if self.sample is None else self.sample.to(device),
        )


@dataclass
class SampleBatch:
    """One batch as the head consumes it.

    Fields:
        node_features: alias -> :class:`NodeFeatureBatch`. **Insertion order is the
            head's concat order**, which is why `pred` checks the alias list.
        label: float32 ``[B]`` — 0.0 / 1.0.
        index: int64 ``[B]`` — the sample's row in the task, kept on the CPU so
            predictions can be written back against it.
        sample_feature: float32 ``[B, S]`` or ``None``.
        group: int64 ``[B]`` or ``None`` — group code per sample.
    """

    node_features: dict[str, NodeFeatureBatch]
    label: Tensor
    index: Tensor
    sample_feature: Tensor | None = None
    group: Tensor | None = None

    @property
    def size(self) -> int:
        return int(self.label.numel())

    def to(self, device: torch.device | str) -> "SampleBatch":
        return SampleBatch(
            {name: node_feature.to(device) for name, node_feature in self.node_features.items()},
            self.label.to(device),
            self.index,
            None if self.sample_feature is None else self.sample_feature.to(device),
            None if self.group is None else self.group.to(device),
        )


class Collate:
    """Picklable collate function; holds only the feature kinds and gene indices."""

    def __init__(self, kinds: dict[str, str], gene_index: dict[str, Tensor]):
        self.kinds = kinds
        self.gene_index = gene_index

    def __call__(self, rows: list[dict[str, Any]]) -> SampleBatch:
        """``B`` rows from :meth:`TaskDataset.__getitem__` -> one :class:`SampleBatch`."""
        node_features: dict[str, NodeFeatureBatch] = {}
        for name, kind in self.kinds.items():
            if kind == "dense":
                # B x [G] -> [B,G]
                value = torch.stack([row["node_features"][name] for row in rows])
                node_features[name] = NodeFeatureBatch("dense", self.gene_index[name], value)
                continue
            genes = [row["node_features"][name][0] for row in rows]   # B x [v_i]
            values = [row["node_features"][name][1] for row in rows]  # B x [v_i]
            counts = torch.tensor([item.numel() for item in genes], dtype=torch.long)  # [B]
            node_features[name] = NodeFeatureBatch(
                "sparse",
                torch.cat(genes),                      # [V]
                torch.cat(values).reshape(-1, 1),      # [V,1]
                # [V]: sample 0 repeated v_0 times, then sample 1, ...
                torch.repeat_interleave(torch.arange(len(rows), dtype=torch.long), counts),
            )
        sample_feature = (
            torch.stack([row["sample_feature"] for row in rows])
            if rows and rows[0]["sample_feature"] is not None
            else None
        )
        group = (
            torch.tensor([row["group"] for row in rows], dtype=torch.long)
            if rows and rows[0]["group"] is not None
            else None
        )
        return SampleBatch(
            node_features=node_features,
            label=torch.stack([row["label"] for row in rows]),
            index=torch.tensor([row["index"] for row in rows], dtype=torch.long),
            sample_feature=sample_feature,
            group=group,
        )


class TaskDataset(Dataset):
    """Rows of one prepared task, addressed through an optional index subset."""

    def __init__(self, task: Task, indices: Sequence[int] | np.ndarray | None = None):
        self.task = task
        self.node_features = task.node_features
        self._tables: dict[str, Any] = {}
        self._gene_index: dict[str, Tensor] = {}
        for node_feature in task.node_features:
            if node_feature.dense:
                self._tables[node_feature.name] = node_feature.matrix()
                self._gene_index[node_feature.name] = torch.from_numpy(
                    node_feature.gene_index().astype(np.int64, copy=True)
                )
            else:
                self._tables[node_feature.name] = node_feature.csr()
        self._rows = {node_feature.name: task.rows(node_feature.name) for node_feature in task.node_features}
        self._labels = task.labels()
        self._groups = task.groups()
        self._sample_features = task.sample_features()
        self.indices = (
            np.arange(self._labels.size, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

    def __len__(self) -> int:
        return int(self.indices.size)

    @property
    def targets(self) -> np.ndarray:
        """float32 ``[len(self)]`` — the labels of this subset, in its own order."""
        return np.asarray(self._labels[self.indices])

    @property
    def gene_index(self) -> dict[str, Tensor]:
        """alias -> int64 ``[G]`` graph node ids, for the dense features only."""
        return self._gene_index

    def collate(self) -> Collate:
        return Collate(
            {node_feature.name: node_feature.kind for node_feature in self.node_features}, self._gene_index
        )

    def subset(self, indices: Sequence[int] | np.ndarray) -> "TaskDataset":
        clone = copy.copy(self)
        clone.indices = self.indices[np.asarray(indices, dtype=np.int64)]
        return clone

    def row(self, sample: int, node_feature: str) -> int:
        rows = self._rows[node_feature]
        return sample if rows is None else int(rows[sample])

    def __getitem__(self, item: int) -> dict[str, Any]:
        """One sample: dense features as float32 ``[G]``, sparse as (int64 ``[v]``,
        float32 ``[v]``) pairs, plus its label, row index, ``[S]`` sample features and
        group code. :class:`Collate` assembles ``B`` of these into batch tensors."""
        index = int(self.indices[item])
        node_features: dict[str, Any] = {}
        for node_feature in self.node_features:
            row = self.row(index, node_feature.name)
            if node_feature.dense:
                node_features[node_feature.name] = torch.from_numpy(
                    np.array(self._tables[node_feature.name][row], dtype=np.float32, copy=True)
                )
            else:
                ptr, gene, value = self._tables[node_feature.name]
                start, stop = int(ptr[row]), int(ptr[row + 1])
                node_features[node_feature.name] = (
                    torch.from_numpy(np.array(gene[start:stop], dtype=np.int64, copy=True)),
                    torch.from_numpy(np.array(value[start:stop], dtype=np.float32, copy=True)),
                )
        return {
            "node_features": node_features,
            "label": torch.tensor(float(self._labels[index]), dtype=torch.float32),
            "index": index,
            "sample_feature": (
                None
                if self._sample_features is None
                else torch.from_numpy(
                    np.array(self._sample_features[index], dtype=np.float32, copy=True)
                )
            ),
            "group": None if self._groups is None else int(self._groups[index]),
        }
