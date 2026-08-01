"""Sample-level batching over a prepared task, independent of the dataset."""

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
class ChannelBatch:
    """One channel of a batch.

    ``dense``: ``value`` is ``[batch, genes]`` and ``gene`` the shared ``[genes]``
    node indices. ``sparse``: ``value`` is ``[values, 1]``, ``gene`` the matching
    ``[values]`` node indices and ``sample`` the ``[values]`` row in the batch.
    """

    kind: str
    gene: Tensor
    value: Tensor
    sample: Tensor | None = None

    @property
    def dense(self) -> bool:
        return self.kind == "dense"

    def to(self, device: torch.device | str) -> "ChannelBatch":
        return ChannelBatch(
            self.kind,
            self.gene.to(device),
            self.value.to(device),
            None if self.sample is None else self.sample.to(device),
        )


@dataclass
class SampleBatch:
    channels: dict[str, ChannelBatch]
    label: Tensor
    index: Tensor
    covariate: Tensor | None = None
    group: Tensor | None = None

    @property
    def size(self) -> int:
        return int(self.label.numel())

    def to(self, device: torch.device | str) -> "SampleBatch":
        return SampleBatch(
            {name: channel.to(device) for name, channel in self.channels.items()},
            self.label.to(device),
            self.index,
            None if self.covariate is None else self.covariate.to(device),
            None if self.group is None else self.group.to(device),
        )


class Collate:
    """Picklable collate function; holds only the channel kinds and gene indices."""

    def __init__(self, kinds: dict[str, str], gene_index: dict[str, Tensor]):
        self.kinds = kinds
        self.gene_index = gene_index

    def __call__(self, rows: list[dict[str, Any]]) -> SampleBatch:
        channels: dict[str, ChannelBatch] = {}
        for name, kind in self.kinds.items():
            if kind == "dense":
                value = torch.stack([row["channels"][name] for row in rows])
                channels[name] = ChannelBatch("dense", self.gene_index[name], value)
                continue
            genes = [row["channels"][name][0] for row in rows]
            values = [row["channels"][name][1] for row in rows]
            counts = torch.tensor([item.numel() for item in genes], dtype=torch.long)
            channels[name] = ChannelBatch(
                "sparse",
                torch.cat(genes),
                torch.cat(values).reshape(-1, 1),
                torch.repeat_interleave(torch.arange(len(rows), dtype=torch.long), counts),
            )
        covariate = (
            torch.stack([row["covariate"] for row in rows])
            if rows and rows[0]["covariate"] is not None
            else None
        )
        group = (
            torch.tensor([row["group"] for row in rows], dtype=torch.long)
            if rows and rows[0]["group"] is not None
            else None
        )
        return SampleBatch(
            channels=channels,
            label=torch.stack([row["label"] for row in rows]),
            index=torch.tensor([row["index"] for row in rows], dtype=torch.long),
            covariate=covariate,
            group=group,
        )


class TaskDataset(Dataset):
    """Rows of one prepared task, addressed through an optional index subset."""

    def __init__(self, task: Task, indices: Sequence[int] | np.ndarray | None = None):
        self.task = task
        self.channels = task.channels
        self._tables: dict[str, Any] = {}
        self._gene_index: dict[str, Tensor] = {}
        for channel in task.channels:
            if channel.dense:
                self._tables[channel.name] = channel.matrix()
                self._gene_index[channel.name] = torch.from_numpy(
                    channel.gene_index().astype(np.int64, copy=True)
                )
            else:
                self._tables[channel.name] = channel.csr()
        self._rows = {channel.name: task.rows(channel.name) for channel in task.channels}
        self._labels = task.labels()
        self._groups = task.groups()
        self._covariates = task.covariates()
        self.indices = (
            np.arange(self._labels.size, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

    def __len__(self) -> int:
        return int(self.indices.size)

    @property
    def targets(self) -> np.ndarray:
        return np.asarray(self._labels[self.indices])

    @property
    def gene_index(self) -> dict[str, Tensor]:
        return self._gene_index

    def collate(self) -> Collate:
        return Collate(
            {channel.name: channel.kind for channel in self.channels}, self._gene_index
        )

    def subset(self, indices: Sequence[int] | np.ndarray) -> "TaskDataset":
        clone = copy.copy(self)
        clone.indices = self.indices[np.asarray(indices, dtype=np.int64)]
        return clone

    def row(self, sample: int, channel: str) -> int:
        rows = self._rows[channel]
        return sample if rows is None else int(rows[sample])

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        channels: dict[str, Any] = {}
        for channel in self.channels:
            row = self.row(index, channel.name)
            if channel.dense:
                channels[channel.name] = torch.from_numpy(
                    np.array(self._tables[channel.name][row], dtype=np.float32, copy=True)
                )
            else:
                ptr, gene, value = self._tables[channel.name]
                start, stop = int(ptr[row]), int(ptr[row + 1])
                channels[channel.name] = (
                    torch.from_numpy(np.array(gene[start:stop], dtype=np.int64, copy=True)),
                    torch.from_numpy(np.array(value[start:stop], dtype=np.float32, copy=True)),
                )
        return {
            "channels": channels,
            "label": torch.tensor(float(self._labels[index]), dtype=torch.float32),
            "index": index,
            "covariate": (
                None
                if self._covariates is None
                else torch.from_numpy(
                    np.array(self._covariates[index], dtype=np.float32, copy=True)
                )
            ),
            "group": None if self._groups is None else int(self._groups[index]),
        }
