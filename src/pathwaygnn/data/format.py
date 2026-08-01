"""The single dataset format that every ``pathwaygnn`` command consumes.

Dataset-specific preprocessing lives outside this package (see
``pathwaygnn_datasets``). Preprocessing must finish before training starts and
must leave behind exactly this layout::

    <root>/dataset.json                 manifest; ``name`` identifies the dataset
    <root>/graph.pt                     {"edge_index": int64[2,E], "edge_type": int64[E]}
    <root>/nodes.json                   node index -> name (HGNC symbol or ID)
    <root>/relations.json               relation index -> name
    <root>/channels/<channel>/          a gene-value table, shared by tasks
        channel.json                    {"kind": "sparse"|"dense", ...}
        ptr.npy gene.npy value.npy      kind == sparse: CSR rows over node indices
        matrix.npy gene_index.npy       kind == dense: memmapped [rows, genes]
    <root>/tasks/<task>/
        task.json                       manifest: channel aliases, groups, covariates
        labels.npy                      float32[samples], binary
        groups.npy                      int64[samples]        optional
        covariates.npy                  float32[samples, dim] optional
        rows/<alias>.npy                int64[samples]        optional, identity if absent

A *channel* is one gene-value view of a sample (a perturbation signature, a
disease signature, an expression profile). Channels are dataset-level tables so
that several tasks can share one table. A *task* is one binary prediction problem
that binds each channel to a local alias (``task.json``'s ``channels`` maps alias
-> channel) and maps its samples to channel rows through ``rows/<alias>.npy``.
Tasks that describe the same problem on different sample sets therefore use the
same aliases, which keeps their model configurations interchangeable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

FORMAT = "pathwaygnn/1"
KINDS = ("sparse", "dense")


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class Channel:
    """A gene-value table; ``dense`` tables are memmapped, sparse ones are CSR.

    ``name`` is the alias the task uses, ``source`` the dataset-level channel.
    """

    name: str
    source: str
    kind: str
    root: Path
    num_rows: int
    num_features: int | None = None

    @property
    def dense(self) -> bool:
        return self.kind == "dense"

    def gene_index(self) -> np.ndarray:
        return np.load(self.root / "gene_index.npy")

    def matrix(self) -> np.ndarray:
        return np.load(self.root / "matrix.npy", mmap_mode="r")

    def csr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(  # type: ignore[return-value]
            np.load(self.root / f"{name}.npy", mmap_mode="r")
            for name in ("ptr", "gene", "value")
        )


@dataclass(frozen=True)
class Task:
    """One binary prediction problem defined over a dataset's channels."""

    name: str
    root: Path
    channels: tuple[Channel, ...]
    num_samples: int
    seed_offset: int
    covariate_names: tuple[str, ...]
    group_names: tuple[str, ...]
    manifest: dict[str, Any]

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

    @property
    def covariate_dim(self) -> int:
        return len(self.covariate_names)

    @property
    def num_groups(self) -> int:
        return len(self.group_names)

    def labels(self) -> np.ndarray:
        return np.load(self.root / "labels.npy")

    def groups(self) -> np.ndarray | None:
        path = self.root / "groups.npy"
        return np.load(path, mmap_mode="r") if path.is_file() else None

    def covariates(self) -> np.ndarray | None:
        path = self.root / "covariates.npy"
        return np.load(path, mmap_mode="r") if path.is_file() else None

    def rows(self, alias: str) -> np.ndarray | None:
        """Sample -> channel row; ``None`` means the identity mapping."""
        path = self.root / "rows" / f"{alias}.npy"
        return np.load(path, mmap_mode="r") if path.is_file() else None


@dataclass(frozen=True)
class GraphDataset:
    """Read-only handle on a prepared dataset directory."""

    root: Path
    name: str
    num_nodes: int
    num_relations: int
    manifest: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def open(cls, root: str | Path, expected_name: str | None = None) -> "GraphDataset":
        root = Path(root)
        manifest_path = root / "dataset.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{root} is not a prepared pathwaygnn dataset ({manifest_path} is missing). "
                "Run the dataset's preprocessing command first (see `pathwaygnn-data --help`)."
            )
        manifest = read_json(manifest_path)
        if manifest.get("format") != FORMAT:
            raise ValueError(
                f"{manifest_path} declares format {manifest.get('format')!r}, expected {FORMAT!r}"
            )
        if expected_name is not None and manifest["name"] != expected_name:
            raise ValueError(
                f"the config selects dataset {expected_name!r} but {root} holds "
                f"{manifest['name']!r}; fix `dataset.name` or `dataset.dir`"
            )
        return cls(
            root=root,
            name=manifest["name"],
            num_nodes=int(manifest["num_nodes"]),
            num_relations=int(manifest["num_relations"]),
            manifest=manifest,
        )

    def graph(self) -> tuple[Tensor, Tensor]:
        graph = torch.load(self.root / "graph.pt", map_location="cpu", weights_only=True)
        return graph["edge_index"], graph["edge_type"]

    def node_names(self) -> list[str]:
        return read_json(self.root / "nodes.json")

    def relation_names(self) -> list[str]:
        return read_json(self.root / "relations.json")

    @property
    def task_names(self) -> list[str]:
        return list(self.manifest["tasks"])

    def channel(self, source: str, alias: str | None = None) -> Channel:
        try:
            entry = self.manifest["channels"][source]
        except KeyError as error:
            raise KeyError(
                f"dataset {self.name!r} has no channel {source!r}; "
                f"available: {sorted(self.manifest['channels'])}"
            ) from error
        return Channel(
            name=alias or source,
            source=source,
            kind=entry["kind"],
            root=self.root / "channels" / source,
            num_rows=int(entry["num_rows"]),
            num_features=entry.get("num_features"),
        )

    def task(self, name: str) -> Task:
        root = self.root / "tasks" / name
        if not (root / "task.json").is_file():
            raise KeyError(f"dataset {self.name!r} has no task {name!r}; available: {self.task_names}")
        manifest = read_json(root / "task.json")
        return Task(
            name=name,
            root=root,
            channels=tuple(
                self.channel(source, alias) for alias, source in manifest["channels"].items()
            ),
            num_samples=int(manifest["num_samples"]),
            seed_offset=int(manifest.get("seed_offset", self.task_names.index(name))),
            covariate_names=tuple(manifest.get("covariate_names", ())),
            group_names=tuple(manifest.get("group_names", ())),
            manifest=manifest,
        )


def open_dataset(cfg: dict[str, Any]) -> GraphDataset:
    """Resolve the ``dataset:`` block of a config into a dataset handle."""
    if "dataset" not in cfg:
        raise KeyError(
            "the config needs a `dataset:` block with `dir:` and `name:` "
            "(use `defaults: [dataset.yaml]` from configs/tr or configs/cancer)"
        )
    section = cfg["dataset"]
    return GraphDataset.open(section["dir"], section.get("name"))


def open_task(cfg: dict[str, Any], task: str | None = None) -> tuple[GraphDataset, Task]:
    dataset = open_dataset(cfg)
    name = task if task is not None else cfg["dataset"]["task"]
    return dataset, dataset.task(name)


@dataclass(frozen=True)
class TaskChannel:
    """A task's binding of one dataset channel: which table, and which rows."""

    source: str
    rows: np.ndarray | None = None


class DatasetWriter:
    """Writer used by preprocessing to emit the format described in this module."""

    def __init__(self, root: str | Path, name: str, source: dict[str, Any] | None = None):
        self.root = Path(root).resolve()
        self.name = name
        self.source = dict(source or {})
        self.channels: dict[str, dict[str, Any]] = {}
        self.tasks: list[str] = []
        self.graph_info: dict[str, int] = {}
        self.root.mkdir(parents=True, exist_ok=True)

    def write_graph(
        self,
        edge_index: Tensor,
        edge_type: Tensor,
        node_names: Sequence[str],
        relation_names: Sequence[str],
    ) -> None:
        if edge_index.size(1) != edge_type.numel():
            raise ValueError("edge_index and edge_type disagree on the number of edges")
        if int(edge_index.max()) >= len(node_names):
            raise ValueError("edge_index references nodes that nodes.json does not name")
        torch.save(
            {"edge_index": edge_index.long(), "edge_type": edge_type.long()}, self.root / "graph.pt"
        )
        write_json(self.root / "nodes.json", [str(item) for item in node_names])
        write_json(self.root / "relations.json", [str(item) for item in relation_names])
        self.graph_info = {
            "num_nodes": len(node_names),
            "num_relations": len(relation_names),
            "num_edges": int(edge_index.size(1)),
        }

    def _channel_root(self, name: str) -> Path:
        root = self.root / "channels" / name
        root.mkdir(parents=True, exist_ok=True)
        return root

    def sparse_channel(
        self, name: str, ptr: Iterable[int], gene: Iterable[int], value: Iterable[float]
    ) -> None:
        root = self._channel_root(name)
        ptr = np.asarray(ptr, dtype=np.int64)
        gene = np.asarray(gene, dtype=np.int64)
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if gene.size != value.size or ptr[-1] != gene.size:
            raise ValueError(f"channel {name!r}: ptr/gene/value are inconsistent")
        for field_name, array in (("ptr", ptr), ("gene", gene), ("value", value)):
            np.save(root / f"{field_name}.npy", array)
        self._register(name, "sparse", int(ptr.size - 1), None, {"num_values": int(gene.size)})

    def dense_channel(self, name: str, num_rows: int, num_features: int) -> Path:
        """Register a dense channel and return its directory.

        The caller writes ``matrix.npy`` (float32 ``[num_rows, num_features]``) and
        ``gene_index.npy`` (int64 ``[num_features]``) itself, so that large
        matrices can be streamed straight into a memmap.
        """
        root = self._channel_root(name)
        self._register(name, "dense", int(num_rows), int(num_features), {})
        return root

    def _register(
        self, name: str, kind: str, num_rows: int, num_features: int | None, extra: dict[str, Any]
    ) -> None:
        entry = {"kind": kind, "num_rows": num_rows, **extra}
        if num_features is not None:
            entry["num_features"] = num_features
        write_json(self.root / "channels" / name / "channel.json", entry)
        self.channels[name] = entry

    def write_task(
        self,
        name: str,
        labels: np.ndarray,
        channels: dict[str, "TaskChannel"] | None = None,
        covariates: np.ndarray | None = None,
        covariate_names: Sequence[str] = (),
        groups: np.ndarray | None = None,
        group_names: Sequence[str] = (),
        seed_offset: int | None = None,
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        root = self.root / "tasks" / name
        root.mkdir(parents=True, exist_ok=True)
        labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        np.save(root / "labels.npy", labels)
        for alias, binding in (channels or {}).items():
            if binding.source not in self.channels:
                raise KeyError(f"task {name!r} references unknown channel {binding.source!r}")
            expected = self.channels[binding.source]["num_rows"]
            if binding.rows is None:
                if expected != labels.size:
                    raise ValueError(
                        f"task {name!r} maps every sample to its own row of channel "
                        f"{binding.source!r}, but the channel has {expected} rows and the task "
                        f"{labels.size} samples"
                    )
                continue
            row = np.asarray(binding.rows, dtype=np.int64).reshape(-1)
            if row.size != labels.size or (row.size and (row.min() < 0 or row.max() >= expected)):
                raise ValueError(f"task {name!r}: rows/{alias}.npy is out of range")
            (root / "rows").mkdir(exist_ok=True)
            np.save(root / "rows" / f"{alias}.npy", row)
        if covariates is not None:
            covariates = np.asarray(covariates, dtype=np.float32)
            if covariates.shape != (labels.size, len(covariate_names)):
                raise ValueError(
                    f"task {name!r}: covariates {covariates.shape} do not match "
                    f"{labels.size} samples x {len(covariate_names)} names"
                )
            np.save(root / "covariates.npy", covariates)
        if groups is not None:
            groups = np.asarray(groups, dtype=np.int64).reshape(-1)
            if groups.size != labels.size:
                raise ValueError(f"task {name!r}: groups.npy does not match the sample count")
            np.save(root / "groups.npy", groups)
        manifest = {
            "name": name,
            "num_samples": int(labels.size),
            "num_positive": int((labels == 1).sum()),
            "seed_offset": len(self.tasks) if seed_offset is None else int(seed_offset),
            "channels": {alias: binding.source for alias, binding in (channels or {}).items()},
            "covariate_names": [str(item) for item in covariate_names],
            "group_names": [str(item) for item in group_names],
            "source": dict(source or {}),
        }
        write_json(root / "task.json", manifest)
        self.tasks.append(name)
        return manifest

    def finish(self) -> dict[str, Any]:
        if not self.graph_info:
            raise RuntimeError("write_graph must be called before finish")
        for name, entry in self.channels.items():
            root = self.root / "channels" / name
            required = ("matrix.npy", "gene_index.npy") if entry["kind"] == "dense" else ()
            for file_name in required:
                if not (root / file_name).is_file():
                    raise RuntimeError(f"channel {name!r} is missing {file_name}")
        manifest = {
            "format": FORMAT,
            "name": self.name,
            **self.graph_info,
            "channels": self.channels,
            "tasks": self.tasks,
            "source": self.source,
        }
        write_json(self.root / "dataset.json", manifest)
        return manifest
