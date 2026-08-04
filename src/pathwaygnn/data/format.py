"""The single dataset format that every ``pathwaygnn`` command consumes.

Dataset-specific preprocessing lives outside this package (see
``pathwaygnn_datasets``). Preprocessing must finish before training starts and
must leave behind exactly this layout::

    <root>/dataset.json                 manifest; ``name`` identifies the dataset
    <root>/graph.pt                     {"edge_index": int64[2,E], "edge_type": int64[E]}
    <root>/nodes.json                   node index -> name (HGNC symbol or ID)
    <root>/relations.json               relation index -> name
    <root>/node_features/<name>/        a gene-value table, shared by tasks
        node_feature.json               {"kind": "sparse"|"dense", ...}
        ptr.npy gene.npy value.npy      kind == sparse: CSR rows over node indices
        matrix.npy gene_index.npy       kind == dense: memmapped [rows, genes]
    <root>/tasks/<task>/
        task.json                       manifest: aliases, groups, sample-level features
        labels.npy                      float32[samples], binary
        groups.npy                      int64[samples]        optional
        sample_features.npy             float32[samples, dim] optional
        rows/<alias>.npy                int64[samples]        optional, identity if absent

A *node-level feature* is one gene-value view of a sample (a perturbation
signature, a disease signature, an expression profile): one value per graph node.
These tables are dataset-level, so several tasks can share one. A *sample-level
feature* is instead one dense vector per sample, carrying whatever is not attached
to a gene (cancer type, mutational spectra, a compound fingerprint).

A *task* is one binary prediction problem that binds each node-level feature to a
local alias (``task.json``'s ``node_features`` maps alias -> table) and maps its
samples to that table's rows through ``rows/<alias>.npy``. Tasks that describe the
same problem on different sample sets therefore use the same aliases, which keeps
their model configurations interchangeable.
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
class NodeFeature:
    """A gene-value table; ``dense`` tables are memmapped, sparse ones are CSR.

    ``name`` is the alias the task uses, ``source`` the dataset-level table.
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
    """One binary prediction problem defined over a dataset's node-level features."""

    name: str
    root: Path
    node_features: tuple[NodeFeature, ...]
    num_samples: int
    seed_offset: int
    sample_feature_names: tuple[str, ...]
    group_names: tuple[str, ...]
    manifest: dict[str, Any]

    @property
    def node_feature_names(self) -> tuple[str, ...]:
        return tuple(node_feature.name for node_feature in self.node_features)

    @property
    def sample_feature_dim(self) -> int:
        return len(self.sample_feature_names)

    @property
    def num_groups(self) -> int:
        return len(self.group_names)

    def labels(self) -> np.ndarray:
        return np.load(self.root / "labels.npy")

    def groups(self) -> np.ndarray | None:
        path = self.root / "groups.npy"
        return np.load(path, mmap_mode="r") if path.is_file() else None

    def sample_features(self) -> np.ndarray | None:
        path = self.root / "sample_features.npy"
        return np.load(path, mmap_mode="r") if path.is_file() else None

    def rows(self, alias: str) -> np.ndarray | None:
        """Sample -> feature-table row; ``None`` means the identity mapping."""
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

    def node_feature(self, source: str, alias: str | None = None) -> NodeFeature:
        try:
            entry = self.manifest["node_features"][source]
        except KeyError as error:
            raise KeyError(
                f"dataset {self.name!r} has no node-level feature {source!r}; "
                f"available: {sorted(self.manifest['node_features'])}"
            ) from error
        return NodeFeature(
            name=alias or source,
            source=source,
            kind=entry["kind"],
            root=self.root / "node_features" / source,
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
            node_features=tuple(
                self.node_feature(source, alias) for alias, source in manifest["node_features"].items()
            ),
            num_samples=int(manifest["num_samples"]),
            seed_offset=int(manifest.get("seed_offset", self.task_names.index(name))),
            sample_feature_names=tuple(manifest.get("sample_feature_names", ())),
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
class TaskNodeFeature:
    """A task's binding of one node-level feature: which table, and which rows."""

    source: str
    rows: np.ndarray | None = None


class DatasetWriter:
    """Writer used by preprocessing to emit the format described in this module."""

    def __init__(self, root: str | Path, name: str, source: dict[str, Any] | None = None):
        self.root = Path(root).resolve()
        self.name = name
        self.source = dict(source or {})
        self.node_features: dict[str, dict[str, Any]] = {}
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

    def _node_feature_root(self, name: str) -> Path:
        root = self.root / "node_features" / name
        root.mkdir(parents=True, exist_ok=True)
        return root

    def sparse_node_feature(
        self, name: str, ptr: Iterable[int], gene: Iterable[int], value: Iterable[float]
    ) -> None:
        root = self._node_feature_root(name)
        ptr = np.asarray(ptr, dtype=np.int64)
        gene = np.asarray(gene, dtype=np.int64)
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if gene.size != value.size or ptr[-1] != gene.size:
            raise ValueError(f"node-level feature {name!r}: ptr/gene/value are inconsistent")
        for field_name, array in (("ptr", ptr), ("gene", gene), ("value", value)):
            np.save(root / f"{field_name}.npy", array)
        self._register(name, "sparse", int(ptr.size - 1), None, {"num_values": int(gene.size)})

    def dense_node_feature(self, name: str, num_rows: int, num_features: int) -> Path:
        """Register a dense node-level feature and return its directory.

        The caller writes ``matrix.npy`` (float32 ``[num_rows, num_features]``) and
        ``gene_index.npy`` (int64 ``[num_features]``) itself, so that large
        matrices can be streamed straight into a memmap.
        """
        root = self._node_feature_root(name)
        self._register(name, "dense", int(num_rows), int(num_features), {})
        return root

    def _register(
        self, name: str, kind: str, num_rows: int, num_features: int | None, extra: dict[str, Any]
    ) -> None:
        entry = {"kind": kind, "num_rows": num_rows, **extra}
        if num_features is not None:
            entry["num_features"] = num_features
        write_json(self.root / "node_features" / name / "node_feature.json", entry)
        self.node_features[name] = entry

    def write_task(
        self,
        name: str,
        labels: np.ndarray,
        node_features: dict[str, "TaskNodeFeature"] | None = None,
        sample_features: np.ndarray | None = None,
        sample_feature_names: Sequence[str] = (),
        groups: np.ndarray | None = None,
        group_names: Sequence[str] = (),
        seed_offset: int | None = None,
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        root = self.root / "tasks" / name
        root.mkdir(parents=True, exist_ok=True)
        labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        np.save(root / "labels.npy", labels)
        for alias, binding in (node_features or {}).items():
            if binding.source not in self.node_features:
                raise KeyError(f"task {name!r} references unknown node-level feature {binding.source!r}")
            expected = self.node_features[binding.source]["num_rows"]
            if binding.rows is None:
                if expected != labels.size:
                    raise ValueError(
                        f"task {name!r} maps every sample to its own row of node-level feature "
                        f"{binding.source!r}, but the node_feature has {expected} rows and the task "
                        f"{labels.size} samples"
                    )
                continue
            row = np.asarray(binding.rows, dtype=np.int64).reshape(-1)
            if row.size != labels.size or (row.size and (row.min() < 0 or row.max() >= expected)):
                raise ValueError(f"task {name!r}: rows/{alias}.npy is out of range")
            (root / "rows").mkdir(exist_ok=True)
            np.save(root / "rows" / f"{alias}.npy", row)
        if sample_features is not None:
            sample_features = np.asarray(sample_features, dtype=np.float32)
            if sample_features.shape != (labels.size, len(sample_feature_names)):
                raise ValueError(
                    f"task {name!r}: sample features {sample_features.shape} do not match "
                    f"{labels.size} samples x {len(sample_feature_names)} names"
                )
            np.save(root / "sample_features.npy", sample_features)
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
            "node_features": {alias: binding.source for alias, binding in (node_features or {}).items()},
            "sample_feature_names": [str(item) for item in sample_feature_names],
            "group_names": [str(item) for item in group_names],
            "source": dict(source or {}),
        }
        write_json(root / "task.json", manifest)
        self.tasks.append(name)
        return manifest

    def finish(self) -> dict[str, Any]:
        if not self.graph_info:
            raise RuntimeError("write_graph must be called before finish")
        for name, entry in self.node_features.items():
            root = self.root / "node_features" / name
            required = ("matrix.npy", "gene_index.npy") if entry["kind"] == "dense" else ()
            for file_name in required:
                if not (root / file_name).is_file():
                    raise RuntimeError(f"node-level feature {name!r} is missing {file_name}")
        manifest = {
            "format": FORMAT,
            "name": self.name,
            **self.graph_info,
            "node_features": self.node_features,
            "tasks": self.tasks,
            "source": self.source,
        }
        write_json(self.root / "dataset.json", manifest)
        return manifest
