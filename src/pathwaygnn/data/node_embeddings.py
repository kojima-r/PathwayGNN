"""Pre-computed node vectors, aligned onto a prepared graph's node list.

This is the engine side of an *external* node embedding: a table that names its
rows (``{node name: vector}``) and was produced by something else entirely — for
this repository, ``embedding_pc/`` runs a protein language model over the UniProt
sequence of every PathwayCommons gene and a molecule model over every ChEBI
compound. Nothing here knows that; a table is read, its names are matched against
``nodes.json``, and what comes out is one block of vectors per *group* plus the
graph node ids they belong to. Nodes the table does not name keep the encoder's
learned ``nn.Embedding`` row (see :class:`pathwaygnn.models.encoder.ExternalNodeEmbedding`).

A *group* is a set of nodes sharing one source model and therefore one width: the
protein block is 2560-dimensional and the chemical block 512-dimensional, so they
cannot share an adapter. Groups are what the encoder builds one ``nn.Linear`` for.

A table may also carry *aliases*: the same rows under another ID namespace, so one
file serves corpora that name their nodes differently (``data_tr`` uses gene
symbols, ``data_cancer`` numeric HGNC IDs). Which namespace to use is configured,
never guessed — see :class:`RawGroup`.

Tensor shapes below use these symbols: ``N`` graph nodes, ``g`` a group, ``m``
the nodes one group covers, ``D`` that group's width.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

NORMALIZATIONS = ("standardize", "l2", "none")
COMBINATIONS = ("add", "replace")
# `RelationalGIN.reset_parameters` initialises its learned rows at this std; the
# adapters are scaled against it so both kinds of row start out comparable.
EMBEDDING_INIT_STD = 0.1


def _identifier(name: str) -> str:
    """A group name usable as an ``nn.ModuleDict`` key and a buffer name."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", str(name)).strip("_")
    return cleaned or "group"


@dataclass(frozen=True)
class EmbeddingGroup:
    """One source model's block: which graph nodes it covers, and their vectors."""

    name: str
    nodes: np.ndarray    # int64 [m] — graph node ids, ascending
    vectors: np.ndarray  # float32 [m, D]

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def rms(self) -> float:
        """Root mean square of one coordinate — what the adapter's input scale is."""
        return float(np.sqrt(np.mean(np.square(self.vectors, dtype=np.float64))))

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.size)


@dataclass(frozen=True)
class NodeEmbeddingTable:
    """An external table already matched against one dataset's ``nodes.json``."""

    path: Path
    normalize: str
    combine: str
    init_std: float
    bias: bool
    aliases: tuple[str, ...]
    num_nodes: int
    groups: tuple[EmbeddingGroup, ...]
    num_named: int    # rows the file holds, before matching
    num_aliased: int  # covered nodes that matched through an alias, not the row's own name

    @property
    def num_covered(self) -> int:
        return sum(group.num_nodes for group in self.groups)

    @property
    def spec(self) -> dict[str, Any]:
        """What a checkpoint records so the adapter can be rebuilt from the file.

        The vectors themselves are *not* stored in the checkpoint (they are ~200 MB
        and unchanged by training), so this carries the path they were read from,
        how they were normalized, and the shape of every adapter.
        """
        return {
            "path": str(self.path),
            "normalize": self.normalize,
            "combine": self.combine,
            "init_std": self.init_std,
            "bias": self.bias,
            "aliases": list(self.aliases),
            "num_covered": self.num_covered,
            "num_aliased": self.num_aliased,
            "coverage": round(self.num_covered / max(self.num_nodes, 1), 6),
            "groups": {
                group.name: {"dim": group.dim, "num_nodes": group.num_nodes}
                for group in self.groups
            },
        }


def settings(block: Any) -> dict[str, Any] | None:
    """Normalize a ``model.node_embeddings:`` config value.

    Accepts a path string, a mapping (``path`` plus options), ``None``/``false``
    for "no external embeddings", and a recorded :attr:`NodeEmbeddingTable.spec`
    (whose extra keys are informational and ignored here).
    """
    if block is None or block is False:
        return None
    if isinstance(block, (str, Path)):
        block = {"path": str(block)}
    if not isinstance(block, dict):
        raise TypeError(
            "`model.node_embeddings` takes a path or a mapping with `path:`, "
            f"not {type(block).__name__}"
        )
    if not block.get("path"):
        raise KeyError("`model.node_embeddings` needs a `path:` to the vector table")
    normalize = str(block.get("normalize", "l2")).lower()
    if normalize not in NORMALIZATIONS:
        raise ValueError(
            f"`model.node_embeddings.normalize` must be one of {NORMALIZATIONS}, got {normalize!r}"
        )
    combine = str(block.get("combine", "add")).lower()
    if combine not in COMBINATIONS:
        raise ValueError(
            f"`model.node_embeddings.combine` must be one of {COMBINATIONS}, got {combine!r}"
        )
    aliases = block.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    aliases = [str(namespace) for namespace in aliases]
    init_std = block.get("init_std")
    # How large an adapted row starts out, per coordinate. `replace` *is* the node's
    # representation, so it starts at the learned rows' own scale (nn.Embedding is
    # initialised at std 0.1); `add` is a correction on top of a learned row, so it
    # starts at a tenth of that and the training decides how far to open it.
    default = EMBEDDING_INIT_STD if combine == "replace" else EMBEDDING_INIT_STD / 10
    return {
        "path": str(block["path"]),
        "normalize": normalize,
        "combine": combine,
        "aliases": aliases,
        "init_std": default if init_std is None else float(init_std),
        "bias": bool(block.get("bias", True)),
    }


@dataclass
class RawGroup:
    """One group as the file stores it, before any graph is involved.

    ``aliases`` maps an *ID namespace* (``hgnc_id``, ``entrez_id``, …) to the extra
    names it offers for the same rows. Namespaces stay separate because a bare
    number can be a valid identifier in more than one of them — HGNC:5 is A1BG
    while Entrez 5 is a different gene — so which one a corpus means is the
    caller's decision (``model.node_embeddings.aliases:``), never a guess.
    """

    names: list[str]
    vectors: np.ndarray  # [rows, D]
    aliases: dict[str, tuple[list[str], np.ndarray]] = field(default_factory=dict)


def _read_npz(path: Path) -> dict[str, RawGroup]:
    """``<group>_names`` + ``<group>_embeddings`` pairs, or a bare ``names``/``embeddings``.

    Alias arrays are optional and named ``<group>_alias_<namespace>_names`` /
    ``<group>_alias_<namespace>_rows`` (int64 row indices into the embeddings).
    """
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    table: dict[str, RawGroup] = {}
    for key in sorted(keys):
        if (not key.endswith("_names") and key != "names") or "_alias_" in key:
            continue
        group = "default" if key == "names" else key[: -len("_names")]
        for suffix in ("embeddings", "vectors", "embedding"):
            values = f"{group}_{suffix}" if group != "default" else suffix
            if values in keys:
                break
        else:
            raise KeyError(
                f"{path}: {key!r} has no matching vectors; expected "
                f"{group}_embeddings (or {group}_vectors)"
            )
        table[group] = RawGroup([str(name) for name in data[key]], np.asarray(data[values]))
    if not table:
        raise KeyError(
            f"{path} holds {sorted(keys)}; an npz table needs `<group>_names` and "
            "`<group>_embeddings` arrays (or plain `names`/`embeddings`)"
        )
    for group, raw in table.items():
        prefix = f"{group}_alias_"
        for key in sorted(keys):
            if not (key.startswith(prefix) and key.endswith("_names")):
                continue
            namespace = key[len(prefix) : -len("_names")]
            rows_key = f"{prefix}{namespace}_rows"
            if rows_key not in keys:
                raise KeyError(f"{path}: {key!r} has no matching {rows_key!r}")
            raw.aliases[namespace] = (
                [str(name) for name in data[key]],
                np.asarray(data[rows_key], dtype=np.int64),
            )
    return table


def _read_json(path: Path) -> dict[str, RawGroup]:
    """``{node name: vector}``, grouped by node type if a ``.meta.json`` sits beside it.

    ``embedding_pc`` writes ``node_embeddings.json`` next to
    ``node_embeddings.meta.json``, whose ``node_type`` map names each row's source
    model and whose ``alias_to_name`` holds ``{namespace: {alias: name}}``. Without
    that file the rows are grouped by width — the property the adapters actually
    depend on — and carry no aliases.
    """
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must hold a JSON object mapping node name -> vector")
    meta: dict[str, Any] = {}
    meta_path = path.with_suffix("").with_suffix(".meta.json") if path.suffix == ".json" else None
    if meta_path is not None and meta_path.is_file():
        with meta_path.open(encoding="utf-8") as handle:
            meta = json.load(handle)
    node_type = dict(meta.get("node_type") or {})
    collected: dict[str, tuple[list[str], list[Any]]] = {}
    for name, vector in raw.items():
        group = node_type.get(name) or f"dim{len(vector)}"
        names, vectors = collected.setdefault(group, ([], []))
        names.append(str(name))
        vectors.append(vector)
    table = {
        group: RawGroup(names, np.asarray(vectors, dtype=np.float32))
        for group, (names, vectors) in collected.items()
    }
    for namespace, mapping in (meta.get("alias_to_name") or {}).items():
        for group, raw_group in table.items():
            row_of = {name: row for row, name in enumerate(raw_group.names)}
            found = [(alias, row_of[name]) for alias, name in mapping.items() if name in row_of]
            if found:
                raw_group.aliases[namespace] = (
                    [alias for alias, _ in found],
                    np.array([row for _, row in found], dtype=np.int64),
                )
    return table


def read_table(path: str | Path) -> dict[str, RawGroup]:
    """Read a vector table without matching it against any graph."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; `model.node_embeddings.path` takes the node vector table "
            "written by embedding_pc/scripts/build_node_embeddings.py "
            "(node_embeddings.npz, or the slower node_embeddings.json)"
        )
    if path.suffix == ".npz":
        return _read_npz(path)
    if path.suffix == ".json":
        return _read_json(path)
    raise ValueError(f"{path}: expected a .npz or .json vector table, got {path.suffix!r}")


def _normalize(vectors: np.ndarray, how: str) -> np.ndarray:
    """Decide what a group's vectors keep before the adapter sees them.

    The adapter's initialisation divides by the group's RMS either way, so this is
    not about overall scale; it is about what is *kept*. ``l2`` (the default) keeps
    only the direction of each row, which is the part the source models are
    validated on — cosine similarity — and is what pre-trains best here.
    ``standardize`` z-scores each dimension over the covered nodes, which on
    ESMC vectors amplifies the near-constant dimensions into noise and measurably
    hurts (``README.md`` §5.1). ``none`` hands them over verbatim.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if how == "l2":
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norm, 1e-12)
    if how == "standardize":
        mean = vectors.mean(axis=0, keepdims=True)
        std = vectors.std(axis=0, keepdims=True)
        return (vectors - mean) / np.maximum(std, 1e-6)
    return vectors


def load_node_embeddings(
    block: Any, node_names: Sequence[str] | None
) -> NodeEmbeddingTable | None:
    """Read the configured table and match it against ``node_names``.

    Args:
        block: the ``model.node_embeddings:`` value, or a recorded spec; ``None``
            means the encoder keeps its learned embedding for every node.
        node_names: the dataset's ``nodes.json``, i.e. graph node id -> name.

    Returns:
        A :class:`NodeEmbeddingTable` whose groups hold only the nodes this graph
        has, or ``None`` when nothing is configured.
    """
    options = settings(block)
    if options is None:
        return None
    if node_names is None:
        raise ValueError(
            "external node embeddings are matched by node name, so the dataset's "
            "nodes.json is required (pass `node_names=dataset.node_names()`)"
        )
    index = {str(name): position for position, name in enumerate(node_names)}
    table = read_table(options["path"])
    available = sorted({namespace for raw in table.values() for namespace in raw.aliases})
    unknown = [namespace for namespace in options["aliases"] if namespace not in available]
    if unknown:
        raise ValueError(
            f"{options['path']} carries no alias namespace {unknown}; it offers "
            f"{available or 'none'} (`model.node_embeddings.aliases` picks which ID namespace "
            "the graph's node names are in, e.g. [hgnc_id] for data_cancer)"
        )
    groups: list[EmbeddingGroup] = []
    taken: dict[int, str] = {}
    num_named = num_aliased = 0
    for group, raw in sorted(table.items()):
        vectors = np.asarray(raw.vectors)
        if vectors.ndim != 2 or len(raw.names) != vectors.shape[0]:
            raise ValueError(
                f"{options['path']}: group {group!r} has {len(raw.names)} names but "
                f"vectors of shape {vectors.shape}"
            )
        num_named += len(raw.names)
        # The table's own names first, then each requested alias namespace in the
        # order it was configured: a node keeps the first row that claims it.
        candidates: list[tuple[str, int, bool]] = [
            (name, row, False) for row, name in enumerate(raw.names)
        ]
        for namespace in options["aliases"]:
            alias_names, alias_rows = raw.aliases.get(namespace, ([], np.zeros(0, np.int64)))
            if len(alias_rows) and int(alias_rows.max()) >= vectors.shape[0]:
                raise ValueError(
                    f"{options['path']}: alias namespace {namespace!r} of group {group!r} "
                    "points past the group's rows"
                )
            candidates += [
                (name, int(row), True) for name, row in zip(alias_names, alias_rows)
            ]
        rows: dict[int, tuple[int, bool]] = {}
        for name, row, is_alias in candidates:
            node = index.get(name)
            if node is not None:
                rows.setdefault(node, (row, is_alias))
        if not rows:
            continue
        for node in rows:
            previous = taken.setdefault(node, group)
            if previous != group:
                raise ValueError(
                    f"{options['path']}: node {node_names[node]!r} is covered by both "
                    f"group {previous!r} and group {group!r}; a node takes one vector"
                )
        ordered = sorted(rows)
        num_aliased += sum(rows[node][1] for node in ordered)
        nodes = np.fromiter(ordered, dtype=np.int64, count=len(ordered))
        selected = vectors[[rows[node][0] for node in ordered]]
        if not np.isfinite(selected).all():
            raise ValueError(f"{options['path']}: group {group!r} contains NaN or Inf")
        groups.append(
            EmbeddingGroup(
                name=_identifier(group),
                nodes=nodes,
                vectors=_normalize(selected, options["normalize"]),
            )
        )
    if not groups:
        raise ValueError(
            f"{options['path']} names none of this dataset's {len(index)} nodes; the table's "
            "names must match nodes.json (gene symbols and CHEBI:<id> for embedding_pc"
            f", or set `model.node_embeddings.aliases` to one of {available or 'none'})"
        )
    return NodeEmbeddingTable(
        path=Path(options["path"]),
        normalize=options["normalize"],
        combine=options["combine"],
        init_std=options["init_std"],
        bias=options["bias"],
        aliases=tuple(options["aliases"]),
        num_nodes=len(index),
        groups=tuple(groups),
        num_named=num_named,
        num_aliased=num_aliased,
    )


def check_spec(table: NodeEmbeddingTable, spec: dict[str, Any]) -> None:
    """Fail loudly when a rebuilt table cannot drive a checkpoint's adapters.

    The adapter shapes have to match — the group names and their widths — and so
    does ``combine``, which decides what the adapted vector *does* to a node's row.
    The covered node *count* may legitimately differ, because ``pred`` may score a
    different corpus with the same graph size.
    """
    recorded = {name: int(entry["dim"]) for name, entry in (spec.get("groups") or {}).items()}
    rebuilt = {group.name: group.dim for group in table.groups}
    if recorded != rebuilt:
        raise ValueError(
            f"{table.path} yields adapter groups {rebuilt}, but the checkpoint was trained "
            f"with {recorded}; point `model.node_embeddings.path` at the table it was "
            "pre-trained against, or re-run pre-training"
        )
    if spec.get("combine", "add") != table.combine:
        raise ValueError(
            f"the checkpoint was pre-trained with combine={spec.get('combine')!r}, but this "
            f"config asks for {table.combine!r}; the two build different node representations"
        )
