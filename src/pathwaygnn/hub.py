"""Publish a trained checkpoint to the HuggingFace Hub, and start from a published one.

Two directions, one reference syntax.

**Publishing** (``pathwaygnn hg``) takes a checkpoint this engine wrote — the
encoder from ``pretrain``, or the head from ``finetune``/``cv`` — and pushes it
with everything needed to use it again: a ``pathwaygnn.json`` manifest describing
the artifact, and a model card whose usage section is the literal config lines to
paste. ``dry_run: true`` assembles that payload locally and uploads nothing, which
is the way to inspect a release before making it.

**Starting from a published model** needs no new command. Anywhere a config names a
checkpoint — ``pretrained_checkpoint``, ``checkpoint``, ``resume_from`` — a
reference of the form::

    hf://<owner>/<name>[/<path in repo>][@<revision>]

is downloaded to the local Hub cache and used in place of a path, so `cv`,
`finetune`, `ig`, `pred` and a resumed `pretrain` all take one without changes. The
path defaults to ``model.pt``, which is what the uploader writes.

``huggingface_hub`` is an optional dependency (the ``hub`` extra); it is imported
inside the functions that need it, and never at all for a dry run.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

HUB_SCHEME = "hf://"
#: The checkpoint name the uploader writes, and the default a reference resolves to.
DEFAULT_FILENAME = "model.pt"
MANIFEST_NAME = "pathwaygnn.json"
CARD_NAME = "README.md"
INSTALL_HINT = (
    "the HuggingFace Hub needs `huggingface_hub`: pip install -e '.[hub]' "
    "(or set `dry_run: true` to assemble the upload locally without it)"
)


@dataclass(frozen=True)
class HubReference:
    """A parsed ``hf://`` reference."""

    repo_id: str
    filename: str = DEFAULT_FILENAME
    revision: str | None = None

    def __str__(self) -> str:
        suffix = f"@{self.revision}" if self.revision else ""
        return f"{HUB_SCHEME}{self.repo_id}/{self.filename}{suffix}"


def is_reference(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(HUB_SCHEME)


def parse_reference(value: str) -> HubReference:
    """``hf://owner/name/sub/dir/model.pt@v1`` -> repo ``owner/name``, that file, ``v1``.

    The first two segments are the repository id, because that is what a Hub id
    always is; everything after it is the path inside the repository.
    """
    if not is_reference(value):
        raise ValueError(f"{value!r} is not a {HUB_SCHEME} reference")
    body = value[len(HUB_SCHEME) :]
    revision = None
    if "@" in body:
        body, _, revision = body.rpartition("@")
        if "/" in revision:  # an '@' inside the path, not a revision marker
            body, revision = f"{body}@{revision}", None
    segments = [segment for segment in body.split("/") if segment]
    if len(segments) < 2:
        raise ValueError(
            f"{value!r} is missing a repository: expected "
            f"{HUB_SCHEME}<owner>/<name>[/<path>][@<revision>]"
        )
    return HubReference(
        repo_id="/".join(segments[:2]),
        filename="/".join(segments[2:]) or DEFAULT_FILENAME,
        revision=revision,
    )


def resolve_checkpoint(value: str | Path, token: str | None = None) -> Path:
    """A local path for ``value``, downloading it first if it is an ``hf://`` reference."""
    if not is_reference(value):
        return Path(value)
    reference = parse_reference(str(value))
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - exercised via a stub in tests
        raise ImportError(f"{value} cannot be fetched: {INSTALL_HINT}") from error
    return Path(
        hf_hub_download(
            repo_id=reference.repo_id,
            filename=reference.filename,
            revision=reference.revision,
            token=token,
        )
    )


def describe_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """What kind of artifact a checkpoint is, and the facts a model card needs.

    ``encoder``: the pre-trained relational graph encoder plus its DistMult relation
    embedding, usable as any config's ``pretrained_checkpoint``. ``head``: a trained
    sample-level model, usable by ``pred`` and ``ig``, which also needs the encoder
    it was trained against.
    """
    model_config = dict(checkpoint.get("model_config") or {})
    if "predictor" in checkpoint:
        kind = "head"
    elif "encoder" in checkpoint and "relation" in checkpoint:
        kind = "encoder"
    else:
        raise ValueError(
            "this file is not a pathwaygnn checkpoint: expected `predictor` (a trained "
            "head from finetune/cv) or `encoder` and `relation` (a pre-trained encoder "
            f"from pretrain), found {sorted(checkpoint)}"
        )
    return {
        "kind": kind,
        "dataset": checkpoint.get("dataset"),
        "task": checkpoint.get("task"),
        "epoch": checkpoint.get("epoch"),
        "model_config": model_config,
        "variant": dict(checkpoint.get("variant") or {}),
        "needs_pretrained_checkpoint": bool(
            kind == "head" and model_config.get("use_graph", False)
        ),
        "trained_against": checkpoint.get("pretrained_checkpoint"),
    }


def _card(description: dict[str, Any], reference: HubReference, cfg: dict[str, Any]) -> str:
    """The model card: what this is, and the config lines that consume it."""
    kind = description["kind"]
    model_config = description["model_config"]
    dataset = description["dataset"] or "unknown"
    tags = ["pathwaygnn", "graph-neural-network", "biology", kind]
    front = [
        "---",
        "library_name: pathwaygnn",
        f"license: {cfg.get('license', 'mit')}",
        "tags:",
        *[f"  - {tag}" for tag in tags],
        "---",
        "",
    ]
    if kind == "encoder":
        what = (
            f"A **pre-trained relational graph encoder** (`RelationalGIN`) for the "
            f"`{dataset}` pathway graph: {model_config.get('num_nodes')} nodes and "
            f"{model_config.get('num_relations')} relation types, hidden dimension "
            f"{model_config.get('hidden_dim')}, {model_config.get('num_layers')} layers. "
            "Pre-trained on edge prediction with a DistMult relation embedding."
        )
        usage = [
            "Point any downstream config at it — `cv`, `finetune`, `ig` and `pred` all",
            "accept a Hub reference where they accept a path:",
            "",
            "    pretrained_checkpoint: " + str(reference),
            "",
            "or continue pre-training from it:",
            "",
            "    # configs/<dataset>/pretrain.yaml",
            "    resume_from: " + str(reference),
        ]
    else:
        aliases = ", ".join(f"`{name}`" for name in model_config.get("node_features", []))
        what = (
            f"A **trained sample-level head** for task `{description['task']}` of the "
            f"`{dataset}` dataset. It consumes the node-level features {aliases} in that "
            f"order, {model_config.get('sample_feature_dim', 0)} sample-level features, and "
            + (
                "the node embeddings of a pre-trained encoder."
                if model_config.get("use_graph")
                else "no graph (the graph-free ablation)."
            )
        )
        usage = [
            "Score a prepared dataset with it:",
            "",
            "    # configs/<dataset>/pred.yaml",
            "    checkpoint: " + str(reference),
        ]
        if description["needs_pretrained_checkpoint"]:
            usage += [
                "    pretrained_checkpoint: <the encoder this head was trained against>",
                "",
                "The encoder is required: this head adds node embeddings to every gene value,",
                "so it only reproduces its training behaviour with the same encoder.",
            ]
    lines = front + [
        f"# {cfg.get('title') or reference.repo_id}",
        "",
        what,
        "",
        "Produced by [PathwayGNN](https://github.com/) — see `pathwaygnn.json` in this",
        "repository for the full manifest.",
        "",
        "## Usage",
        "",
        *usage,
        "",
        "## What is in this repository",
        "",
        f"- `{DEFAULT_FILENAME}` — the checkpoint (`torch.load`, `weights_only=False`)",
        f"- `{MANIFEST_NAME}` — dataset, model shape and provenance, machine readable",
    ]
    for name in cfg.get("extra_files", []):
        lines.append(f"- `{Path(name).name}` — copied from the training run")
    if cfg.get("notes"):
        lines += ["", "## Notes", "", str(cfg["notes"])]
    lines += [
        "",
        "## Caveats",
        "",
        "- **The node indexing is part of the model.** These weights are indexed by the",
        "  node order of the graph they were trained on, so data scored with them has to be",
        "  prepared over that same graph. `pathwaygnn` checks the node and relation counts",
        "  and refuses a mismatch, but identical counts over a different node order cannot",
        "  be detected — reuse the dataset build, not just its shape.",
        "",
    ]
    return "\n".join(lines)


def stage_upload(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any], HubReference]:
    """Assemble exactly what would be pushed, in ``output_dir``.

    Kept separate from the upload so that a release can be inspected — and tested —
    without a network or a token.
    """
    source = resolve_checkpoint(cfg["checkpoint"], cfg.get("token"))
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} is missing; point `checkpoint:` at a `pretrain`/`finetune` best.pt "
            "or a `cv` fold's model.pt"
        )
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    description = describe_checkpoint(checkpoint)
    repo_id = cfg["repo_id"]
    reference = HubReference(repo_id=repo_id, filename=DEFAULT_FILENAME)
    staging = Path(cfg["output_dir"])
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copy2(source, staging / DEFAULT_FILENAME)
    manifest = {
        "format": "pathwaygnn/hub/1",
        "repo_id": repo_id,
        "reference": str(reference),
        "checkpoint": DEFAULT_FILENAME,
        "source": str(source),
        **description,
    }
    copied = []
    for name in cfg.get("extra_files", []):
        path = Path(name)
        if not path.is_file():
            raise FileNotFoundError(f"extra_files lists {path}, which does not exist")
        shutil.copy2(path, staging / path.name)
        copied.append(path.name)
    manifest["extra_files"] = copied
    (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    (staging / CARD_NAME).write_text(_card(description, reference, cfg))
    return staging, manifest, reference


def run_hub_upload(cfg: dict[str, Any]) -> dict[str, Any]:
    """``pathwaygnn hg``: stage a checkpoint and push it to the Hub."""
    staging, manifest, reference = stage_upload(cfg)
    files = sorted(path.name for path in staging.iterdir())
    result = {
        "reference": str(reference),
        "repo_id": cfg["repo_id"],
        "kind": manifest["kind"],
        "dataset": manifest["dataset"],
        "task": manifest["task"],
        "staged": str(staging),
        "files": files,
        "bytes": sum(path.stat().st_size for path in staging.iterdir()),
    }
    if cfg.get("dry_run", False):
        # Nothing leaves the machine: `staged` is the whole payload, for review.
        result["uploaded"] = False
        result["note"] = "dry run; set `dry_run: false` to push this payload"
        print(json.dumps({"staged_files": files}, indent=2))
        return result
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise ImportError(INSTALL_HINT) from error
    api = HfApi(token=cfg.get("token"))
    api.create_repo(
        repo_id=cfg["repo_id"],
        repo_type="model",
        private=bool(cfg.get("private", True)),
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=cfg["repo_id"],
        folder_path=str(staging),
        repo_type="model",
        revision=cfg.get("revision"),
        commit_message=cfg.get(
            "commit_message",
            f"pathwaygnn {manifest['kind']} for {manifest['dataset']}"
            + (f" / {manifest['task']}" if manifest["task"] else ""),
        ),
    )
    result["uploaded"] = True
    result["private"] = bool(cfg.get("private", True))
    result["commit"] = getattr(commit, "commit_url", str(commit))
    return result
