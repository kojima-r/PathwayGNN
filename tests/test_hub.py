"""Publishing to the HuggingFace Hub, and starting from a published checkpoint.

``huggingface_hub`` is an optional extra, so nothing here imports it: the reference
parsing and the staged payload need no network at all, and the two calls that would
reach one are exercised against a stub installed in ``sys.modules``.
"""

import io
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from conftest import build_dataset
from pathwaygnn.data.format import GraphDataset
from pathwaygnn.hub import (
    DEFAULT_FILENAME,
    HubReference,
    describe_checkpoint,
    is_reference,
    parse_reference,
    resolve_checkpoint,
    run_hub_upload,
    stage_upload,
)
from pathwaygnn.training.pretrain import run_pretraining


@pytest.fixture
def hub_stub(monkeypatch):
    """A stand-in for `huggingface_hub` that records calls instead of making them."""
    calls: dict[str, list] = {"download": [], "create_repo": [], "upload_folder": []}
    module = types.ModuleType("huggingface_hub")

    class Api:
        def __init__(self, token=None):
            self.token = token

        def create_repo(self, **kwargs):
            calls["create_repo"].append(kwargs)

        def upload_folder(self, **kwargs):
            calls["upload_folder"].append(kwargs)
            return types.SimpleNamespace(commit_url="https://hf.co/fake/commit/abc")

    def hf_hub_download(repo_id, filename, revision=None, token=None, **kwargs):
        calls["download"].append(
            {"repo_id": repo_id, "filename": filename, "revision": revision, "token": token}
        )
        return str(calls["served"])

    module.HfApi = Api
    module.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


@pytest.fixture
def encoder_checkpoint(tmp_path: Path, dataset: GraphDataset) -> Path:
    run_pretraining({
        "seed": 1,
        "device": "cpu",
        "dataset": {"name": dataset.name, "dir": str(dataset.root)},
        "model": {"hidden_dim": 4, "num_layers": 1, "dropout": 0.0},
        "training": {"epochs": 2, "steps_per_epoch": 2, "batch_size": 8},
        "output_dir": str(tmp_path / "pretrain"),
    })
    return tmp_path / "pretrain" / "last.pt"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("hf://owner/name", HubReference("owner/name", DEFAULT_FILENAME, None)),
        ("hf://owner/name/model.pt", HubReference("owner/name", "model.pt", None)),
        ("hf://owner/name/sub/dir/a.pt", HubReference("owner/name", "sub/dir/a.pt", None)),
        ("hf://owner/name/model.pt@v1", HubReference("owner/name", "model.pt", "v1")),
        ("hf://owner/name@main", HubReference("owner/name", DEFAULT_FILENAME, "main")),
    ],
)
def test_references_parse(value: str, expected: HubReference):
    assert parse_reference(value) == expected
    assert is_reference(value)
    # A parsed reference round-trips through its own string form.
    assert parse_reference(str(expected)) == expected


def test_local_paths_are_left_alone(tmp_path: Path):
    assert not is_reference(str(tmp_path))
    assert resolve_checkpoint(tmp_path / "a.pt") == tmp_path / "a.pt"
    with pytest.raises(ValueError, match="missing a repository"):
        parse_reference("hf://owner")


def test_a_reference_is_downloaded_and_then_used_as_a_path(
    tmp_path: Path, encoder_checkpoint: Path, hub_stub
):
    hub_stub["served"] = encoder_checkpoint
    resolved = resolve_checkpoint("hf://owner/name/model.pt@v2")
    assert resolved == encoder_checkpoint
    assert hub_stub["download"] == [
        {"repo_id": "owner/name", "filename": "model.pt", "revision": "v2", "token": None}
    ]


def test_checkpoint_kinds_are_told_apart(encoder_checkpoint: Path):
    encoder = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)
    assert describe_checkpoint(encoder)["kind"] == "encoder"
    head = {
        "predictor": {},
        "model_config": {"node_features": ["a"], "use_graph": True, "sample_feature_dim": 0},
        "dataset": "toy",
        "task": "main",
    }
    described = describe_checkpoint(head)
    assert described["kind"] == "head"
    assert described["needs_pretrained_checkpoint"] is True
    with pytest.raises(ValueError, match="not a pathwaygnn checkpoint"):
        describe_checkpoint({"state_dict": {}})


def test_staged_payload_is_self_describing(tmp_path: Path, encoder_checkpoint: Path):
    staging, manifest, reference = stage_upload({
        "checkpoint": str(encoder_checkpoint),
        "repo_id": "owner/pathwaygnn-toy",
        "extra_files": [str(encoder_checkpoint.parent / "history.json")],
        "output_dir": str(tmp_path / "staged"),
    })
    assert sorted(path.name for path in staging.iterdir()) == [
        "README.md", "history.json", "model.pt", "pathwaygnn.json"
    ]
    # The checkpoint is copied verbatim, so a consumer loads exactly what was trained.
    assert (staging / "model.pt").read_bytes() == encoder_checkpoint.read_bytes()
    assert manifest == json.loads((staging / "pathwaygnn.json").read_text())
    assert manifest["kind"] == "encoder"
    assert manifest["extra_files"] == ["history.json"]
    card = (staging / "README.md").read_text()
    assert card.startswith("---\nlibrary_name: pathwaygnn")
    # The card tells a reader the exact config lines that consume this artifact.
    assert f"pretrained_checkpoint: {reference}" in card
    assert f"resume_from: {reference}" in card
    assert str(reference) == "hf://owner/pathwaygnn-toy/model.pt"


def test_a_missing_extra_file_fails_before_anything_is_staged(
    tmp_path: Path, encoder_checkpoint: Path
):
    with pytest.raises(FileNotFoundError, match="extra_files"):
        stage_upload({
            "checkpoint": str(encoder_checkpoint),
            "repo_id": "owner/name",
            "extra_files": ["nope.json"],
            "output_dir": str(tmp_path / "staged"),
        })


def test_dry_run_uploads_nothing(tmp_path: Path, encoder_checkpoint: Path, monkeypatch):
    # Any attempt to reach the Hub in a dry run must fail the test, so make the
    # import itself impossible.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    result = run_hub_upload({
        "checkpoint": str(encoder_checkpoint),
        "repo_id": "owner/name",
        "dry_run": True,
        "output_dir": str(tmp_path / "staged"),
    })
    assert result["uploaded"] is False
    assert result["files"] == ["README.md", "model.pt", "pathwaygnn.json"]
    assert result["reference"] == "hf://owner/name/model.pt"


def test_upload_creates_the_repository_and_pushes_the_folder(
    tmp_path: Path, encoder_checkpoint: Path, hub_stub
):
    result = run_hub_upload({
        "checkpoint": str(encoder_checkpoint),
        "repo_id": "owner/name",
        "dry_run": False,
        "private": True,
        "output_dir": str(tmp_path / "staged"),
    })
    assert hub_stub["create_repo"] == [
        {"repo_id": "owner/name", "repo_type": "model", "private": True, "exist_ok": True}
    ]
    pushed = hub_stub["upload_folder"][0]
    assert pushed["repo_id"] == "owner/name"
    assert Path(pushed["folder_path"]) == tmp_path / "staged"
    assert pushed["repo_type"] == "model"
    assert result["uploaded"] is True and result["private"] is True
    assert result["commit"] == "https://hf.co/fake/commit/abc"


def test_repositories_are_private_unless_asked_otherwise(
    tmp_path: Path, encoder_checkpoint: Path, hub_stub
):
    run_hub_upload({
        "checkpoint": str(encoder_checkpoint),
        "repo_id": "owner/name",
        "dry_run": False,
        "output_dir": str(tmp_path / "staged"),
    })
    assert hub_stub["create_repo"][0]["private"] is True


def _epochs(function, *args, **kwargs) -> list[dict]:
    """The per-epoch records `pretrain` prints (history.json only keeps improvements)."""
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        function(*args, **kwargs)
    return [
        json.loads(line) for line in buffer.getvalue().splitlines() if line.startswith('{"epoch"')
    ]


def _pretrain_config(dataset: GraphDataset, output: Path, epochs: int, **extra) -> dict:
    return {
        "seed": 7,
        "device": "cpu",
        "dataset": {"name": dataset.name, "dir": str(dataset.root)},
        "model": {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0},
        "training": {"epochs": epochs, "steps_per_epoch": 4, "batch_size": 16},
        "output_dir": str(output),
        **extra,
    }


def test_resuming_reproduces_an_uninterrupted_run(tmp_path: Path, dataset: GraphDataset):
    """The property that makes `resume_from` a resume and not a warm restart."""
    _epochs(run_pretraining, _pretrain_config(dataset, tmp_path / "a", 3))
    resumed = _epochs(
        run_pretraining,
        _pretrain_config(
            dataset, tmp_path / "b", 2, resume_from=str(tmp_path / "a" / "last.pt")
        ),
    )
    straight = _epochs(run_pretraining, _pretrain_config(dataset, tmp_path / "c", 5))
    assert [record["epoch"] for record in resumed] == [4, 5]
    # Same weights, same optimizer moments, same sampling stream.
    assert resumed == [record for record in straight if record["epoch"] >= 4]


def test_resuming_from_a_hub_reference(tmp_path: Path, dataset: GraphDataset, hub_stub):
    _epochs(run_pretraining, _pretrain_config(dataset, tmp_path / "a", 2))
    hub_stub["served"] = tmp_path / "a" / "last.pt"
    resumed = _epochs(
        run_pretraining,
        _pretrain_config(dataset, tmp_path / "b", 1, resume_from="hf://owner/name/model.pt"),
    )
    assert [record["epoch"] for record in resumed] == [3]
    assert hub_stub["download"][0]["repo_id"] == "owner/name"


def test_resume_refuses_a_head_or_a_different_graph(tmp_path: Path, dataset: GraphDataset):
    head = tmp_path / "head.pt"
    torch.save({"predictor": {}, "model_config": {}}, head)
    with pytest.raises(KeyError, match="not a `pretrain` checkpoint"):
        run_pretraining(_pretrain_config(dataset, tmp_path / "x", 1, resume_from=str(head)))

    _epochs(run_pretraining, _pretrain_config(dataset, tmp_path / "a", 1))
    bigger = build_dataset(tmp_path / "bigger", name="bigger")
    checkpoint = torch.load(tmp_path / "a" / "last.pt", map_location="cpu", weights_only=False)
    checkpoint["model_config"]["num_nodes"] += 1
    torch.save(checkpoint, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="nodes and"):
        run_pretraining(
            _pretrain_config(bigger, tmp_path / "y", 1, resume_from=str(tmp_path / "wrong.pt"))
        )


def test_a_published_encoder_shape_wins_over_the_local_config(
    tmp_path: Path, dataset: GraphDataset
):
    """A resumed run must not be silently reshaped by the config it is launched with."""
    _epochs(run_pretraining, _pretrain_config(dataset, tmp_path / "a", 1))
    config = _pretrain_config(
        dataset, tmp_path / "b", 1, resume_from=str(tmp_path / "a" / "last.pt")
    )
    config["model"] = {"hidden_dim": 64, "num_layers": 3, "dropout": 0.5}
    _epochs(run_pretraining, config)
    written = torch.load(tmp_path / "b" / "last.pt", map_location="cpu", weights_only=False)
    assert written["model_config"]["hidden_dim"] == 8
    assert written["model_config"]["num_layers"] == 1
