"""External node vectors: reading, matching, adapting, and surviving a checkpoint."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pathwaygnn.data.format import GraphDataset
from pathwaygnn.data.node_embeddings import load_node_embeddings, read_table, settings
from pathwaygnn.models.encoder import RelationalGIN, encoder_config, load_encoder

# The conftest dataset names its nodes N0..N23; two groups of different widths,
# the way embedding_pc emits proteins (2560) and chemicals (512).
WIDE, NARROW = 6, 3
WIDE_NODES = ["N0", "N1", "N2", "N3"]
NARROW_NODES = ["N5", "N6"]


def write_npz(path: Path, unknown: bool = False, aliases: bool = False) -> Path:
    generator = np.random.default_rng(0)
    wide = list(WIDE_NODES) + (["NOT_A_NODE"] if unknown else [])
    arrays = {
        "protein_names": np.array(wide, dtype=object),
        "protein_embeddings":
            generator.normal(size=(len(wide), WIDE), scale=8.0).astype(np.float32),
        "chemical_names": np.array(NARROW_NODES, dtype=object),
        "chemical_embeddings":
            generator.normal(size=(len(NARROW_NODES), NARROW)).astype(np.float32),
    }
    if aliases:
        # As embedding_pc emits them: the same rows under another ID namespace. The
        # two namespaces disagree about what "N7" means — which is the real situation
        # (HGNC:5 and Entrez 5 are different genes) and why the caller must choose.
        arrays["protein_alias_hgnc_id_names"] = np.array(["N7", "N8"], dtype=object)
        arrays["protein_alias_hgnc_id_rows"] = np.array([0, 1], dtype=np.int64)
        arrays["protein_alias_entrez_id_names"] = np.array(["N7", "N1"], dtype=object)
        arrays["protein_alias_entrez_id_rows"] = np.array([1, 2], dtype=np.int64)
    np.savez(path, **arrays)
    return path


def write_json(path: Path, with_meta: bool) -> Path:
    generator = np.random.default_rng(0)
    vectors = {name: generator.normal(size=WIDE).round(5).tolist() for name in WIDE_NODES}
    vectors.update({name: generator.normal(size=NARROW).round(5).tolist() for name in NARROW_NODES})
    path.write_text(json.dumps(vectors), encoding="utf-8")
    if with_meta:
        node_type = {name: "protein" for name in WIDE_NODES}
        node_type.update({name: "chemical" for name in NARROW_NODES})
        path.with_suffix("").with_suffix(".meta.json").write_text(
            json.dumps({"node_type": node_type}), encoding="utf-8"
        )
    return path


def test_settings_shorthand_and_validation() -> None:
    assert settings(None) is None
    assert settings("table.npz") == {
        "path": "table.npz", "normalize": "l2", "combine": "add", "aliases": [],
        "init_std": 0.01, "bias": True,
    }
    # One namespace may be given as a bare string.
    assert settings({"path": "t.npz", "aliases": "hgnc_id"})["aliases"] == ["hgnc_id"]
    # `replace` starts the adapted row at the learned rows' own scale instead.
    assert settings({"path": "t.npz", "combine": "replace"})["init_std"] == 0.1
    # A recorded spec round-trips: its extra keys are informational.
    assert settings({"path": "t.npz", "normalize": "standardize", "combine": "replace",
                     "aliases": ["hgnc_id"], "init_std": 0.2, "bias": False,
                     "num_covered": 3}) == {
        "path": "t.npz", "normalize": "standardize", "combine": "replace",
        "aliases": ["hgnc_id"], "init_std": 0.2, "bias": False,
    }
    with pytest.raises(KeyError, match="needs a `path:`"):
        settings({"normalize": "l2"})
    with pytest.raises(ValueError, match="normalize"):
        settings({"path": "t.npz", "normalize": "whiten"})
    with pytest.raises(ValueError, match="combine"):
        settings({"path": "t.npz", "combine": "concat"})


def test_npz_and_json_tables_agree(tmp_path: Path, dataset: GraphDataset) -> None:
    """Both readers produce the same groups; the JSON one uses its meta file."""
    names = dataset.node_names()
    from_npz = load_node_embeddings(str(write_npz(tmp_path / "t.npz")), names)
    from_json = load_node_embeddings(str(write_json(tmp_path / "t.json", with_meta=True)), names)
    assert [group.name for group in from_npz.groups] == ["chemical", "protein"]
    assert [group.name for group in from_json.groups] == ["chemical", "protein"]
    assert from_npz.spec["groups"] == {
        "chemical": {"dim": NARROW, "num_nodes": len(NARROW_NODES)},
        "protein": {"dim": WIDE, "num_nodes": len(WIDE_NODES)},
    }
    assert from_npz.num_covered == len(WIDE_NODES) + len(NARROW_NODES)
    assert from_npz.spec["coverage"] == round(from_npz.num_covered / dataset.num_nodes, 6)
    # Without the meta file the rows are grouped by width instead.
    by_width = load_node_embeddings(str(write_json(tmp_path / "w.json", with_meta=False)), names)
    assert [group.name for group in by_width.groups] == [f"dim{NARROW}", f"dim{WIDE}"]


def test_only_named_nodes_are_covered(tmp_path: Path, dataset: GraphDataset) -> None:
    table = load_node_embeddings(
        str(write_npz(tmp_path / "t.npz", unknown=True)), dataset.node_names()
    )
    protein = next(group for group in table.groups if group.name == "protein")
    # "NOT_A_NODE" is dropped; the graph's own nodes come out in node-id order.
    assert protein.nodes.tolist() == [0, 1, 2, 3]
    assert table.num_named == len(WIDE_NODES) + 1 + len(NARROW_NODES)
    with pytest.raises(ValueError, match="names none of this dataset"):
        load_node_embeddings(str(write_npz(tmp_path / "t.npz")), ["X", "Y"])
    with pytest.raises(FileNotFoundError, match="build_node_embeddings"):
        load_node_embeddings(str(tmp_path / "missing.npz"), dataset.node_names())
    with pytest.raises(KeyError, match="no matching vectors"):
        np.savez(tmp_path / "bad.npz", protein_names=np.array(["N0"], dtype=object))
        read_table(tmp_path / "bad.npz")


def test_aliases_are_opt_in_and_never_guessed(tmp_path: Path, dataset: GraphDataset) -> None:
    """A second ID namespace covers nodes the row names miss — once asked for.

    The two namespaces in the fixture disagree about what "N7" means, which is the
    real situation (HGNC:5 and Entrez 5 are different genes), so the config picks.
    """
    path = str(write_npz(tmp_path / "t.npz", aliases=True))
    names = dataset.node_names()

    plain = load_node_embeddings(path, names)
    assert plain.num_covered == 6 and plain.num_aliased == 0  # N0..N3, N5, N6

    hgnc = load_node_embeddings({"path": path, "aliases": ["hgnc_id"]}, names)
    protein = next(group for group in hgnc.groups if group.name == "protein")
    # N7 and N8 join, pointing at the rows N0 and N1 already use.
    assert protein.nodes.tolist() == [0, 1, 2, 3, 7, 8]
    assert hgnc.num_aliased == 2 and hgnc.spec["aliases"] == ["hgnc_id"]
    assert np.allclose(protein.vectors[4], protein.vectors[0])  # N7 shares N0's row

    # The other namespace sends N7 to a different row, and its "N1" alias loses to
    # the row that already carries that name.
    entrez = load_node_embeddings({"path": path, "aliases": ["entrez_id"]}, names)
    other = next(group for group in entrez.groups if group.name == "protein")
    assert other.nodes.tolist() == [0, 1, 2, 3, 7]
    assert np.allclose(other.vectors[4], other.vectors[1])
    assert np.allclose(other.vectors[1], hgnc.groups[1].vectors[1])  # N1 unchanged

    with pytest.raises(ValueError, match="no alias namespace"):
        load_node_embeddings({"path": path, "aliases": ["ensembl_gene_id"]}, names)
    # A table without aliases says so rather than silently covering nothing extra.
    with pytest.raises(ValueError, match="offers none"):
        load_node_embeddings(
            {"path": str(write_npz(tmp_path / "plain.npz")), "aliases": ["hgnc_id"]}, names
        )


def test_json_tables_carry_aliases_in_their_meta(tmp_path: Path, dataset: GraphDataset) -> None:
    path = write_json(tmp_path / "t.json", with_meta=True)
    meta_path = path.with_suffix("").with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["alias_to_name"] = {"hgnc_id": {"N7": "N0"}}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    table = load_node_embeddings({"path": str(path), "aliases": ["hgnc_id"]}, dataset.node_names())
    protein = next(group for group in table.groups if group.name == "protein")
    assert protein.nodes.tolist() == [0, 1, 2, 3, 7]
    assert table.num_aliased == 1


def test_normalization_keeps_what_it_claims(tmp_path: Path, dataset: GraphDataset) -> None:
    path = str(write_npz(tmp_path / "t.npz"))
    names = dataset.node_names()
    unit = load_node_embeddings(path, names)  # l2 is the default
    assert np.allclose(np.linalg.norm(unit.groups[1].vectors, axis=1), 1.0, atol=1e-5)
    standardized = load_node_embeddings(
        {"path": path, "normalize": "standardize"}, names
    ).groups[1].vectors
    assert np.allclose(standardized.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(standardized.std(axis=0), 1.0, atol=1e-4)
    raw = load_node_embeddings({"path": path, "normalize": "none"}, names)
    assert raw.groups[1].vectors.std() > 4.0  # scale 8.0, left alone


def _encoder(dataset: GraphDataset, table, seed: int = 0) -> RelationalGIN:
    torch.manual_seed(seed)
    return RelationalGIN(
        dataset.num_nodes, dataset.num_relations, hidden_dim=4, num_layers=1, dropout=0,
        external=table,
    )


COVERED = sorted([0, 1, 2, 3] + [5, 6])


@pytest.mark.parametrize("combine", ["add", "replace"])
def test_only_covered_nodes_change_and_the_rest_stay_learned(
    tmp_path: Path, dataset: GraphDataset, combine: str
) -> None:
    table = load_node_embeddings(
        {"path": str(write_npz(tmp_path / "t.npz")), "combine": combine}, dataset.node_names()
    )
    encoder = _encoder(dataset, table)
    matrix = encoder.node_embedding_matrix()
    assert matrix.shape == (dataset.num_nodes, 4)
    learned = encoder.embedding.weight
    adapted = {
        int(node): encoder.external.adapters[group](
            encoder.external.get_buffer(f"vectors_{group}")
        )[position]
        for group in ("protein", "chemical")
        for position, node in enumerate(encoder.external.get_buffer(f"nodes_{group}"))
    }
    for node in range(dataset.num_nodes):
        if node not in COVERED:
            assert torch.equal(matrix[node], learned[node]), node
            continue
        expected = adapted[node] + (learned[node] if combine == "add" else 0.0)
        assert torch.allclose(matrix[node], expected, atol=1e-6), node
        assert not torch.allclose(matrix[node], learned[node]), node
    # embed_nodes is the same matrix, gathered — that is what partition mode uses.
    nodes = torch.tensor([0, 5, 9])
    assert torch.allclose(encoder.embed_nodes(nodes), matrix[nodes])
    # The adapters learn; the vectors do not, and are not in the checkpoint.
    keys = [name for name in encoder.state_dict() if name.startswith("external")]
    assert keys == [
        "external.adapters.chemical.weight", "external.adapters.chemical.bias",
        "external.adapters.protein.weight", "external.adapters.protein.bias",
    ]
    encoder(*dataset.graph()).sum().backward()
    assert encoder.external.adapters["protein"].weight.grad.abs().sum() > 0
    # nn.Embedding stays a used parameter (DDP requires that): every row under
    # `add`, and the uncovered rows under `replace`.
    grad = encoder.embedding.weight.grad
    assert grad is not None and float(grad[4].abs().sum()) > 0.0
    assert (float(grad[COVERED].abs().sum()) == 0.0) is (combine == "replace")


def test_the_adapted_rows_start_at_the_configured_scale(
    tmp_path: Path, dataset: GraphDataset
) -> None:
    """The init is scale-aware, so `normalize` does not change how loud it starts."""
    path = str(write_npz(tmp_path / "t.npz"))
    for normalize in ("l2", "standardize", "none"):
        table = load_node_embeddings(
            {"path": path, "normalize": normalize, "combine": "replace"}, dataset.node_names()
        )
        encoder = _encoder(dataset, table)
        adapted = encoder.node_embedding_matrix()[COVERED]
        # 0.1 per coordinate, the same std nn.Embedding's rows are initialised at;
        # 24 rows x 4 columns is a small sample, hence the loose tolerance.
        assert 0.03 < float(adapted.std()) < 0.3, normalize


def test_the_adapters_are_the_only_addition(tmp_path: Path, dataset: GraphDataset) -> None:
    """Everything but the adapters keeps the initialisation it had without them.

    The table is read and the adapters built after ``reset_parameters``, so a run
    without ``node_embeddings:`` draws exactly the random numbers it drew before
    the option existed, and a run with them is a clean ablation of the same encoder.
    """
    plain = _encoder(dataset, None)
    table = load_node_embeddings(str(write_npz(tmp_path / "t.npz")), dataset.node_names())
    external = _encoder(dataset, table)
    assert plain.external is None and not [k for k in plain.state_dict() if "external" in k]
    for name, weight in plain.state_dict().items():
        assert torch.equal(weight, external.state_dict()[name]), name
    assert encoder_config(plain, 0.0).keys() == {
        "num_nodes", "num_relations", "hidden_dim", "num_layers", "dropout"
    }
    assert encoder_config(external, 0.0)["node_embeddings"]["path"].endswith("t.npz")


def _pretrain(tmp_path: Path, dataset: GraphDataset, table_path: str, **model) -> Path:
    from pathwaygnn.training.pretrain import run_pretraining

    output = tmp_path / f"pretrain_{len(model)}"
    run_pretraining({
        "seed": 1,
        "device": "cpu",
        "dataset": {"name": dataset.name, "dir": str(dataset.root)},
        "model": {
            "hidden_dim": 4, "num_layers": 1, "dropout": 0.0,
            "node_embeddings": {"path": table_path, **model},
        },
        "training": {"epochs": 1, "steps_per_epoch": 1, "batch_size": 8},
        "output_dir": str(output),
    })
    return output / "best.pt"


def test_pretrain_records_the_table_and_load_encoder_rebuilds_it(
    tmp_path: Path, dataset: GraphDataset
) -> None:
    path = str(write_npz(tmp_path / "t.npz"))
    checkpoint_path = _pretrain(tmp_path, dataset, path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec = checkpoint["model_config"]["node_embeddings"]
    assert spec["path"] == path and spec["num_covered"] == 6

    names = dataset.node_names()
    encoder, _ = load_encoder(checkpoint_path, dataset.num_nodes, dataset.num_relations,
                              node_names=names)
    assert encoder.external is not None
    # The vectors are re-read, so the rebuilt encoder reproduces the saved one:
    # node 0 is a covered protein, and under the default `add` its row is the
    # learned row plus the adapted vector.
    assert torch.allclose(
        encoder.node_embedding_matrix()[0],
        encoder.embedding.weight[0]
        + encoder.external.adapters["protein"](encoder.external.get_buffer("vectors_protein"))[0],
        atol=1e-6,
    )
    # ... and the file may be pointed at from elsewhere, as long as it matches.
    moved = write_npz(tmp_path / "moved.npz")
    load_encoder(checkpoint_path, dataset.num_nodes, dataset.num_relations,
                 node_names=names, node_embeddings=str(moved))

    with pytest.raises(ValueError, match="matched by node name"):
        load_encoder(checkpoint_path, dataset.num_nodes, dataset.num_relations)
    with pytest.raises(ValueError, match="combine="):
        load_encoder(checkpoint_path, dataset.num_nodes, dataset.num_relations,
                     node_names=names,
                     node_embeddings={"path": path, "combine": "replace"})
    with pytest.raises(ValueError, match="adapter groups"):
        wrong = tmp_path / "wrong.npz"
        np.savez(wrong, names=np.array(WIDE_NODES, dtype=object),
                 embeddings=np.zeros((len(WIDE_NODES), WIDE + 1), dtype=np.float32))
        load_encoder(checkpoint_path, dataset.num_nodes, dataset.num_relations,
                     node_names=names, node_embeddings=str(wrong))


def test_a_plain_checkpoint_refuses_an_external_table(
    tmp_path: Path, dataset: GraphDataset, pretrained: Path
) -> None:
    with pytest.raises(ValueError, match="pre-trained without external node embeddings"):
        load_encoder(pretrained, dataset.num_nodes, dataset.num_relations,
                     node_names=dataset.node_names(),
                     node_embeddings=str(write_npz(tmp_path / "t.npz")))


def test_partition_mode_feeds_the_adapters_too(tmp_path: Path, dataset: GraphDataset) -> None:
    """A partition step embeds only its own nodes, adapted ones included."""
    from pathwaygnn.training.pretrain import run_pretraining

    path = str(write_npz(tmp_path / "t.npz"))
    output = tmp_path / "pretrain_partitioned"
    run_pretraining({
        "seed": 5,
        "device": "cpu",
        "dataset": {"name": dataset.name, "dir": str(dataset.root)},
        "model": {"hidden_dim": 4, "num_layers": 1, "dropout": 0.0,
                  "node_embeddings": {"path": path}},
        "training": {
            "epochs": 1, "batch_size": 8,
            "partition": {"dir": str(tmp_path / "parts"), "num_parts": 4, "parts_per_batch": 2},
        },
        "output_dir": str(output),
    })
    encoder, _ = load_encoder(output / "best.pt", dataset.num_nodes, dataset.num_relations,
                              node_names=dataset.node_names())
    assert encoder.external is not None
    # The adapters moved, so the subgraph steps did flow gradient through them.
    fresh = _encoder(dataset, load_node_embeddings(path, dataset.node_names()), seed=5)
    assert not torch.equal(
        encoder.external.adapters["protein"].weight,
        fresh.external.adapters["protein"].weight,
    )


def test_cv_ig_and_pred_run_on_an_external_encoder(tmp_path: Path, dataset: GraphDataset) -> None:
    """The consuming commands need no `node_embeddings:` of their own."""
    from pathwaygnn.training.cv import run_cv
    from pathwaygnn.training.ig import run_ig
    from pathwaygnn.training.predict import run_prediction

    path = str(write_npz(tmp_path / "t.npz"))
    checkpoint_path = _pretrain(tmp_path, dataset, path)
    run_dir = tmp_path / "cv"
    cfg = {
        "seed": 5,
        "device": "cpu",
        "folds": 2,
        "dataset": {"name": dataset.name, "dir": str(dataset.root), "tasks": ["main"]},
        "pretrained_checkpoint": str(checkpoint_path),
        "model": {"embedding_dim": 4, "hidden_dim": 4, "dropout": 0.0,
                  "node_embeddings": {"path": path}},
        "training": {"epochs": 1, "batch_size": 8, "end_to_end": True, "resume": False},
        "variants": [{"name": "gnn", "use_graph": True}],
        "output_dir": str(run_dir),
    }
    run_cv(cfg)
    fold = run_dir / "main" / "gnn" / "fold_0" / "model.pt"
    assert fold.is_file()
    ig_output = tmp_path / "ig"
    result = run_ig({
        **cfg,
        "dataset": {"name": dataset.name, "dir": str(dataset.root), "task": "main"},
        "run_dir": str(run_dir), "variant": "gnn", "fold": 0, "steps": 2, "max_samples": 2,
        "output_dir": str(ig_output),
    })
    # Attribution is over the matrix the graph convolves, adapted rows included.
    assert result["num_samples"] == 2
    score = np.load(ig_output / "attributions.npz")["graph_score"]
    assert score.shape == (dataset.num_nodes,) and np.isfinite(score).all()

    # `pred` rebuilds the same encoder from the checkpoint's recorded table: its
    # probabilities must match the fold's stored held-out ones.
    pred_output = tmp_path / "pred"
    run_prediction({
        "device": "cpu",
        "dataset": {"name": dataset.name, "dir": str(dataset.root), "task": "main"},
        "checkpoint": str(fold),
        "pretrained_checkpoint": str(checkpoint_path),
        "batch_size": 8,
        "output_dir": str(pred_output),
    })
    stored = np.load(run_dir / "main" / "gnn" / "fold_0" / "predictions.npz")
    scored = {}
    with (pred_output / "predictions.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            scored[int(row["sample_index"])] = float(row["probability"])
    for sample, probability in zip(stored["sample_index"], stored["probability"]):
        assert scored[int(sample)] == pytest.approx(float(probability), abs=1e-6)
