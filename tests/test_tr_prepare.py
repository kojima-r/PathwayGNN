import csv
import json
from pathlib import Path

import numpy as np
import torch

from pathwaygnn.data.format import GraphDataset
from pathwaygnn.data.samples import TaskDataset
from pathwaygnn.models.encoder import GraphPretrainer, RelationalGIN
from pathwaygnn.models.predictor import SampleLevelModel
from pathwaygnn.training.finetune import stratified_split
from pathwaygnn.training.metrics import binary_auc
from pathwaygnn_datasets.tr.prepare import prepare_tr_dataset


def _write(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows(rows)


def make_raw(root: Path) -> None:
    _write(
        root / "PathwayCommons12.All.hgnc.sif",
        [["A", "activates", "B"], ["B", "binds", "C"]],
    )
    _write(
        root / "disease_specific_signature.tsv",
        [
            ["do_id", "human_gene_name", "expression"],
            ["D1", "A", 1.0],
            ["D1", "B", -0.5],
            ["D2", "C", 0.8],
            ["D2", "UNKNOWN", 0.9],
        ],
    )
    signature = [
        ["pert_iname", "A", "B", "C", "OUT"],
        ["P1", 1, 0, -1, 2],
        ["P2", 0, 1, 0.5, 2],
    ]
    labels = [
        ["gene", "doid", "label"],
        ["P1", "D1", 1],
        ["P1", "D2", 0],
        ["P2", "D1", 0],
        ["P2", "D2", 1],
        ["MISSING", "D1", 1],
    ]
    for name in ("knockdown_signature_sample.tsv", "overexpression_signature_sample.tsv"):
        _write(root / name, signature)
    for name in ("inhibitory_target_disease.tsv", "activatory_target_disease.tsv"):
        _write(root / name, labels)


def test_prepare_writes_a_generic_dataset(tmp_path: Path) -> None:
    raw, prepared = tmp_path / "raw", tmp_path / "prepared"
    raw.mkdir()
    make_raw(raw)
    manifest = prepare_tr_dataset(raw, prepared)
    assert manifest["name"] == "tr"
    assert (manifest["num_nodes"], manifest["num_relations"]) == (3, 2)
    assert manifest["num_edges"] == 4  # both directions of both edges
    assert manifest["source"]["disease_rows_skipped"] == 1
    assert sorted(manifest["channels"]) == ["disease", "perturbation_kd", "perturbation_oe"]
    assert manifest["tasks"] == ["kd_inh", "oe_act"]

    dataset = GraphDataset.open(prepared, "tr")
    assert dataset.node_names() == ["A", "B", "C"]
    assert dataset.relation_names() == ["activates", "binds"]
    task = dataset.task("kd_inh")
    # Both tasks expose the same aliases, so a model config transfers between them.
    assert task.channel_names == dataset.task("oe_act").channel_names == ("perturbation", "disease")
    assert [channel.source for channel in task.channels] == ["perturbation_kd", "disease"]
    assert task.num_samples == 4 and task.manifest["source"]["label_rows_skipped"] == 1
    assert task.group_names == ("D1", "D2")
    assert task.covariate_dim == 0
    assert task.seed_offset == 0 and dataset.task("oe_act").seed_offset == 1
    assert json.loads((prepared / "tasks/kd_inh/task.json").read_text())["num_positive"] == 2


def test_prepared_samples_feed_the_models(tmp_path: Path) -> None:
    raw, prepared = tmp_path / "raw", tmp_path / "prepared"
    raw.mkdir()
    make_raw(raw)
    prepare_tr_dataset(raw, prepared)
    dataset = GraphDataset.open(prepared, "tr")
    edge_index, edge_type = dataset.graph()
    encoder = RelationalGIN(dataset.num_nodes, dataset.num_relations, hidden_dim=8, dropout=0)
    pretrainer = GraphPretrainer(encoder)
    positive = edge_index[:, :2]
    scores = pretrainer(edge_index, edge_type, positive, edge_type[:2], positive.flip(0))
    assert scores[0].shape == scores[1].shape == (2,)

    task = dataset.task("kd_inh")
    data = TaskDataset(task)
    batch = data.collate()([data[0], data[1]])
    # The disease signature is shared, the perturbation table is per task.
    assert batch.channels["perturbation"].kind == "sparse"
    assert batch.covariate is None
    logits = SampleLevelModel(task.channel_names, embedding_dim=8, hidden_dim=8)(
        batch, encoder(edge_index, edge_type)
    )
    assert logits.shape == (2,)
    logits.sum().backward()


def test_metrics_and_stratified_split() -> None:
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
    train, valid, test = stratified_split(target, (0.5, 0.25, 0.25), 42)
    assert [len(train), len(valid), len(test)] == [4, 2, 2]
    assert target[train].sum() == 2
    assert binary_auc(torch.tensor([0, 1]), torch.tensor([0.1, 0.9])) == 1.0
