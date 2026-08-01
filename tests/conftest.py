from pathlib import Path

import numpy as np
import pytest
import torch

from pathwaygnn.data.format import DatasetWriter, GraphDataset, TaskChannel

NUM_NODES = 24
NUM_RELATIONS = 2
NUM_SAMPLES = 24
NUM_GENES = 6
GROUPS = ["g0", "g1", "g2"]
COVARIATES = ["c0", "c1", "c2"]


def build_dataset(root: Path, name: str = "toy") -> GraphDataset:
    """A tiny prepared dataset with one sparse and one dense channel."""
    generator = np.random.default_rng(0)
    source, target, relation = [], [], []
    for node in range(NUM_NODES):
        for step in (1, 2):
            source.append(node)
            target.append((node + step) % NUM_NODES)
            relation.append(step - 1)
    # A hub so that degree centrality varies; a regular graph would make the
    # degree/attribution correlation undefined.
    for node in range(3, 9):
        source.append(0)
        target.append(node)
        relation.append(1)
    writer = DatasetWriter(root, name, source={"synthetic": True})
    writer.write_graph(
        torch.tensor([source + target, target + source], dtype=torch.long),
        torch.tensor(relation + relation, dtype=torch.long),
        [f"N{index}" for index in range(NUM_NODES)],
        [f"r{index}" for index in range(NUM_RELATIONS)],
    )

    rows = 6
    ptr, gene, value = [0], [], []
    for row in range(rows):
        for column in range(3):
            gene.append((row * 3 + column) % NUM_NODES)
            value.append(float(generator.normal()))
        ptr.append(len(gene))
    writer.sparse_channel("signature", ptr, gene, value)

    channel_dir = writer.dense_channel("expression", NUM_SAMPLES, NUM_GENES)
    np.save(
        channel_dir / "matrix.npy",
        generator.normal(size=(NUM_SAMPLES, NUM_GENES)).astype(np.float32),
    )
    np.save(channel_dir / "gene_index.npy", np.arange(NUM_GENES, dtype=np.int64))

    labels = np.tile([0.0, 1.0], NUM_SAMPLES // 2).astype(np.float32)
    writer.write_task(
        "main",
        labels,
        channels={
            "expression": TaskChannel("expression"),
            "signature": TaskChannel("signature", np.arange(NUM_SAMPLES) % rows),
        },
        covariates=np.eye(len(COVARIATES), dtype=np.float32)[
            np.arange(NUM_SAMPLES) % len(COVARIATES)
        ],
        covariate_names=COVARIATES,
        groups=np.arange(NUM_SAMPLES) % len(GROUPS),
        group_names=GROUPS,
        seed_offset=7,
    )
    writer.finish()
    return GraphDataset.open(root, name)


@pytest.fixture
def dataset(tmp_path: Path) -> GraphDataset:
    return build_dataset(tmp_path / "prepared")


@pytest.fixture
def pretrained(tmp_path: Path, dataset: GraphDataset) -> Path:
    from pathwaygnn.training.pretrain import run_pretraining

    output = tmp_path / "pretrain"
    run_pretraining({
        "seed": 1,
        "device": "cpu",
        "dataset": {"name": dataset.name, "dir": str(dataset.root)},
        "model": {"hidden_dim": 4, "num_layers": 1, "dropout": 0.0},
        "training": {"epochs": 1, "steps_per_epoch": 1, "batch_size": 8},
        "output_dir": str(output),
    })
    return output / "best.pt"
