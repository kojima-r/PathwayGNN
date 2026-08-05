"""The tensor shapes the model's docstrings promise, asserted.

Every dimension checked here is written down in a docstring or an inline comment in
`models/encoder.py`, `models/predictor.py`, `data/samples.py`, `data/partition.py`,
`data/format.py` or `training/pretrain.py`. The point is that those annotations cannot
quietly go stale: if a shape changes, this fails rather than the comment becoming a
lie. Symbols follow the modules' own: N nodes, R relations, E edges, H hidden, B
samples, G dense genes, V sparse values, S sample features, K scored edges.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from conftest import GROUPS, NUM_GENES, NUM_NODES, NUM_RELATIONS, SAMPLE_FEATURES
from pathwaygnn.data.format import GraphDataset
from pathwaygnn.data.partition import PartitionLoader, PartitionStore, write_partitions
from pathwaygnn.data.samples import TaskDataset
from pathwaygnn.models.encoder import GraphPretrainer, RelationalGIN
from pathwaygnn.models.predictor import SampleLevelModel, _scatter_sum
from pathwaygnn.training.pretrain import partition_edges

H = 6
B = 4
K = 7


def test_prepared_format_array_shapes(dataset: GraphDataset) -> None:
    edge_index, edge_type = dataset.graph()
    num_edges = int(dataset.manifest["num_edges"])
    assert edge_index.shape == (2, num_edges) and edge_index.dtype == torch.int64
    assert edge_type.shape == (num_edges,) and edge_type.dtype == torch.int64
    assert int(edge_type.max()) < NUM_RELATIONS
    assert int(edge_index.max()) < NUM_NODES

    task = dataset.task("main")
    assert task.labels().shape == (task.num_samples,)
    assert task.groups().shape == (task.num_samples,)
    assert task.sample_features().shape == (task.num_samples, len(SAMPLE_FEATURES))
    assert task.rows("signature").shape == (task.num_samples,)
    assert task.rows("expression") is None  # identity mapping, so no file

    dense = dataset.node_feature("expression")
    assert dense.matrix().shape == (dense.num_rows, NUM_GENES)
    assert dense.gene_index().shape == (NUM_GENES,)
    ptr, gene, value = dataset.node_feature("signature").csr()
    assert ptr.shape == (dataset.node_feature("signature").num_rows + 1,)
    assert gene.shape == value.shape == (int(ptr[-1]),)


def test_batch_shapes_match_the_two_feature_kinds(dataset: GraphDataset) -> None:
    data = TaskDataset(dataset.task("main"))
    batch = next(iter(DataLoader(data, batch_size=B, shuffle=False, collate_fn=data.collate())))
    assert batch.size == B
    assert batch.label.shape == (B,)
    assert batch.index.shape == (B,)
    assert batch.sample_feature.shape == (B, len(SAMPLE_FEATURES))
    assert batch.group.shape == (B,)
    assert int(batch.group.max()) < len(GROUPS)

    dense = batch.node_features["expression"]
    assert dense.dense and dense.sample is None
    assert dense.value.shape == (B, NUM_GENES)      # [B,G]
    assert dense.gene.shape == (NUM_GENES,)         # [G], shared by every sample
    sparse = batch.node_features["signature"]
    assert not sparse.dense
    values = sparse.value.shape[0]                  # V
    assert sparse.value.shape == (values, 1)        # [V,1]
    assert sparse.gene.shape == (values,)           # [V]
    assert sparse.sample.shape == (values,)         # [V]
    assert int(sparse.sample.max()) < B
    # Graph node ids, so both index the encoder's embedding rows directly.
    assert int(dense.gene.max()) < NUM_NODES and int(sparse.gene.max()) < NUM_NODES


def test_encoder_shapes(dataset: GraphDataset) -> None:
    encoder = RelationalGIN(NUM_NODES, NUM_RELATIONS, hidden_dim=H, num_layers=2)
    edge_index, edge_type = dataset.graph()
    assert encoder.embedding.weight.shape == (NUM_NODES, H)
    assert encoder(edge_index, edge_type).shape == (NUM_NODES, H)          # [N,H]
    # The readout consumes all L layers concatenated.
    assert encoder.readout.in_features == H * 2

    # forward_from_embedding takes whatever row set it is given: a subgraph's n rows.
    nodes = torch.arange(5)
    keep = torch.isin(edge_index[0], nodes) & torch.isin(edge_index[1], nodes)
    local = torch.searchsorted(nodes, edge_index[:, keep])
    out = encoder.forward_from_embedding(encoder.embedding(nodes), local, edge_type[keep])
    assert out.shape == (nodes.numel(), H)                                  # [n,H]


def test_pretrainer_scores_one_value_per_edge(dataset: GraphDataset) -> None:
    model = GraphPretrainer(RelationalGIN(NUM_NODES, NUM_RELATIONS, hidden_dim=H, num_layers=1))
    edge_index, edge_type = dataset.graph()
    assert model.relation.weight.shape == (NUM_RELATIONS, H)
    embeddings = model.encoder(edge_index, edge_type)
    edges = edge_index[:, :K]
    assert model.score(embeddings, edges, edge_type[:K]).shape == (K,)      # [K]
    positive, negative = model(edge_index, edge_type, edges, edge_type[:K], edges)
    assert positive.shape == negative.shape == (K,)


def test_head_shapes_and_the_scatter_helper(dataset: GraphDataset) -> None:
    task = dataset.task("main")
    data = TaskDataset(task)
    batch = next(iter(DataLoader(data, batch_size=B, shuffle=False, collate_fn=data.collate())))
    model = SampleLevelModel(
        node_features=task.node_feature_names,
        embedding_dim=H,
        hidden_dim=H,
        sample_feature_dim=task.sample_feature_dim,
        use_graph=True,
        use_sample_features=True,
    )
    node_embeddings = torch.randn(NUM_NODES, H)                             # [N,E]
    assert model(batch, node_embeddings).shape == (B,)                      # [B]
    # The graph-free variant takes the same batch and no embeddings.
    free = SampleLevelModel(
        node_features=task.node_feature_names, embedding_dim=H, hidden_dim=H, use_graph=False
    )
    assert free(batch).shape == (B,)

    values = torch.randn(11, H)                                            # [V,H]
    index = torch.randint(B, (11,))                                        # [V]
    assert _scatter_sum(values, index, B).shape == (B, H)                  # [B,H]


def test_partition_batch_and_sampled_edge_shapes(
    dataset: GraphDataset, tmp_path: Path
) -> None:
    write_partitions(dataset, tmp_path / "parts", 4)
    store = PartitionStore.open(tmp_path / "parts", dataset)
    loader = PartitionLoader(store, parts_per_batch=2, shuffle=False)
    batch = next(iter(loader))
    n, e = batch.num_nodes, batch.num_edges
    assert batch.nodes.shape == (n,)                                        # [n]
    assert batch.edge_index.shape == (2, e)                                 # [2,e]
    assert batch.edge_type.shape == (e,)                                    # [e]
    assert batch.part_ids.shape == (2,)                                     # [parts_per_batch]
    assert int(batch.edge_index.max()) < n                                  # local indices
    assert int(batch.nodes.max()) < NUM_NODES                               # global node ids

    generator = torch.Generator().manual_seed(0)
    positive, types, negative = partition_edges(
        batch, K, generator, torch.device("cpu"), False, NUM_RELATIONS
    )
    assert positive.shape == negative.shape == (2, K)                       # [2,K]
    assert types.shape == (K,)                                              # [K]
    # Balanced sampling draws K per relation *present in this subgraph*.
    present = len({int(value) for value in batch.edge_type})
    balanced, _, _ = partition_edges(
        batch, K, generator, torch.device("cpu"), True, NUM_RELATIONS
    )
    assert balanced.shape == (2, K * present)

    # A partition batch feeds the model through the `nodes` argument.
    model = GraphPretrainer(RelationalGIN(NUM_NODES, NUM_RELATIONS, hidden_dim=H, num_layers=1))
    scores = model(batch.edge_index, batch.edge_type, positive, types, negative, batch.nodes)
    assert scores[0].shape == scores[1].shape == (K,)


def test_documented_dtypes_hold(dataset: GraphDataset) -> None:
    """Indices are int64 and values float32 throughout, as the annotations say."""
    task = dataset.task("main")
    assert task.labels().dtype == np.float32
    assert task.groups().dtype == np.int64
    assert task.sample_features().dtype == np.float32
    assert dataset.node_feature("expression").matrix().dtype == np.float32
    assert dataset.node_feature("expression").gene_index().dtype == np.int64
    data = TaskDataset(task)
    batch = next(iter(DataLoader(data, batch_size=B, shuffle=False, collate_fn=data.collate())))
    assert batch.label.dtype == torch.float32 and batch.index.dtype == torch.int64
    for node_feature in batch.node_features.values():
        assert node_feature.value.dtype == torch.float32
        assert node_feature.gene.dtype == torch.int64
