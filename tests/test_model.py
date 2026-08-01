import pytest
import torch
from torch import nn

from pathwaygnn.data.format import GraphDataset
from pathwaygnn.data.samples import ChannelBatch, SampleBatch, TaskDataset
from pathwaygnn.models.encoder import RelationalGIN, encoder_config, load_encoder
from pathwaygnn.models.predictor import SampleLevelModel, build_model


def _batch(dataset: GraphDataset, size: int = 4) -> SampleBatch:
    data = TaskDataset(dataset.task("main"))
    return data.collate()([data[index] for index in range(size)])


@pytest.mark.parametrize("block", ["plain", "paper"])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("use_covariates", [False, True])
def test_forward_and_backward(dataset: GraphDataset, block, use_graph, use_covariates) -> None:
    task = dataset.task("main")
    encoder = RelationalGIN(dataset.num_nodes, dataset.num_relations, hidden_dim=4, dropout=0)
    embeddings = encoder(*dataset.graph())
    model = SampleLevelModel(
        channels=task.channel_names,
        embedding_dim=4,
        hidden_dim=5,
        covariate_dim=task.covariate_dim,
        use_graph=use_graph,
        use_covariates=use_covariates,
        batch_norm=block == "paper",
        dropout=0.0 if block == "paper" else 0.1,
        block=block,
    )
    logits = model(_batch(dataset), embeddings if use_graph else None)
    assert logits.shape == (4,)
    logits.sum().backward()
    assert model.output[0].weight.grad is not None


def test_graph_is_required_when_declared(dataset: GraphDataset) -> None:
    task = dataset.task("main")
    model = SampleLevelModel(task.channel_names, 4, covariate_dim=task.covariate_dim)
    with pytest.raises(ValueError, match="node_embeddings are required"):
        model(_batch(dataset))
    with pytest.raises(ValueError, match="requires a task with covariates"):
        SampleLevelModel(task.channel_names, 4, covariate_dim=0, use_covariates=True)
    with pytest.raises(ValueError, match="block must be one of"):
        SampleLevelModel(task.channel_names, 4, block="fancy")


def test_dense_and_sparse_channels_agree(dataset: GraphDataset) -> None:
    """The same values fed as a dense row or as sparse triples must score alike."""
    torch.manual_seed(0)
    values = torch.tensor([[0.5, -1.5, 2.0], [1.0, 0.0, -0.5]])
    genes = torch.tensor([1, 4, 7])
    dense = SampleBatch(
        channels={"c": ChannelBatch("dense", genes, values)},
        label=torch.zeros(2),
        index=torch.arange(2),
    )
    sparse = SampleBatch(
        channels={"c": ChannelBatch(
            "sparse",
            genes.repeat(2),
            values.reshape(-1, 1),
            torch.tensor([0, 0, 0, 1, 1, 1]),
        )},
        label=torch.zeros(2),
        index=torch.arange(2),
    )
    embeddings = torch.randn(dataset.num_nodes, 4)
    model = SampleLevelModel(["c"], embedding_dim=4, hidden_dim=4, use_graph=True).eval()
    assert torch.allclose(model(dense, embeddings), model(sparse, embeddings), atol=1e-6)


def test_config_round_trip_and_module_order(dataset: GraphDataset) -> None:
    task = dataset.task("main")
    model = build_model(
        task.channel_names, task.covariate_dim, 4,
        {"hidden_dim": 5, "batch_norm": True, "block": "paper"},
        use_graph=True, use_covariates=True,
    )
    clone = SampleLevelModel.from_config(model.config)
    assert clone.config == model.config
    clone.load_state_dict(model.state_dict())
    # The parameter order fixes the initialisation draws, so keep it pinned:
    # value projection, then every channel's gene block, then their aggregation
    # blocks, then the covariate branch, then the head.
    assert [name for name, _ in model.named_parameters() if name.endswith("0.weight")] == [
        "value_projection.0.weight",
        "gene_blocks.0.0.weight",
        "gene_blocks.1.0.weight",
        "aggregate_blocks.0.0.weight",
        "aggregate_blocks.1.0.weight",
        "covariate_block.0.weight",
        "output.0.weight",
    ]


def test_paper_block_layout() -> None:
    """`paper` blocks end on an activation and batch norm; `plain` ones do not."""
    paper = SampleLevelModel(["c"], 4, batch_norm=True, block="paper", use_graph=False)
    plain = SampleLevelModel(["c"], 4, dropout=0.2, block="plain", use_graph=False)
    assert [type(m) for m in paper.gene_blocks[0]] == [
        nn.Linear, nn.ELU, nn.BatchNorm1d, nn.Linear, nn.ELU, nn.BatchNorm1d
    ]
    assert [type(m) for m in plain.gene_blocks[0]] == [nn.Linear, nn.ELU, nn.Dropout, nn.Linear]
    assert [type(m) for m in plain.output] == [nn.Linear, nn.ELU, nn.Dropout, nn.Linear]


def test_encoder_checkpoint_guard(dataset: GraphDataset, pretrained) -> None:
    encoder, checkpoint = load_encoder(pretrained, dataset.num_nodes, dataset.num_relations)
    assert checkpoint["dataset"] == dataset.name
    assert encoder_config(encoder, 0.0)["hidden_dim"] == encoder.hidden_dim
    overridden = encoder.forward_from_embedding(encoder.embedding.weight, *dataset.graph())
    assert torch.allclose(encoder(*dataset.graph()), overridden)
    with pytest.raises(ValueError, match="re-run pre-training on this dataset"):
        load_encoder(pretrained, dataset.num_nodes + 1, dataset.num_relations)
