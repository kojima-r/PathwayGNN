# data_tr — target repositioning

`raw/` contains the source files copied without modification from
`SLGCN-TR/data/raw`. These files are versioned with the project.

Preprocessing is a separate step from training. Run it from the repository root:

```bash
conda activate gnn
bash scripts/tr/prepare.sh          # pathwaygnn-data tr-prepare --config configs/tr/prepare.yaml
```

That writes `prepared/`, the generic dataset every `pathwaygnn` command reads
(`dataset.json`, `graph.pt`, `nodes.json`, `relations.json`, `channels/`,
`tasks/`; see `src/pathwaygnn/data/format.py`). `prepared/` is excluded from Git
because it is fully reproducible from `raw/`; `prepared/dataset.json` and
`prepared/tasks/*/task.json` record input statistics and filtered-row counts.

Files used for training:

- `PathwayCommons12.All.hgnc.sif`
- `disease_specific_signature.tsv`
- `knockdown_signature_sample.tsv`
- `overexpression_signature_sample.tsv`
- `inhibitory_target_disease.tsv`
- `activatory_target_disease.tsv`

`推論用データ/` and `Target_repositioning_データ完全版説明.txt` are retained as
part of the original raw-data bundle, although the current training
preprocessor does not consume them.

Integrity can be checked with:

```bash
(cd data_tr/raw && sha256sum -c SHA256SUMS)
```
