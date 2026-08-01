# data_cancer — TCGA cancer prognosis (Inoue et al.)

- `processed/` — the **upstream** bundle as published with the paper's code
  (`graph.tsv`, `vertices_dic.tsv`, `relationships_dic.tsv`,
  `<n>years_labels.tsv`, `<n>years_sample.tsv`, `<n>years_node_input.tsv`).
  It keeps the original name and is the *input* to preprocessing.
- `rawdata_TCGA/`, `PathwayCommons13.All.hgnc.txt` — additional source material.
- `prepared/` — generated. The generic dataset that `pathwaygnn` reads.

```bash
conda activate gnn
bash scripts/cancer/reproduce_paper.sh prepare   # pathwaygnn-data cancer-prepare
```

Preprocessing streams the three-column node-input TSVs (4.2 GB total, about 1.5
minutes) into memmappable matrices under `prepared/channels/expression_<n>year/`,
one task per verification year under `prepared/tasks/<n>year/`, and validates the
per-year sample counts against the published Supplementary Table 1. Re-running it
reuses an existing matrix whose shape already matches.

`artifacts/` is the pre-refactor layout and is no longer read; `prepared/`
supersedes it and can be deleted once regenerated.
