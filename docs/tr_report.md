# PathwayGNN target repositioning report

## What this report covers

Dataset **tr** at `/data1/kojima/PathwayGNN/PathwayGNN/data_tr/raw`, prepared
into `data_tr/prepared`: 30,918 graph nodes, 3,673,654 directed
edges, 13 relation types, tasks kd_inh, oe_act. Run status: 4 cross-validation conditions, 2 holdout runs, 2 baseline runs, 2 attribution runs.
Graph pre-training: 100 epochs, final DistMult loss 0.9473, final pairwise accuracy 0.8818.

Every number below comes from artifacts under `outputs/tr/`, and every table is also written as TSV
under `outputs/tr/report/`. Cross-validation and the graph-free baselines use the same stratified 5-fold
split (seed 42, `StratifiedKFold(shuffle=True)`), so those model comparisons are on identical folds;
attribution runs on fold 0 of the graph variant, and holdout fine-tuning uses its own 70/15/15 split.

## Dataset audit

| task | samples | positive | positive_ratio | perturbations | diseases_used | diseases_total | mean_genes_perturbation | mean_genes_disease | signature_genes_skipped | label_rows_skipped |
|---|---|---|---|---|---|---|---|---|---|---|
| kd_inh | 6944 | 567 | 0.0817 | 4345 | 31 | 235 | 958 | 1017 | 19 | 224 |
| oe_act | 450 | 37 | 0.0822 | 3114 | 15 | 235 | 958 | 1017 | 19 | 0 |

`diseases_used` counts the diseases that actually appear in a task's labels, out of
`diseases_total` in the shared disease channel. The `mean_genes_*` columns are the mean number of
non-zero genes per row of each channel after the 1e-7 cutoff.

## Cross-validation (`pathwaygnn cv`)

| task | variant | uses_graph | mean_auc | std_auc | min_fold_auc | max_fold_auc | pooled_auc | folds |
|---|---|---|---|---|---|---|---|---|
| kd_inh | mlp | False | 0.5000 | 0.0334 | 0.4353 | 0.5295 | 0.5127 | 5 |
| kd_inh | gnn_mlp | True | 0.5059 | 0.0305 | 0.4708 | 0.5469 | 0.5116 | 5 |
| oe_act | mlp | False | 0.7045 | 0.1090 | 0.5577 | 0.8689 | 0.6886 | 5 |
| oe_act | gnn_mlp | True | 0.6986 | 0.1421 | 0.4733 | 0.8659 | 0.6756 | 5 |

`pooled_auc` is computed once over the concatenated held-out predictions of all folds, which is why
it can sit outside the min/max of the per-fold values.

Effect of the graph encoder:

| task | baseline | baseline_auc | graph_variant | graph_auc | delta |
|---|---|---|---|---|---|
| kd_inh | mlp | 0.5000 | gnn_mlp | 0.5059 | 0.0059 |
| oe_act | mlp | 0.7045 | gnn_mlp | 0.6986 | -0.0059 |

## Graph-free baselines (`pathwaygnn benchmark`)

| task | model | auc | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|---|
| kd_inh | logistic_regression | 0.6545 | 0.7127 | 0.1626 | 0.6050 | 0.2562 |
| kd_inh | random_forest | 0.7520 | 0.9042 | 0.2851 | 0.1111 | 0.1590 |
| kd_inh | xgboost | 0.7402 | 0.9165 | 0.4339 | 0.0529 | 0.0934 |
| kd_inh | mlp (pathwaygnn cv) | 0.5000 | NA | NA | NA | NA |
| kd_inh | gnn_mlp (pathwaygnn cv) | 0.5059 | NA | NA | NA | NA |
| oe_act | logistic_regression | 0.4541 | 0.7400 | 0.0474 | 0.1036 | 0.0648 |
| oe_act | random_forest | 0.6868 | 0.8733 | 0.2733 | 0.3321 | 0.2849 |
| oe_act | xgboost | 0.4663 | 0.8956 | 0.0000 | 0.0000 | 0.0000 |
| oe_act | mlp (pathwaygnn cv) | 0.7045 | NA | NA | NA | NA |
| oe_act | gnn_mlp (pathwaygnn cv) | 0.6986 | NA | NA | NA | NA |

The baselines consume exactly the same features as the GNN — the sparse perturbation and disease
signatures — without the pathway graph. Only ROC-AUC is comparable to the cross-validation rows;
the threshold metrics are omitted there because `cv` records ROC-AUC only.

## Holdout fine-tuning (`pathwaygnn finetune`)

| task | train | valid | test | epochs_run | best_epoch | best_valid_auc | test_auc | test_accuracy | test_precision | test_recall | test_f1 | test_predicted_positive_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kd_inh | 4859 | 1041 | 1044 | 23 | 3 | 0.5848 | 0.5940 | 0.0824 | 0.0824 | 1.0000 | 0.1522 | 1.0000 |
| oe_act | 314 | 66 | 70 | 30 | 10 | 0.9443 | 0.7392 | 0.8714 | 0.4000 | 0.5714 | 0.4706 | 0.1429 |

This protocol is a single stratified 70/15/15 split with early stopping on validation ROC-AUC and
`pos_weight` from the training class ratio, so its numbers are not directly comparable with the
5-fold results above. Compare `best_valid_auc` with `test_auc`: on oe_act the validation split holds
only a few dozen samples, so selecting on it overstates the held-out result. Note also that
`test_predicted_positive_ratio` reveals when a model simply predicts the positive class for
everything, which is what `pos_weight` encourages on these imbalanced labels.

## Per-disease breakdown

Top 15 diseases per task by held-out sample count:

| task | disease | samples | positives | auc_mlp | auc_gnn_mlp |
|---|---|---|---|---|---|
| kd_inh | DOID:0050156 | 224 | 16 | 0.4669 | 0.5925 |
| kd_inh | DOID:0050589 | 224 | 20 | 0.5271 | 0.4162 |
| kd_inh | DOID:1040 | 224 | 4 | 0.4347 | 0.5591 |
| kd_inh | DOID:10534 | 224 | 24 | 0.4893 | 0.5200 |
| kd_inh | DOID:10652 | 224 | 6 | 0.4063 | 0.5356 |
| kd_inh | DOID:12449 | 224 | 6 | 0.6036 | 0.5986 |
| kd_inh | DOID:1380 | 224 | 1 | 0.0493 | 0.4888 |
| kd_inh | DOID:13810 | 224 | 7 | 0.5270 | 0.4605 |
| kd_inh | DOID:14330 | 224 | 7 | 0.5734 | 0.5510 |
| kd_inh | DOID:1612 | 224 | 63 | 0.4859 | 0.4634 |
| kd_inh | DOID:1793 | 224 | 31 | 0.4051 | 0.4307 |
| kd_inh | DOID:1883 | 224 | 1 | 0.6211 | 0.6906 |
| kd_inh | DOID:1909 | 224 | 16 | 0.4141 | 0.5717 |
| kd_inh | DOID:2394 | 224 | 25 | 0.5398 | 0.5314 |
| kd_inh | DOID:2841 | 224 | 10 | 0.4516 | 0.4033 |
| oe_act | DOID:0050589 | 30 | 1 | 0.8966 | 0.7241 |
| oe_act | DOID:1206 | 30 | 1 | 0.0345 | 0.0345 |
| oe_act | DOID:14330 | 30 | 2 | 0.2321 | 0.4464 |
| oe_act | DOID:1612 | 30 | 6 | 0.2986 | 0.2153 |
| oe_act | DOID:2394 | 30 | 1 | 0.6207 | 0.8621 |
| oe_act | DOID:3265 | 30 | 1 | 0.0345 | 0.6552 |
| oe_act | DOID:4450 | 30 | 3 | 0.4321 | 0.3827 |
| oe_act | DOID:676 | 30 | 1 | 0.8621 | 0.1379 |
| oe_act | DOID:7148 | 30 | 1 | 0.1034 | 0.1379 |
| oe_act | DOID:8552 | 30 | 1 | 0.3793 | 0.3103 |
| oe_act | DOID:9119 | 30 | 1 | 0.5862 | 0.7241 |
| oe_act | DOID:9256 | 30 | 1 | 0.0690 | 0.0690 |
| oe_act | DOID:9352 | 30 | 15 | 0.4400 | 0.3511 |
| oe_act | DOID:9538 | 30 | 1 | 0.6897 | 0.1034 |
| oe_act | DOID:9744 | 30 | 1 | 0.4828 | 0.3103 |

The full table for every disease is in `outputs/tr/report/per_disease_auc.tsv`.

## Integrated Gradients (`pathwaygnn ig`)

| task | rank | node | ig_l2 | degree |
|---|---|---|---|---|
| kd_inh | 1 | CHEBI:4667 | 0.0130 | 28040 |
| kd_inh | 2 | CHEBI:9925 | 0.0130 | 28040 |
| kd_inh | 3 | CHEBI:39867 | 0.0112 | 28094 |
| kd_inh | 4 | CHEBI:60654 | 0.0101 | 28040 |
| kd_inh | 5 | CHEBI:2504 | 0.0093 | 19434 |
| kd_inh | 6 | CHEBI:33216 | 0.0083 | 9618 |
| kd_inh | 7 | CHEBI:4031 | 0.0083 | 16040 |
| kd_inh | 8 | CHEBI:29678 | 0.0079 | 9428 |
| kd_inh | 9 | CHEBI:31522 | 0.0078 | 13546 |
| kd_inh | 10 | CHEBI:46195 | 0.0077 | 14174 |
| kd_inh | 11 | CHEBI:64816 | 0.0071 | 13546 |
| kd_inh | 12 | CHEBI:28748 | 0.0071 | 13574 |
| kd_inh | 13 | CHEBI:46024 | 0.0069 | 9846 |
| kd_inh | 14 | SP1 | 0.0067 | 8120 |
| kd_inh | 15 | CHEBI:29865 | 0.0066 | 13518 |
| oe_act | 1 | CHEBI:39867 | 0.0235 | 28094 |
| oe_act | 2 | CHEBI:4667 | 0.0212 | 28040 |
| oe_act | 3 | CHEBI:60654 | 0.0195 | 28040 |
| oe_act | 4 | CHEBI:9925 | 0.0174 | 28040 |
| oe_act | 5 | CHEBI:2504 | 0.0158 | 19434 |
| oe_act | 6 | CHEBI:64816 | 0.0130 | 13546 |
| oe_act | 7 | CHEBI:4031 | 0.0127 | 16040 |
| oe_act | 8 | CHEBI:33364 | 0.0123 | 13042 |
| oe_act | 9 | CHEBI:29678 | 0.0118 | 9428 |
| oe_act | 10 | CHEBI:23414 | 0.0117 | 11862 |
| oe_act | 11 | CHEBI:46195 | 0.0115 | 14174 |
| oe_act | 12 | CHEBI:16469 | 0.0113 | 11488 |
| oe_act | 13 | CHEBI:31440 | 0.0112 | 11862 |
| oe_act | 14 | CHEBI:46024 | 0.0106 | 9846 |
| oe_act | 15 | MYC | 0.0102 | 10084 |

Top 10 attributed genes per channel:

| task | channel | rank | node | signed_ig |
|---|---|---|---|---|
| kd_inh | perturbation | 1 | SATB1 | -0.0001 |
| kd_inh | perturbation | 2 | MCOLN1 | -0.0000 |
| kd_inh | perturbation | 3 | MIF | 0.0000 |
| kd_inh | perturbation | 4 | PCNA | 0.0000 |
| kd_inh | perturbation | 5 | CSRP1 | 0.0000 |
| kd_inh | perturbation | 6 | ST3GAL5 | -0.0000 |
| kd_inh | perturbation | 7 | CHMP4A | 0.0000 |
| kd_inh | perturbation | 8 | SPP1 | -0.0000 |
| kd_inh | perturbation | 9 | ABCC5 | -0.0000 |
| kd_inh | perturbation | 10 | TIMM9 | 0.0000 |
| kd_inh | disease | 1 | AQP4 | -0.0000 |
| kd_inh | disease | 2 | MYOM2 | -0.0000 |
| kd_inh | disease | 3 | ZNF766 | -0.0000 |
| kd_inh | disease | 4 | RPS16 | 0.0000 |
| kd_inh | disease | 5 | SLC16A12 | -0.0000 |
| kd_inh | disease | 6 | FMO4 | -0.0000 |
| kd_inh | disease | 7 | FMO2 | -0.0000 |
| kd_inh | disease | 8 | CCL18 | -0.0000 |
| kd_inh | disease | 9 | CCL2 | -0.0000 |
| kd_inh | disease | 10 | PROM1 | 0.0000 |
| oe_act | perturbation | 1 | POLR2I | 0.0000 |
| oe_act | perturbation | 2 | PLCB3 | -0.0000 |
| oe_act | perturbation | 3 | PPARG | -0.0000 |
| oe_act | perturbation | 4 | PSMB8 | -0.0000 |
| oe_act | perturbation | 5 | MCOLN1 | 0.0000 |
| oe_act | perturbation | 6 | OXA1L | 0.0000 |
| oe_act | perturbation | 7 | SNX11 | -0.0000 |
| oe_act | perturbation | 8 | HMOX1 | -0.0000 |
| oe_act | perturbation | 9 | HSPA8 | 0.0000 |
| oe_act | perturbation | 10 | CBR3 | -0.0000 |
| oe_act | disease | 1 | MYOM2 | 0.0000 |
| oe_act | disease | 2 | DDX17 | 0.0000 |
| oe_act | disease | 3 | LY6E | -0.0000 |
| oe_act | disease | 4 | CCL23 | -0.0000 |
| oe_act | disease | 5 | IGFBP3 | 0.0000 |
| oe_act | disease | 6 | ORM2 | -0.0000 |
| oe_act | disease | 7 | NEBL | 0.0000 |
| oe_act | disease | 8 | YBX3 | 0.0000 |
| oe_act | disease | 9 | HBA2 | -0.0000 |
| oe_act | disease | 10 | FOLR3 | -0.0000 |

Degree/attribution Pearson correlation:
kd_inh r=0.823 (50 samples, 50 steps), oe_act r=0.899 (50 samples, 50 steps).

Attribution mass concentrates on the highest-degree nodes, and the top of the ranking is dominated
by PathwayCommons chemical entities (`CHEBI:*`) rather than genes — the same degree-driven pattern
the cancer reproduction reports. Read the ranking as where the encoder puts its mass, not as
evidence of a disease-specific mechanism; the per-channel table below is the gene-level view.

## Plots

![dataset_composition.png](tr_report_assets/dataset_composition.png)

![pretraining_diagnostics.png](tr_report_assets/pretraining_diagnostics.png)

![cv_auc_by_task_variant.png](tr_report_assets/cv_auc_by_task_variant.png)

![cv_fold_auc_boxplot.png](tr_report_assets/cv_fold_auc_boxplot.png)

![cv_roc_curves.png](tr_report_assets/cv_roc_curves.png)

![cv_training_curves.png](tr_report_assets/cv_training_curves.png)

![per_disease_auc_scatter.png](tr_report_assets/per_disease_auc_scatter.png)

![per_disease_auc_vs_samples.png](tr_report_assets/per_disease_auc_vs_samples.png)

![benchmark_vs_gnn.png](tr_report_assets/benchmark_vs_gnn.png)

![finetune_curves.png](tr_report_assets/finetune_curves.png)

![ig_degree_vs_score.png](tr_report_assets/ig_degree_vs_score.png)

![ig_top_nodes.png](tr_report_assets/ig_top_nodes.png)

## Exact commands

    conda activate gnn
    bash scripts/tr/prepare.sh
    pathwaygnn pretrain  --config configs/tr/pretrain.yaml
    pathwaygnn cv        --config configs/tr/cv.yaml
    pathwaygnn finetune  --config configs/tr/finetune_kd_inh.yaml
    pathwaygnn finetune  --config configs/tr/finetune_oe_act.yaml
    pathwaygnn benchmark --config configs/tr/benchmark_kd_inh.yaml
    pathwaygnn benchmark --config configs/tr/benchmark_oe_act.yaml
    pathwaygnn ig        --config configs/tr/ig_kd_inh.yaml
    pathwaygnn ig        --config configs/tr/ig_oe_act.yaml
    pathwaygnn-data tr-report --config configs/tr/report.yaml

## Interpretation scope

These are the numbers this pipeline currently produces on this data, not a claim that the
architecture works on this task. Read them with three caveats:

* **kd_inh sits at chance** for both variants while the tree baselines reach far higher ROC-AUC on
  the identical folds. On this task the graph pipeline extracts less than plain feature models do.
* **oe_act is small** (450 samples,
  37 positive), so its fold spread is wide and the
  mean over five folds is a weak estimate.
* **One pre-training run** feeds every downstream number; no seed sweep was performed, and the
  encoder is frozen by default (`end_to_end: false` in `configs/tr/cv.yaml`).

Per-disease ROC-AUC is undefined wherever a disease's held-out samples are single-class, and is
reported as NA in that case.
