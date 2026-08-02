# PathwayGNN target repositioning report

## What this report covers

Dataset **tr** at `/data1/kojima/PathwayGNN/PathwayGNN/data_tr/processed`, prepared
into `data_tr/prepared`: 30,895 graph nodes, 3,671,958 directed
edges, 13 relation types, tasks kd_inh, oe_act. Run status: 4 cross-validation conditions, 2 holdout runs, 2 baseline runs, 2 attribution runs.
Graph pre-training: 100 epochs, final DistMult loss 0.9667, final pairwise accuracy 0.8669.

Every number below comes from artifacts under `outputs/tr/`, and every table is also written as TSV
under `outputs/tr/report/`. Cross-validation and the graph-free baselines use the same stratified 5-fold
split (seed 42, `StratifiedKFold(shuffle=True)`), so those model comparisons are on identical folds;
attribution runs on fold 0 of the graph variant, and holdout fine-tuning uses its own 70/15/15 split.

## Dataset audit

| task | samples | positive | positive_ratio | perturbations | diseases_used | diseases_total | mean_genes_perturbation | mean_genes_disease | signature_genes_skipped | label_rows_skipped |
|---|---|---|---|---|---|---|---|---|---|---|
| kd_inh | 61101 | 5013 | 0.0820 | 33817 | 31 | 177 | 975 | 994 | 2 | 255 |
| oe_act | 3465 | 294 | 0.0848 | 20131 | 15 | 177 | 975 | 994 | 2 | 0 |

`diseases_used` counts the diseases that actually appear in a task's labels, out of
`diseases_total` in the shared disease channel. The `mean_genes_*` columns are the mean number of
non-zero genes per row of each channel after the 1e-7 cutoff.

## Cross-validation (`pathwaygnn cv`)

| task | variant | uses_graph | mean_auc | std_auc | mean_accuracy | mean_precision | mean_recall | mean_f1 | min_fold_auc | max_fold_auc | pooled_auc | folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kd_inh | mlp | False | 0.5175 | 0.0351 | 0.3993 | 0.0759 | 0.6399 | 0.1236 | 0.4994 | 0.5877 | 0.5171 | 5 |
| kd_inh | gnn_mlp | True | 0.5000 | 0.0000 | 0.4164 | 0.0492 | 0.6000 | 0.0910 | 0.5000 | 0.5000 | 0.5001 | 5 |
| oe_act | mlp | False | 0.7046 | 0.0419 | 0.7547 | 0.1925 | 0.5850 | 0.2890 | 0.6385 | 0.7652 | 0.6983 | 5 |
| oe_act | gnn_mlp | True | 0.6885 | 0.0916 | 0.7244 | 0.2786 | 0.5966 | 0.3478 | 0.5236 | 0.8034 | 0.7116 | 5 |

`pooled_auc` is computed once over the concatenated held-out predictions of all folds, which is why
it can sit outside the min/max of the per-fold values. The `mean_accuracy`/`precision`/`recall`/`f1`
columns score the same folds at a fixed **0.5 decision threshold**; ROC-AUC is threshold-free, so a
condition can rank well and still sit at a poor operating point (or the reverse).

`cv` weights the positive class by `pos_weight` (10.75–11.19 across folds, i.e. negatives/positives of each fold's training split), which is the rule `finetune` uses, so both protocols optimise the same loss; the 0.5 operating point is therefore comparable between the two tables.

Effect of the graph encoder:

| task | baseline | baseline_auc | graph_variant | graph_auc | delta |
|---|---|---|---|---|---|
| kd_inh | mlp | 0.5175 | gnn_mlp | 0.5000 | -0.0175 |
| oe_act | mlp | 0.7046 | gnn_mlp | 0.6885 | -0.0161 |

## Graph-free baselines (`pathwaygnn benchmark`)

| task | model | auc | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|---|
| kd_inh | logistic_regression | 0.7550 | 0.7049 | 0.1714 | 0.6774 | 0.2736 |
| kd_inh | random_forest | 0.8086 | 0.9147 | 0.4260 | 0.1145 | 0.1803 |
| kd_inh | xgboost | 0.8025 | 0.9186 | 0.6611 | 0.0164 | 0.0319 |
| kd_inh | mlp (pathwaygnn cv) | 0.5175 | 0.3993 | 0.0759 | 0.6399 | 0.1236 |
| kd_inh | gnn_mlp (pathwaygnn cv) | 0.5000 | 0.4164 | 0.0492 | 0.6000 | 0.0910 |
| oe_act | logistic_regression | 0.4342 | 0.7417 | 0.0831 | 0.2041 | 0.1181 |
| oe_act | random_forest | 0.8032 | 0.9140 | 0.4945 | 0.3606 | 0.4164 |
| oe_act | xgboost | 0.7197 | 0.9039 | 0.1477 | 0.0341 | 0.0553 |
| oe_act | mlp (pathwaygnn cv) | 0.7046 | 0.7547 | 0.1925 | 0.5850 | 0.2890 |
| oe_act | gnn_mlp (pathwaygnn cv) | 0.6885 | 0.7244 | 0.2786 | 0.5966 | 0.3478 |

The baselines consume exactly the same features as the GNN — the sparse perturbation and disease
signatures — without the pathway graph. All five metrics are on the same footing: both sides are
the mean over the same five folds, and both threshold at 0.5.

## Holdout fine-tuning (`pathwaygnn finetune`)

| task | train | valid | test | epochs_run | best_epoch | best_valid_auc | test_auc | test_accuracy | test_precision | test_recall | test_f1 | test_predicted_positive_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kd_inh | 42770 | 9164 | 9167 | 22 | 2 | 0.6128 | 0.5869 | 0.4498 | 0.0996 | 0.7092 | 0.1747 | 0.5846 |
| oe_act | 2424 | 519 | 522 | 45 | 25 | 0.6999 | 0.6451 | 0.9176 | 0.5385 | 0.3111 | 0.3944 | 0.0498 |

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
| kd_inh | DOID:0050156 | 1971 | 131 | 0.4840 | 0.4970 |
| kd_inh | DOID:0050589 | 1971 | 166 | 0.5176 | 0.5285 |
| kd_inh | DOID:1040 | 1971 | 41 | 0.4776 | 0.5122 |
| kd_inh | DOID:10534 | 1971 | 211 | 0.5012 | 0.5120 |
| kd_inh | DOID:10652 | 1971 | 47 | 0.4666 | 0.4403 |
| kd_inh | DOID:12449 | 1971 | 50 | 0.5049 | 0.4764 |
| kd_inh | DOID:1380 | 1971 | 8 | 0.5549 | 0.5966 |
| kd_inh | DOID:13810 | 1971 | 65 | 0.4503 | 0.5099 |
| kd_inh | DOID:14330 | 1971 | 55 | 0.5324 | 0.5044 |
| kd_inh | DOID:1612 | 1971 | 581 | 0.5124 | 0.5107 |
| kd_inh | DOID:1793 | 1971 | 280 | 0.5438 | 0.4834 |
| kd_inh | DOID:1883 | 1971 | 8 | 0.5038 | 0.5868 |
| kd_inh | DOID:1909 | 1971 | 163 | 0.4819 | 0.4875 |
| kd_inh | DOID:2394 | 1971 | 222 | 0.5355 | 0.5077 |
| kd_inh | DOID:2841 | 1971 | 85 | 0.4682 | 0.4858 |
| oe_act | DOID:0050589 | 231 | 9 | 0.5100 | 0.4655 |
| oe_act | DOID:1206 | 231 | 10 | 0.6552 | 0.4921 |
| oe_act | DOID:14330 | 231 | 18 | 0.4038 | 0.3911 |
| oe_act | DOID:1612 | 231 | 46 | 0.4528 | 0.4663 |
| oe_act | DOID:2394 | 231 | 9 | 0.4645 | 0.4389 |
| oe_act | DOID:3265 | 231 | 9 | 0.7465 | 0.5631 |
| oe_act | DOID:4450 | 231 | 27 | 0.5684 | 0.4986 |
| oe_act | DOID:676 | 231 | 9 | 0.6141 | 0.5891 |
| oe_act | DOID:7148 | 231 | 9 | 0.4189 | 0.6044 |
| oe_act | DOID:8552 | 231 | 9 | 0.7543 | 0.5916 |
| oe_act | DOID:9119 | 231 | 8 | 0.4857 | 0.5555 |
| oe_act | DOID:9256 | 231 | 10 | 0.2566 | 0.5172 |
| oe_act | DOID:9352 | 231 | 103 | 0.4758 | 0.4675 |
| oe_act | DOID:9538 | 231 | 9 | 0.4419 | 0.4800 |
| oe_act | DOID:9744 | 231 | 9 | 0.4464 | 0.5020 |

The full table for every disease is in `outputs/tr/report/per_disease_auc.tsv`.

## Integrated Gradients (`pathwaygnn ig`)

| task | rank | node | ig_l2 | degree |
|---|---|---|---|---|
| kd_inh | 1 | CHEBI:60654 | 0.0095 | 28022 |
| kd_inh | 2 | CHEBI:4667 | 0.0085 | 28022 |
| kd_inh | 3 | CHEBI:9925 | 0.0085 | 28022 |
| kd_inh | 4 | CHEBI:39867 | 0.0074 | 28076 |
| kd_inh | 5 | CHEBI:2504 | 0.0059 | 19418 |
| kd_inh | 6 | CHEBI:33364 | 0.0056 | 13036 |
| kd_inh | 7 | MYC | 0.0055 | 10080 |
| kd_inh | 8 | ESR1 | 0.0055 | 10072 |
| kd_inh | 9 | CHEBI:64816 | 0.0053 | 13544 |
| kd_inh | 10 | CHEBI:4031 | 0.0050 | 16036 |
| kd_inh | 11 | CHEBI:23965 | 0.0050 | 11310 |
| kd_inh | 12 | CHEBI:28748 | 0.0050 | 13572 |
| kd_inh | 13 | JUN | 0.0050 | 8654 |
| kd_inh | 14 | CHEBI:23414 | 0.0048 | 11862 |
| kd_inh | 15 | PTK2 | 0.0048 | 5418 |
| oe_act | 1 | CHEBI:4667 | 0.0117 | 28022 |
| oe_act | 2 | CHEBI:39867 | 0.0107 | 28076 |
| oe_act | 3 | CHEBI:60654 | 0.0102 | 28022 |
| oe_act | 4 | CHEBI:9925 | 0.0098 | 28022 |
| oe_act | 5 | JUN | 0.0070 | 8654 |
| oe_act | 6 | CHEBI:28748 | 0.0066 | 13572 |
| oe_act | 7 | ESR1 | 0.0066 | 10072 |
| oe_act | 8 | CHEBI:64816 | 0.0062 | 13544 |
| oe_act | 9 | CHEBI:2504 | 0.0062 | 19418 |
| oe_act | 10 | CHEBI:33364 | 0.0062 | 13036 |
| oe_act | 11 | CHEBI:23965 | 0.0060 | 11310 |
| oe_act | 12 | CHEBI:4031 | 0.0059 | 16036 |
| oe_act | 13 | CHEBI:29678 | 0.0057 | 9424 |
| oe_act | 14 | CHEBI:16469 | 0.0057 | 11482 |
| oe_act | 15 | CHEBI:27899 | 0.0056 | 13032 |

Top 10 attributed genes per channel:

| task | channel | rank | node | signed_ig |
|---|---|---|---|---|
| kd_inh | perturbation | 1 | CAST | -0.0001 |
| kd_inh | perturbation | 2 | OXA1L | -0.0001 |
| kd_inh | perturbation | 3 | CSRP1 | -0.0001 |
| kd_inh | perturbation | 4 | ABHD4 | -0.0001 |
| kd_inh | perturbation | 5 | GRN | -0.0001 |
| kd_inh | perturbation | 6 | PLCB3 | -0.0001 |
| kd_inh | perturbation | 7 | C2CD2 | -0.0001 |
| kd_inh | perturbation | 8 | GTPBP8 | -0.0001 |
| kd_inh | perturbation | 9 | TIMM9 | -0.0001 |
| kd_inh | perturbation | 10 | CALU | -0.0001 |
| kd_inh | disease | 1 | MYOM2 | -0.0000 |
| kd_inh | disease | 2 | RPS4Y1 | -0.0000 |
| kd_inh | disease | 3 | DEFA1B | -0.0000 |
| kd_inh | disease | 4 | BPIFB1 | -0.0000 |
| kd_inh | disease | 5 | MSMB | -0.0000 |
| kd_inh | disease | 6 | H1-4 | -0.0000 |
| kd_inh | disease | 7 | CLCA2 | -0.0000 |
| kd_inh | disease | 8 | HBA1 | 0.0000 |
| kd_inh | disease | 9 | CLC | -0.0000 |
| kd_inh | disease | 10 | SLPI | -0.0000 |
| oe_act | perturbation | 1 | MAP7 | -0.0000 |
| oe_act | perturbation | 2 | PLCB3 | -0.0000 |
| oe_act | perturbation | 3 | SNX11 | -0.0000 |
| oe_act | perturbation | 4 | SESN1 | -0.0000 |
| oe_act | perturbation | 5 | C2CD2 | -0.0000 |
| oe_act | perturbation | 6 | MCOLN1 | -0.0000 |
| oe_act | perturbation | 7 | HSPA1A | -0.0000 |
| oe_act | perturbation | 8 | PTPRC | -0.0000 |
| oe_act | perturbation | 9 | DNAJB1 | -0.0000 |
| oe_act | perturbation | 10 | ME2 | -0.0000 |
| oe_act | disease | 1 | MYOM2 | 0.0000 |
| oe_act | disease | 2 | IGHA2 | 0.0000 |
| oe_act | disease | 3 | RPS18 | -0.0000 |
| oe_act | disease | 4 | IFITM3 | -0.0000 |
| oe_act | disease | 5 | TMSB10 | -0.0000 |
| oe_act | disease | 6 | OLFM4 | -0.0000 |
| oe_act | disease | 7 | CEACAM6 | -0.0000 |
| oe_act | disease | 8 | IFITM2 | -0.0000 |
| oe_act | disease | 9 | IFITM1 | -0.0000 |
| oe_act | disease | 10 | DDX17 | 0.0000 |

Degree/attribution Pearson correlation:
kd_inh r=0.766 (50 samples, 50 steps), oe_act r=0.789 (50 samples, 50 steps).

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
* **oe_act is small** (3465 samples,
  294 positive), so its fold spread is wide and the
  mean over five folds is a weak estimate.
* **One pre-training run** feeds every downstream number; no seed sweep was performed, and the
  encoder is frozen by default (`end_to_end: false` in `configs/tr/cv.yaml`).

Per-disease ROC-AUC is undefined wherever a disease's held-out samples are single-class, and is
reported as NA in that case.
