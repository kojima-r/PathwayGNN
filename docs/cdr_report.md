# PathwayGNN cancer drug-response report

## What this report covers

Dataset **cdr** — the GraphCDRScan corpus (GDSC1 dose response, Cell Model Passports
mutations, Reactome functional interactions) — prepared from
`data_cdr/processed/full_features` into `data_cdr/prepared`:
13,606 graph nodes, 536,274 directed edges,
356 relation types, 107,418 samples built from
760 cell lines x 168 compounds, tasks
sensitive_drugwise, sensitive_global. Run status: 8 cross-validation conditions, 2 holdout runs, 2 baseline runs, 2 attribution runs. Graph pre-training: 100 epochs, final DistMult loss 0.7109, final pairwise accuracy 0.9056.
Best cross-validated condition per task — sensitive_drugwise: gnn_mlp_cov 0.7212, sensitive_global: gnn_mlp_cov 0.9243.

A sample is one *(cell line, compound)* pair. Preprocessing turns the GDSC `LN_IC50` into a binary
label, because `pathwaygnn` trains binary problems only:

* **sensitive_drugwise** — 1 when `LN_IC50` is below the *same compound's* median. Every compound
  contributes ~50% positives, so the compound's overall potency carries no signal and the label can
  only be predicted from the cell line.
* **sensitive_global** — 1 when `LN_IC50` is below the median over all samples. Here the compound
  identity alone explains most of the label.

Each sample carries one sparse node-level feature and one sample-level feature vector:

* node_feature `mutation` — the number of mutations per Cancer-Gene-Census gene of the cell line, indexed
  by graph node. Because the profile depends only on the cell line, the
  107,418 samples share 760
  distinct rows through `rows/mutation.npy`.
* sample_features — the GraphCDRScan sample-feature vector verbatim: the 96/78/83-context mutational
  spectra of the cell line, its primary-site one-hot and the 3 x 1024-bit RDKit compound
  fingerprint (3,348 values).

Every number below comes from artifacts under `outputs/cdr/`, and every table is also written as TSV
under `outputs/cdr/report/`. Cross-validation and the graph-free baselines use the same stratified 5-fold
split (seed 42, `StratifiedKFold(shuffle=True)`), so those model comparisons are on identical folds;
attribution runs on fold 0 of `gnn_mlp_cov`, and holdout fine-tuning uses its own 70/15/15 split.

## Dataset audit

| task | samples | positive | positive_ratio | cell_lines | compounds | sites_used | sites_total | sample_features | mutation_rows | mean_genes_mutation | label_reference |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sensitive_drugwise | 107418 | 53669 | 0.4996 | 760 | 168 | 19 | 19 | 3348 | 760 | 155.1053 | median LN_IC50 of the same compound |
| sensitive_global | 107418 | 53709 | 0.5000 | 760 | 168 | 19 | 19 | 3348 | 760 | 155.1053 | median LN_IC50 over every sample |

`mutation_rows` is the number of distinct mutation profiles the table stores, and
`mean_genes_mutation` the mean number of mutated census genes per profile. `sites_used` counts the
primary sites that actually appear, out of `sites_total` in the one-hot block.

## Cross-validation (`pathwaygnn cv`)

| task | variant | uses_graph | uses_sample_features | mean_auc | std_auc | mean_accuracy | mean_precision | mean_recall | mean_f1 | min_fold_auc | max_fold_auc | pooled_auc | folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sensitive_drugwise | mlp | False | False | 0.5146 | 0.0132 | 0.5149 | 0.4315 | 0.2327 | 0.2949 | 0.4947 | 0.5302 | 0.5189 | 5 |
| sensitive_drugwise | mlp_cov | False | True | 0.7183 | 0.0021 | 0.6558 | 0.6941 | 0.5567 | 0.6177 | 0.7163 | 0.7218 | 0.7181 | 5 |
| sensitive_drugwise | gnn_mlp | True | False | 0.5719 | 0.0423 | 0.5459 | 0.5645 | 0.5732 | 0.5342 | 0.4921 | 0.6118 | 0.5730 | 5 |
| sensitive_drugwise | gnn_mlp_cov | True | True | 0.7212 | 0.0051 | 0.6569 | 0.6846 | 0.5833 | 0.6290 | 0.7141 | 0.7280 | 0.7206 | 5 |
| sensitive_global | mlp | False | False | 0.5017 | 0.0058 | 0.4990 | 0.3008 | 0.3273 | 0.2792 | 0.4909 | 0.5081 | 0.4996 | 5 |
| sensitive_global | mlp_cov | False | True | 0.9243 | 0.0007 | 0.8396 | 0.8396 | 0.8396 | 0.8396 | 0.9235 | 0.9253 | 0.9241 | 5 |
| sensitive_global | gnn_mlp | True | False | 0.5037 | 0.0043 | 0.5028 | 0.3067 | 0.3696 | 0.2983 | 0.4981 | 0.5099 | 0.5050 | 5 |
| sensitive_global | gnn_mlp_cov | True | True | 0.9243 | 0.0015 | 0.8399 | 0.8441 | 0.8338 | 0.8389 | 0.9221 | 0.9264 | 0.9242 | 5 |

`pooled_auc` is computed once over the concatenated held-out predictions of all folds, which is why
it can sit outside the min/max of the per-fold values. The `mean_accuracy`/`precision`/`recall`/`f1`
columns score the same folds at a fixed **0.5 decision threshold**; ROC-AUC is threshold-free, so a
condition can rank well and still sit at a poor operating point (or the reverse).

`cv` trains with an unweighted BCE loss — unlike `finetune`, which applies `pos_weight` — so on imbalanced labels the 0.5 operating point would drift towards the majority class. Both tasks here are ~50% positive by construction, so accuracy and F1 stay interpretable.

The grid is a two-factor ablation — the pathway graph on/off crossed with the sample-level feature branch
on/off — so each switch can be read with the other held fixed:

| task | factor | held_fixed | off_variant | off_auc | on_variant | on_auc | delta |
|---|---|---|---|---|---|---|---|
| sensitive_drugwise | graph encoder | use_sample_features=False | mlp | 0.5146 | gnn_mlp | 0.5719 | 0.0573 |
| sensitive_drugwise | graph encoder | use_sample_features=True | mlp_cov | 0.7183 | gnn_mlp_cov | 0.7212 | 0.0030 |
| sensitive_drugwise | sample-level features | use_graph=False | mlp | 0.5146 | mlp_cov | 0.7183 | 0.2036 |
| sensitive_drugwise | sample-level features | use_graph=True | gnn_mlp | 0.5719 | gnn_mlp_cov | 0.7212 | 0.1493 |
| sensitive_global | graph encoder | use_sample_features=False | mlp | 0.5017 | gnn_mlp | 0.5037 | 0.0020 |
| sensitive_global | graph encoder | use_sample_features=True | mlp_cov | 0.9243 | gnn_mlp_cov | 0.9243 | 0.0000 |
| sensitive_global | sample-level features | use_graph=False | mlp | 0.5017 | mlp_cov | 0.9243 | 0.4226 |
| sensitive_global | sample-level features | use_graph=True | gnn_mlp | 0.5037 | gnn_mlp_cov | 0.9243 | 0.4206 |

The `use_sample_features` rows are large by construction: the sample-level feature block carries the compound
fingerprint, and `sensitive_global` is mostly a question about the compound. The rows that speak to
the pathway graph are the `graph encoder` ones, and they are only informative where the mutation
node-level feature is the model's *only* view of the sample (`use_sample_features=False`) — with the sample-level features on,
the graph has little left to add.

## Graph-free baselines (`pathwaygnn benchmark`)

| task | model | auc | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|---|
| sensitive_drugwise | logistic_regression | 0.7090 | 0.6561 | 0.6626 | 0.6352 | 0.6485 |
| sensitive_drugwise | random_forest | 0.7355 | 0.6770 | 0.6929 | 0.6350 | 0.6626 |
| sensitive_drugwise | xgboost | 0.7582 | 0.6880 | 0.6998 | 0.6575 | 0.6780 |
| sensitive_drugwise | mlp (pathwaygnn cv) | 0.5146 | 0.5149 | 0.4315 | 0.2327 | 0.2949 |
| sensitive_drugwise | mlp_cov (pathwaygnn cv) | 0.7183 | 0.6558 | 0.6941 | 0.5567 | 0.6177 |
| sensitive_drugwise | gnn_mlp (pathwaygnn cv) | 0.5719 | 0.5459 | 0.5645 | 0.5732 | 0.5342 |
| sensitive_drugwise | gnn_mlp_cov (pathwaygnn cv) | 0.7212 | 0.6569 | 0.6846 | 0.5833 | 0.6290 |
| sensitive_global | logistic_regression | 0.9216 | 0.8384 | 0.8391 | 0.8374 | 0.8382 |
| sensitive_global | random_forest | 0.9061 | 0.8204 | 0.8233 | 0.8161 | 0.8196 |
| sensitive_global | xgboost | 0.9330 | 0.8510 | 0.8535 | 0.8476 | 0.8505 |
| sensitive_global | mlp (pathwaygnn cv) | 0.5017 | 0.4990 | 0.3008 | 0.3273 | 0.2792 |
| sensitive_global | mlp_cov (pathwaygnn cv) | 0.9243 | 0.8396 | 0.8396 | 0.8396 | 0.8396 |
| sensitive_global | gnn_mlp (pathwaygnn cv) | 0.5037 | 0.5028 | 0.3067 | 0.3696 | 0.2983 |
| sensitive_global | gnn_mlp_cov (pathwaygnn cv) | 0.9243 | 0.8399 | 0.8441 | 0.8338 | 0.8389 |

The baselines consume exactly the same features as the GNN — the mutation table expanded to
`[samples, 13,606]` plus the sample-level feature block — without the pathway graph. All five
metrics are on the same footing: both sides are the mean over the same five folds, and both
threshold at 0.5. These are reference points, not tuned models: the features are unscaled
counts and raw bits, so `LogisticRegression` hits its `max_iter=1000` lbfgs limit without
converging, and the forest is capped at 60 trees of depth 12 to finish on a
107,418 x 16,954 matrix.

## Holdout fine-tuning (`pathwaygnn finetune`)

| task | train | valid | test | epochs_run | best_epoch | best_valid_auc | test_auc | test_accuracy | test_precision | test_recall | test_f1 | test_predicted_positive_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sensitive_drugwise | 75192 | 16112 | 16114 | 40 | 40 | 0.7365 | 0.7306 | 0.6636 | 0.6907 | 0.5917 | 0.6374 | 0.4280 |
| sensitive_global | 75192 | 16112 | 16114 | 40 | 38 | 0.9222 | 0.9229 | 0.8368 | 0.8470 | 0.8223 | 0.8344 | 0.4854 |

This protocol is a single stratified 70/15/15 split of `gnn_mlp_cov` with early stopping on
validation ROC-AUC and `pos_weight` from the training class ratio, so its numbers are not directly
comparable with the 5-fold results above.

## Per primary site

| task | primary_site | samples | positives | auc_mlp | auc_mlp_cov | auc_gnn_mlp | auc_gnn_mlp_cov |
|---|---|---|---|---|---|---|---|
| sensitive_drugwise | Lung | 21696 | 8683 | 0.5155 | 0.6638 | 0.5436 | 0.6625 |
| sensitive_drugwise | Haematopoietic and Lymphoid | 17092 | 12869 | 0.5163 | 0.7709 | 0.5684 | 0.7727 |
| sensitive_drugwise | Skin | 7841 | 3814 | 0.5368 | 0.6844 | 0.5253 | 0.6852 |
| sensitive_drugwise | Central Nervous System | 7420 | 3581 | 0.5163 | 0.6503 | 0.5660 | 0.6543 |
| sensitive_drugwise | Breast | 7103 | 2770 | 0.5209 | 0.6435 | 0.5368 | 0.6325 |
| sensitive_drugwise | Large Intestine | 6422 | 2422 | 0.5767 | 0.7224 | 0.5809 | 0.7319 |
| sensitive_drugwise | Head and Neck | 5487 | 2935 | 0.5030 | 0.6693 | 0.5261 | 0.6831 |
| sensitive_drugwise | Esophagus | 4947 | 2406 | 0.5000 | 0.6983 | 0.4944 | 0.6874 |
| sensitive_drugwise | Peripheral Nervous System | 4703 | 2400 | 0.5262 | 0.6369 | 0.5703 | 0.6593 |
| sensitive_drugwise | Kidney | 4403 | 2120 | 0.5307 | 0.6354 | 0.5338 | 0.6541 |
| sensitive_drugwise | Ovary | 4156 | 1866 | 0.5092 | 0.6322 | 0.6031 | 0.6552 |
| sensitive_drugwise | Pancreas | 3887 | 1665 | 0.4658 | 0.6570 | 0.5691 | 0.6782 |
| sensitive_drugwise | Stomach | 3109 | 1445 | 0.5605 | 0.6767 | 0.5705 | 0.6737 |
| sensitive_drugwise | Bladder | 2576 | 1314 | 0.5310 | 0.6775 | 0.5251 | 0.6851 |
| sensitive_drugwise | Thyroid | 2290 | 1291 | 0.5347 | 0.6628 | 0.5506 | 0.6920 |
| sensitive_drugwise | Liver | 2180 | 1093 | 0.5307 | 0.6239 | 0.5020 | 0.6064 |
| sensitive_drugwise | Cervix | 1828 | 791 | 0.4565 | 0.7295 | 0.5922 | 0.7341 |
| sensitive_drugwise | Vulva | 141 | 77 | 0.5186 | 0.7080 | 0.4775 | 0.7291 |
| sensitive_drugwise | Bone | 137 | 127 | 0.6169 | 0.7362 | 0.3843 | 0.7244 |
| sensitive_global | Lung | 21696 | 9711 | 0.4949 | 0.9167 | 0.4995 | 0.9179 |
| sensitive_global | Haematopoietic and Lymphoid | 17092 | 11097 | 0.5066 | 0.9189 | 0.5093 | 0.9195 |
| sensitive_global | Skin | 7841 | 3852 | 0.5036 | 0.9248 | 0.5095 | 0.9247 |
| sensitive_global | Central Nervous System | 7420 | 3629 | 0.4960 | 0.9250 | 0.4917 | 0.9247 |
| sensitive_global | Breast | 7103 | 3144 | 0.4992 | 0.9190 | 0.5126 | 0.9197 |
| sensitive_global | Large Intestine | 6422 | 2730 | 0.4845 | 0.9245 | 0.5085 | 0.9249 |
| sensitive_global | Head and Neck | 5487 | 2773 | 0.4993 | 0.9348 | 0.4984 | 0.9349 |
| sensitive_global | Esophagus | 4947 | 2385 | 0.5136 | 0.9244 | 0.5094 | 0.9224 |
| sensitive_global | Peripheral Nervous System | 4703 | 2455 | 0.4945 | 0.9037 | 0.4920 | 0.9029 |
| sensitive_global | Kidney | 4403 | 2172 | 0.5085 | 0.9180 | 0.5169 | 0.9183 |
| sensitive_global | Ovary | 4156 | 1895 | 0.4868 | 0.9271 | 0.4983 | 0.9260 |
| sensitive_global | Pancreas | 3887 | 1830 | 0.4977 | 0.9303 | 0.5160 | 0.9294 |
| sensitive_global | Stomach | 3109 | 1495 | 0.4978 | 0.9192 | 0.4693 | 0.9164 |
| sensitive_global | Bladder | 2576 | 1266 | 0.4962 | 0.9280 | 0.4983 | 0.9290 |
| sensitive_global | Thyroid | 2290 | 1192 | 0.5191 | 0.9285 | 0.5108 | 0.9273 |
| sensitive_global | Liver | 2180 | 1097 | 0.4987 | 0.9326 | 0.4985 | 0.9318 |
| sensitive_global | Cervix | 1828 | 817 | 0.5233 | 0.9358 | 0.5310 | 0.9371 |
| sensitive_global | Vulva | 141 | 71 | 0.5202 | 0.9700 | 0.4385 | 0.9634 |
| sensitive_global | Bone | 137 | 98 | 0.5315 | 0.9479 | 0.4287 | 0.9414 |

The full table is in `outputs/cdr/report/per_site_auc.tsv`. Per-site ROC-AUC is undefined wherever a site's
held-out samples are single-class, and is reported as NA in that case.

## Integrated Gradients (`pathwaygnn ig`)

Top attributed graph nodes (HGNC ids resolved to approved symbols through
`data_cdr/raw/EnsemblToHGNC.tsv`):

| task | rank | node | ig_l2 | degree |
|---|---|---|---|---|
| sensitive_drugwise | 1 | RPS27A (HGNC:10417) | 0.0061 | 2050 |
| sensitive_drugwise | 2 | UBA52 (HGNC:12458) | 0.0053 | 1962 |
| sensitive_drugwise | 3 | UBB (HGNC:12463) | 0.0052 | 1684 |
| sensitive_drugwise | 4 | UBC (HGNC:12468) | 0.0051 | 1664 |
| sensitive_drugwise | 5 | EP300 (HGNC:3373) | 0.0045 | 2114 |
| sensitive_drugwise | 6 | SP1 (HGNC:11205) | 0.0043 | 1244 |
| sensitive_drugwise | 7 | PARP1 (HGNC:270) | 0.0039 | 436 |
| sensitive_drugwise | 8 | PRKDC (HGNC:9413) | 0.0037 | 238 |
| sensitive_drugwise | 9 | TOP2A (HGNC:11989) | 0.0036 | 412 |
| sensitive_drugwise | 10 | NPM1 (HGNC:7910) | 0.0036 | 624 |
| sensitive_drugwise | 11 | ADRM1 (HGNC:15759) | 0.0035 | 392 |
| sensitive_drugwise | 12 | CTCF (HGNC:13723) | 0.0035 | 1020 |
| sensitive_drugwise | 13 | TRIM28 (HGNC:16384) | 0.0034 | 462 |
| sensitive_drugwise | 14 | FYN (HGNC:4037) | 0.0032 | 846 |
| sensitive_drugwise | 15 | TRAF6 (HGNC:12036) | 0.0032 | 514 |
| sensitive_drugwise | 16 | CCNB1 (HGNC:1579) | 0.0031 | 546 |
| sensitive_drugwise | 17 | BRCA1 (HGNC:1100) | 0.0030 | 696 |
| sensitive_drugwise | 18 | RPA1 (HGNC:10289) | 0.0030 | 750 |
| sensitive_drugwise | 19 | GRB2 (HGNC:4566) | 0.0029 | 922 |
| sensitive_drugwise | 20 | MSH2 (HGNC:7325) | 0.0029 | 164 |
| sensitive_global | 1 | UBC (HGNC:12468) | 0.0003 | 1664 |
| sensitive_global | 2 | PIK3CA (HGNC:8975) | 0.0003 | 758 |
| sensitive_global | 3 | UBA52 (HGNC:12458) | 0.0003 | 1962 |
| sensitive_global | 4 | PIK3R1 (HGNC:8979) | 0.0003 | 820 |
| sensitive_global | 5 | PIK3CB (HGNC:8976) | 0.0002 | 494 |
| sensitive_global | 6 | UBB (HGNC:12463) | 0.0002 | 1684 |
| sensitive_global | 7 | GRB2 (HGNC:4566) | 0.0002 | 922 |
| sensitive_global | 8 | ACTB (HGNC:132) | 0.0002 | 798 |
| sensitive_global | 9 | PLCG1 (HGNC:9065) | 0.0002 | 590 |
| sensitive_global | 10 | RPS27A (HGNC:10417) | 0.0002 | 2050 |
| sensitive_global | 11 | HDAC1 (HGNC:4852) | 0.0002 | 820 |
| sensitive_global | 12 | HSPA8 (HGNC:5241) | 0.0002 | 1356 |
| sensitive_global | 13 | EP300 (HGNC:3373) | 0.0002 | 2114 |
| sensitive_global | 14 | DYNC1I2 (HGNC:2964) | 0.0002 | 686 |
| sensitive_global | 15 | NPM1 (HGNC:7910) | 0.0002 | 624 |
| sensitive_global | 16 | CBL (HGNC:1541) | 0.0002 | 516 |
| sensitive_global | 17 | EGFR (HGNC:3236) | 0.0002 | 490 |
| sensitive_global | 18 | SMARCC2 (HGNC:11105) | 0.0002 | 438 |
| sensitive_global | 19 | PARP1 (HGNC:270) | 0.0002 | 436 |
| sensitive_global | 20 | TRIM28 (HGNC:16384) | 0.0002 | 462 |

Top 10 attributed genes of the `mutation` node-level feature:

| task | node_feature | rank | node | signed_ig |
|---|---|---|---|---|
| sensitive_drugwise | mutation | 1 | MUC16 (HGNC:15582) | -0.0049 |
| sensitive_drugwise | mutation | 2 | EP300 (HGNC:3373) | -0.0041 |
| sensitive_drugwise | mutation | 3 | KMT2C (HGNC:13726) | -0.0038 |
| sensitive_drugwise | mutation | 4 | MUC4 (HGNC:7514) | -0.0035 |
| sensitive_drugwise | mutation | 5 | PABPC1 (HGNC:8554) | -0.0034 |
| sensitive_drugwise | mutation | 6 | ATM (HGNC:795) | -0.0034 |
| sensitive_drugwise | mutation | 7 | NCOR2 (HGNC:7673) | -0.0033 |
| sensitive_drugwise | mutation | 8 | RNF213 (HGNC:14539) | -0.0027 |
| sensitive_drugwise | mutation | 9 | KMT2D (HGNC:7133) | -0.0027 |
| sensitive_drugwise | mutation | 10 | TP53 (HGNC:11998) | -0.0025 |
| sensitive_global | mutation | 1 | MUC4 (HGNC:7514) | 0.0004 |
| sensitive_global | mutation | 2 | MUC16 (HGNC:15582) | 0.0003 |
| sensitive_global | mutation | 3 | KMT2C (HGNC:13726) | 0.0003 |
| sensitive_global | mutation | 4 | LRP1B (HGNC:6693) | 0.0002 |
| sensitive_global | mutation | 5 | PABPC1 (HGNC:8554) | 0.0002 |
| sensitive_global | mutation | 6 | USP6 (HGNC:12629) | 0.0002 |
| sensitive_global | mutation | 7 | PTPRT (HGNC:9682) | 0.0002 |
| sensitive_global | mutation | 8 | BIRC6 (HGNC:13516) | 0.0001 |
| sensitive_global | mutation | 9 | TRRAP (HGNC:12347) | 0.0001 |
| sensitive_global | mutation | 10 | CAMTA1 (HGNC:18806) | 0.0001 |

Top 10 attributed sample-level features:

| task | rank | sample_feature | signed_ig |
|---|---|---|---|
| sensitive_drugwise | 1 | site_Thyroid | 0.0106 |
| sensitive_drugwise | 2 | spectra96_4 | -0.0090 |
| sensitive_drugwise | 3 | site_Large Intestine | 0.0065 |
| sensitive_drugwise | 4 | spectra78_69 | -0.0059 |
| sensitive_drugwise | 5 | site_Haematopoietic and Lymphoid | 0.0054 |
| sensitive_drugwise | 6 | spectra78_15 | 0.0054 |
| sensitive_drugwise | 7 | spectra96_66 | 0.0054 |
| sensitive_drugwise | 8 | spectra96_44 | -0.0052 |
| sensitive_drugwise | 9 | site_Skin | -0.0050 |
| sensitive_drugwise | 10 | spectra96_40 | -0.0049 |
| sensitive_global | 1 | spectra78_36 | -0.0133 |
| sensitive_global | 2 | spectra96_4 | -0.0130 |
| sensitive_global | 3 | site_Large Intestine | 0.0124 |
| sensitive_global | 4 | site_Breast | 0.0121 |
| sensitive_global | 5 | spectra78_17 | -0.0113 |
| sensitive_global | 6 | spectra78_21 | -0.0104 |
| sensitive_global | 7 | spectra96_56 | -0.0103 |
| sensitive_global | 8 | spectra96_64 | 0.0094 |
| sensitive_global | 9 | spectra96_13 | -0.0088 |
| sensitive_global | 10 | spectra78_23 | 0.0078 |

Degree/attribution Pearson correlation:
sensitive_drugwise r=0.769 (40 samples, 25 steps), sensitive_global r=0.745 (40 samples, 25 steps).

The graph ranking is degree-driven — the top of it is the ubiquitin/ribosomal hubs (`UBC`, `UBB`,
`UBA52`, `RPS27A`) that dominate Reactome's functional-interaction network. Read it as where the
encoder puts its mass, not as evidence of a drug-response mechanism; the `mutation` table is
the gene-level view, and the sample-level feature table separates what comes from the compound fingerprint from
what comes from the cell line's spectra and site.

## Plots

![Dataset composition](cdr_report_assets/dataset_composition.png)

![Pretraining diagnostics](cdr_report_assets/pretraining_diagnostics.png)

![CV AUC by task variant](cdr_report_assets/cv_auc_by_task_variant.png)

![Ablation deltas](cdr_report_assets/ablation_deltas.png)

![CV fold AUC boxplot](cdr_report_assets/cv_fold_auc_boxplot.png)

![CV ROC curves](cdr_report_assets/cv_roc_curves.png)

![CV training curves](cdr_report_assets/cv_training_curves.png)

![Per site AUC scatter](cdr_report_assets/per_site_auc_scatter.png)

![Per site AUC vs samples](cdr_report_assets/per_site_auc_vs_samples.png)

![Benchmark vs GNN](cdr_report_assets/benchmark_vs_gnn.png)

![Finetune curves](cdr_report_assets/finetune_curves.png)

![IG degree vs score](cdr_report_assets/ig_degree_vs_score.png)

![IG top nodes](cdr_report_assets/ig_top_nodes.png)

![IG top sample features](cdr_report_assets/ig_top_sample_features.png)

## Exact commands

    conda activate gnn
    # upstream GraphCDRScan stage, only needed to rebuild data_cdr/processed/
    python -m scripts.cdr.upstream.download_raw_data
    python -m scripts.cdr.upstream.prepare_data --config configs/cdr/upstream.json

    bash scripts/cdr/prepare.sh
    pathwaygnn pretrain  --config configs/cdr/pretrain.yaml
    pathwaygnn cv        --config configs/cdr/cv.yaml
    pathwaygnn finetune  --config configs/cdr/finetune_drugwise.yaml
    pathwaygnn finetune  --config configs/cdr/finetune_global.yaml
    pathwaygnn benchmark --config configs/cdr/benchmark_drugwise.yaml
    pathwaygnn benchmark --config configs/cdr/benchmark_global.yaml
    pathwaygnn ig        --config configs/cdr/ig_drugwise.yaml
    pathwaygnn ig        --config configs/cdr/ig_global.yaml
    pathwaygnn-data cdr-report --config configs/cdr/report.yaml

`bash scripts/cdr/reproduce.sh` runs the same list.

## Interpretation scope

These are the numbers this pipeline currently produces on this data, not a claim that the
architecture solves drug-response prediction. Read them with these caveats:

* **Tree baselines are the reference to beat.** On these folds, on sensitive_drugwise the best baseline (xgboost, 0.7582) beats the best pathwaygnn condition (gnn_mlp_cov, 0.7212); on sensitive_global the best baseline (xgboost, 0.9330) beats the best pathwaygnn condition (gnn_mlp_cov, 0.9243). The GNN pipeline is
  not the strongest model here; the graph earns its keep only in the sample_feature-free ablation, where
  it is the difference between chance and a weak but non-zero signal.
* **The two tasks are not equally hard by construction.** `sensitive_global` is largely a question
  about the compound, `sensitive_drugwise` largely a question about the cell line; comparing their
  AUCs against each other says more about the labels than about the model.
* **Folds are random over samples, not over cell lines or compounds.** A cell line appears in both
  the training and the held-out fold with different compounds, so these numbers describe filling in
  a partly observed response matrix, not generalisation to an unseen cell line.
* **The mutation node_feature is a scalar per gene.** GraphCDRScan's per-mutation node features (variant
  type, encoded genomic position) are reduced to a mutation count, because the sample-level head
  projects one value per gene. The spectra keep some of that information at the sample level.
* **These are current-release inputs, not the 2018 CDRscan experiment.** GraphCDRScan substitutes
  GDSC1 (Oct 2023), Cell Model Passports mutations on GRCh38 and RDKit fingerprints for the paper's
  GDSC 6.0, COSMIC v82 and PaDEL descriptors, so nothing here is comparable to published CDRscan
  numbers.
* **One pre-training run** feeds every downstream number; no seed sweep was performed, and the
  encoder is frozen during cross-validation (`end_to_end: false`).
