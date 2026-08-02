# README_config — `configs/` の設定項目リファレンス

`configs/` 以下の YAML は、`pathwaygnn`（学習エンジン）と `pathwaygnn-data`（前処理・レポート）の
両 CLI が読む唯一の入力です。ここでは全キーを、既定値と「どのコマンドが読むか」つきで説明します。

- データセットごとの詳細は [README_data_tr.md](README_data_tr.md) /
  [README_data_cancer.md](README_data_cancer.md) / [README_data_cdr.md](README_data_cdr.md)
- 全体像は [README.md](README.md)

---

## 1. 設定機構（`src/pathwaygnn/config.py`）

`load_config` は約25行しかありません。挙動は次の3点だけです。

1. **`defaults:` の解決** — 文字列のリストで、**その設定ファイルからの相対パス**です。
   列挙順にロードして再帰マージし、最後に自ファイルの内容を上書きします。
   ```yaml
   defaults:
     - dataset.yaml        # configs/cdr/cv.yaml から見て configs/cdr/dataset.yaml
   ```
   `defaults` を持つファイルを `defaults` に指定することもできます
   （例: `configs/cdr/finetune_global.yaml` → `finetune_drugwise.yaml` → `dataset.yaml`）。

2. **マージ規則** — 辞書は**再帰的に**マージ、スカラーとリストは**丸ごと上書き**。
   つまり `variants:` を一部だけ差し替えることはできず、書けばリスト全体が置き換わります。

3. **スキーマ検証なし・CLI 上書きなし** — 値はすべて使用箇所で `.get(key, default)` として
   読まれます。したがって**実効的な既定値は YAML ではなく学習モジュール側にあります**
   （本ドキュメントの既定値はそのソースから採取したものです）。
   タイプミスしたキーはエラーにならず、単に無視されます。

**パスはすべてプロセスの CWD（リポジトリルート）からの相対パス**です。`scripts/*/*.sh` と
`scripts/*/*.py` は自分でルートに `cd` してから実行します。

```bash
pathwaygnn      pretrain|finetune|cv|ig|benchmark  --config <yaml>
pathwaygnn-data tr-prepare|cancer-prepare|cdr-prepare|cancer-map-ids|tr-report|cancer-report|cdr-report --config <yaml>
```

---

## 2. 共通キー

| キー | 型 | 既定値 | 読むコマンド | 説明 |
| --- | --- | --- | --- | --- |
| `dataset.name` | str | （なし） | 全学習コマンド, 全レポート | 使うデータセット名。`<dir>/dataset.json` の `name` と照合され、**不一致なら即座に失敗**する。`dir` の取り違え防止用。 |
| `dataset.dir` | str | （必須） | 同上 | 前処理済みデータセットのディレクトリ。 |
| `dataset.task` | str | （必須） | `finetune`, `ig`, `benchmark` | 単一タスク名。 |
| `dataset.tasks` | list[str] | （`task` にフォールバック） | `cv`, `tr-report`, `cdr-report` | 複数タスク。`cv` はこのリスト全体をグリッドに展開する。レポートでは省略時に `dataset.json` の全タスク（`cancer-report` は年を固定で持つためこのキーを読まない）。 |
| `seed` | int | `42`（`ig` のみ `100`） | 全学習コマンド | Python / NumPy / PyTorch と分割の乱数 seed。 |
| `device` | str | `auto` | `pretrain`, `cv`, `finetune`, `ig` | `cpu` を指定すると CUDA があっても CPU を使う。それ以外の値は「CUDA があれば使う」。**`benchmark` はこのキーを読まない**（sklearn/CPU のため。設定ファイルに書かれていても無視される）。 |
| `output_dir` | str | （必須） | 全コマンド | 成果物の出力先。 |

`dataset.yaml` は各データセットディレクトリに1つだけあり、`dataset:` ブロックだけを持ちます。
実験設定はこれを `defaults:` で取り込むため、**データセットの切り替えは include 1行**です。

```yaml
# configs/cdr/dataset.yaml
dataset:
  name: cdr
  dir: data_cdr/prepared
```

---

## 3. `pathwaygnn pretrain` — グラフ事前学習

`src/pathwaygnn/training/pretrain.py`。唯一の分散対応ループです。

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `model.hidden_dim` | int | `64` | ノード埋め込みと各 GIN 層の次元。**この値が下流の `cv`/`finetune` の `embedding_dim` を決める**（グラフを使う variant では `model.embedding_dim` は無視され encoder の `hidden_dim` が使われる）。 |
| `model.num_layers` | int | `2` | GIN 層数。全層の出力を連結して readout する。 |
| `model.dropout` | float | `0.1` | 各層出力の dropout。チェックポイントの `model_config` に記録される。 |
| `training.epochs` | int | `100` | エポック数。 |
| `training.steps_per_epoch` | int | `100` | 1エポックあたりのステップ数。**1ステップ＝全グラフ forward+backward 1回**なので、コストはサンプル数ではなくこの値に比例する。 |
| `training.batch_size` | int | `4096` | 1 rank あたりの正例エッジ数。グローバルバッチは `WORLD_SIZE × batch_size`。 |
| `training.learning_rate` | float | `1e-3` | AdamW の学習率。 |
| `training.weight_decay` | float | `1e-4` | AdamW の weight decay。 |
| `training.checkpoint_epochs` | list[int] | `[]` | 追加で `epoch_<n>.pt` を保存するエポック。`0` を含めると学習前の初期状態も保存される（cancer の事前学習スイープが使用）。 |
| `training.balanced_relations` | bool | `false` | `true` なら関係ごとに `batch_size` 本ずつ均等サンプリングする（実バッチは `batch_size × num_relations`）。`false` は全エッジから一様サンプリング。 |

出力: `best.pt`（loss 最小）、`last.pt`、`history.json`、`config.json`。
分散実行は `WORLD_SIZE` / `LOCAL_RANK` 環境変数で自動判定し（`torchrun`）、
CUDA なら NCCL、なければ Gloo。rank ごとの seed は `seed + rank`、書き込みは rank 0 のみ。

---

## 4. `pathwaygnn cv` — 層化 k 分割交差検証

`src/pathwaygnn/training/cv.py`。`variants × tasks × folds` のグリッドを1本のプロセスで回します。

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `folds` | int | `5` | `StratifiedKFold(shuffle=True, random_state=seed)` の分割数。 |
| `pretrained_checkpoint` | str | （`use_graph: true` の variant があれば必須） | `pretrain` が書いた `best.pt`。ノード数・関係数がデータセットと違えば **読み込み時に失敗**する。 |
| `write_root_manifest` | bool | `true` | `false` にすると `output_dir` 直下の `config.json` / `cv_results.json` を書かない。1条件ずつ並列実行するとき（`scripts/cdr/run_cv.py`, `scripts/cancer/reproduce_table1.py`）に競合を避けるため使う。 |

### `model:` ブロック（サンプルレベルヘッド `SampleLevelModel`）

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `model.embedding_dim` | int | `32` | 遺伝子の値を射影する次元。**グラフを使わない variant のみ有効**。`use_graph: true` では encoder の `hidden_dim` で上書きされる。 |
| `model.hidden_dim` | int | `embedding_dim` と同じ | 各ブロックの隠れ次元。 |
| `model.dropout` | float | `0.0` | `0` のときは `Dropout` モジュール自体が作られない（乱数列＝重み初期化がリファクタ前と一致するため）。 |
| `model.batch_norm` | bool | `false` | `true` で各ブロックに `BatchNorm1d` を挿入。cancer の論文アーキテクチャ用。 |
| `model.block` | str | `plain` | `plain`: Linear-ELU-[Dropout]-Linear（target repositioning / cdr）。`paper`: Linear-ELU-[BN]-Linear-ELU-[BN]（cancer 生存予測）。 |

### `training:` ブロック

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `training.epochs` | int | `150` | fold ごとのエポック数。 |
| `training.batch_size` | int | `1024` | サンプル数。 |
| `training.num_workers` | int | `0` | DataLoader のワーカー数。大規模データ（cdr の 107k サンプル）では律速要因になるので上げる。 |
| `training.shuffle` | bool | `false` | 学習データをシャッフルするか。cancer は参照実装に合わせて `false`。 |
| `training.learning_rate` | float | `5e-5` | AdamW の学習率。 |
| `training.weight_decay` | float | `0.0` | AdamW の weight decay。 |
| `training.scheduler_patience` | int | `10` | `ReduceLROnPlateau(mode="max", factor=0.5)` の patience。監視対象は held-out AUC。 |
| `training.end_to_end` | bool | `true` | `true` で encoder も同時に更新。`false` なら埋め込みを fold 開始時に1回だけ計算して使い回す（**最大の高速化レバー**）。variant 側の `end_to_end` が優先される。 |
| `training.loss_clip` | float\|null | `null` | サンプルごとの BCE 損失の下限クリップ。cancer は参照実装に合わせて `0.01`。 |
| `training.loss_reduction` | str | `mean` | `sum` または `mean`。cancer は `sum`。 |
| `training.pos_weight` | str\|float\|null | `null` | BCE の陽性クラス重み。`auto` は fold の学習側の `陰性数 / 陽性数`（**`finetune` と同じ規則**）、数値ならその値、`null` なら重み付けなし。採用値は `metrics.json` の `pos_weight` に記録される。cancer / cdr は `null`（公開数値の再現のため）、tr は `auto`。 |
| `training.grad_clip_value` | float\|null | `10.0` | `clip_grad_value_` の閾値。**ヘッドにのみ適用**され encoder には適用されない。 |
| `training.selection` | str | `final_epoch` | `final_epoch`: 最終エポックのモデルを採用（論文プロトコル）。`best_test_auc`: held-out fold の AUC が最良のエポックを採用（＝リーク。公開コード互換のためだけに存在）。 |
| `training.resume` | bool | `true` | fold ディレクトリに `metrics.json` / `predictions.npz` / `model.pt` が揃っていればそれを再利用する。**再実行を強制するにはその fold ディレクトリを削除する**。 |

### `variants:` — アブレーション条件のリスト

```yaml
variants:
  - {name: mlp,         use_graph: false, use_covariates: false, seed_index: 0}
  - {name: gnn_mlp_cov, use_graph: true,  use_covariates: true,  seed_index: 3}
```

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `name` | str | （必須） | 出力ディレクトリ名 `<output_dir>/<task>/<name>/fold_<k>/`。 |
| `use_graph` | bool | （必須） | グラフ encoder のノード埋め込みを遺伝子の値に加算するか。 |
| `use_covariates` | bool | `false` | 共変量分岐を使うか。タスクが共変量を持たない場合は `true` にできない。 |
| `seed_index` | int | リスト中の位置 | seed の variant 成分。**位置ではなくこの値で決まる**ため、1条件だけ単独実行してもグリッド全体と同じ seed になる。 |
| `end_to_end` | bool | `training.end_to_end` | variant 単位の上書き。 |

### 保存される評価値

`cv` は評価のたびに **ROC-AUC と、閾値 0.5 での accuracy / precision / recall / F1** を記録します
（`src/pathwaygnn/training/metrics.py` の `METRICS`）。

| 出力 | 形式 |
| --- | --- |
| `fold_<k>/metrics.json` の `history[]` | エポックごとに `test_auc`, `test_accuracy`, `test_precision`, `test_recall`, `test_f1`, `test_predicted_positive_ratio`, `test_actual_positive_ratio` |
| `fold_<k>/metrics.json` の直下 | 採用エポックの `auc`, `accuracy`, `precision`, `recall`, `f1`, `predicted_positive_ratio`, `actual_positive_ratio` |
| `<variant>/summary.json` | 指標ごとに `mean_<指標>`, `std_<指標>`, `fold_<指標>` |

- **モデル選択（`training.selection`）は ROC-AUC のみで行います。** 閾値指標は記録専用です。
- **AUC は sklearn の `roc_auc_score`** で計算されます（cancer の公開再現値がこれで出ているため）。
  閾値指標だけが `metrics.threshold_metrics` 由来です。
- **`cv` は `pos_weight` を使いません**（使うのは `finetune` だけ）。
  そのため陽性率が低いタスクでは、閾値 0.5 の予測が多数派クラスに潰れて
  precision / recall / F1 が 0 になることがあります。ROC-AUC が chance を上回っていても
  起こりうる現象で、`docs/tr_report.md` の `kd_inh` / `oe_act` が実例です。
- 閾値指標が記録される前に作られた fold は、**次に `cv` を実行したときに
  `predictions.npz` から自動的に補完**されます（再学習は不要）。

### fold seed の決まり方

```
fold_seed = seed + task.seed_offset * 1000 + fold + variant.seed_index * 100
```

`task.seed_offset` はタスクマニフェスト（`task.json`）が持ちます（cancer では検証年 1〜5、
tr/cdr では定義順 0,1）。位置に依存しないので、グリッドの一部だけを走らせても再現します。

---

## 5. `pathwaygnn finetune` — 単一分割の学習

`src/pathwaygnn/training/finetune.py`。train/valid/test を1回だけ切り、
**validation AUC で early stopping** します。`cv` と違い、学習時に訓練データのクラス比から
求めた `pos_weight` を適用する唯一のループです。

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `split` | list[float] | `[0.7, 0.15, 0.15]` | train / valid / test の比率。ラベル層化して分割する。 |
| `variant.use_graph` | bool | `true` | `cv` の `variants[]` と違い、こちらは**単数形 `variant:`** で1条件のみ。 |
| `variant.use_covariates` | bool | `false` | 同上。 |
| `model.embedding_dim` | int | `64` | **`cv` の既定値 32 とは異なる**。グラフ使用時は encoder の `hidden_dim` で上書き。 |
| `model.hidden_dim` / `dropout` / `batch_norm` / `block` | | `cv` と同じ | |
| `training.epochs` | int | `100` | 上限エポック数。 |
| `training.batch_size` | int | `64` | |
| `training.num_workers` | int | `0` | |
| `training.learning_rate` | float | `1e-3` | |
| `training.weight_decay` | float | `1e-4` | |
| `training.patience` | int | `20` | validation AUC が更新されないエポックがこの数続いたら停止。 |
| `training.train_encoder` | bool | `false` | encoder も更新するか（`cv` の `end_to_end` に相当）。 |

学習には訓練データのクラス比から求めた `pos_weight` を使うため、
`test_predicted_positive_ratio` が極端な値なら「全部陽性と答えている」状態を疑ってください。
出力: `best.pt`、`metrics.json`（`test` / `history` / `split` の全インデックス）。

---

## 6. `pathwaygnn benchmark` — グラフなしベースライン

`src/pathwaygnn/training/benchmark.py`。全 channel を `[samples, num_nodes]` の疎行列に展開し、
共変量を横に連結して sklearn / XGBoost にかけます。**`cv` と同じ seed・同じ `StratifiedKFold`**
なので fold は一致します。

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `folds` | int | `5` | |
| `n_jobs` | int | `-1` | Random Forest と XGBoost の並列数。 |
| `logistic_regression.C` | float | `1.0` | 逆正則化強度。`max_iter=1000`, `class_weight="balanced"` は固定。 |
| `random_forest.n_estimators` | int | `200` | |
| `random_forest.max_depth` | int\|null | `null` | `null` は深さ無制限。大きな疎行列では現実的な時間で終わらないので上限を入れる。 |
| `xgboost.enabled` | bool | `true` | `false` にすると XGBoost をスキップ（未インストール時に必要）。 |
| `xgboost.n_estimators` | int | `200` | |
| `xgboost.max_depth` | int | `4` | |
| `xgboost.learning_rate` | float | `0.05` | |
| `xgboost.subsample` | float | `0.8` | |

`scikit-learn` と `xgboost` が必要です（`pip install -e '.[benchmark]'`）。

---

## 7. `pathwaygnn ig` — Integrated Gradients

`src/pathwaygnn/training/ig.py`。`cv` が保存した fold のチェックポイントを読み、
グラフのノード埋め込み行列・各 channel の値・共変量ベクトルに対して帰属を計算します。

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `run_dir` | str | （必須） | `cv` の `output_dir`。`<run_dir>/<task>/<variant>/fold_<fold>/model.pt` を読む。 |
| `variant` | str | （必須） | variant 名（**文字列**。`finetune` の `variant:` 辞書とは別物）。 |
| `fold` | int | `0` | 対象 fold。 |
| `pretrained_checkpoint` | str | （グラフ使用時は必須） | encoder の形状復元に使う。 |
| `seed` | int | `100` | `max_samples` でサンプルを抽選する際の seed。**他コマンドの既定値 42 とは異なる**。 |
| `steps` | int | `50` | 積分ステップ数。 |
| `max_samples` | int\|null | `null` | 帰属を計算する held-out サンプル数。`null` は全件。**1サンプルあたり `steps` 回の全グラフ forward+backward** が走るので、コストは `max_samples × steps` に比例する。 |
| `top_k` | int | `1500` | 出力する TSV ランキングの件数。 |
| `per_group_rankings` | bool | `false` | `true` でグループ（がん種・疾患・原発部位）ごとのノードランキングも書く。グループ数だけファイルが増えるので明示的に有効化する。 |
| `reference` | dict | `{}` | そのまま `ig_summary.json` にコピーされる。既知の参照値（cancer では次数-IG 相関 0.727）を記録するため。 |

**サンプルループが終わるまで一切書き込まない**ため、途中で止めても前回の出力は壊れません。
出力: `top_graph_nodes.tsv`、`top_channel_<alias>.tsv`、`attributions.npz`、`ig_summary.json`。

---

## 8. 前処理コマンド（`pathwaygnn-data`）

これらは `dataset:` ブロックを持ちません。生データを読んで**汎用形式を書き出す**だけです。

### `tr-build-processed`（`configs/tr/build_processed.yaml`）

公開ソース（`data_tr/raw`）から中間バンドル `data_tr/processed/` を作ります。
既にバンドルがある場合は不要です。詳細は [README_data_tr.md](README_data_tr.md)。
**h5py が必要**です（`pip install -e '.[tr-upstream]'`）。

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `raw_dir` | str | `data_tr/raw` | 公開ソースの置き場。 |
| `output_dir` | str | `data_tr/processed` | 書き出し先。 |
| `gctx` | str | `GSE92742_..._n473647x12328.gctx` | LINCS L1000 Level 5 行列（GCTX＝HDF5）。 |
| `sig_info` / `gene_info` | str | `GSE92742_Broad_LINCS_{sig,gene}_info.txt` | シグネチャ条件表とランドマーク遺伝子表。 |
| `creeds` | str | `disease_signatures-v1.0.json` | CREEDS の手動疾患シグネチャ。 |
| `hgnc` | str | `hgnc_complete_set.txt` | 遺伝子シンボル変換表の元。 |
| `graph_sif` | str | `PathwayCommons12.All.hgnc.sif` | パスウェイグラフ。 |
| `kegg_omim` / `disease_ontology` | str | `kegg_disease_omim.list` / `HumanDO.obo` | KEGG DISEASE → OMIM → DOID の変換に使う。 |
| `per_cell_line` | bool | `true` | `true` で摂動プロファイルを `(pert_iname, cell_id)` 単位にする。`false` は細胞株を平均で潰す。 |
| `human_only` | bool | `true` | `true` で CREEDS の `organism == "human"` だけを使う。 |
| `labels_dir` | str | `raw_dir` | DOID 変換済みラベルの置き場。`kegg_labels` が無いときここの2ファイルを現行 HGNC で再変換して使う。 |
| `kegg_labels` | dict | （任意） | `inhibitory` / `activatory` → KEGG ID 版ラベル `.txt`。与えると KEGG→DOID 変換の経路が走る。 |

`per_cell_line` と `human_only` はデータ内容そのものを変えるため、`build_manifest.json` に記録されます。

### `tr-prepare`（`configs/tr/prepare.yaml`）

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `source_dir` | str | （必須） | `data_tr/processed`。必要な6ファイルを自動検出する（旧キー `raw_dir` も可）。 |
| `output_dir` | str | （必須） | 書き出し先。 |
| `cutoff` | float | `1e-7` | シグネチャの絶対値がこの値未満のエントリを疎表現から落とす閾値。 |

### `cancer-build-processed`（`configs/cancer/build_processed.yaml`）

生の TCGA データから中間バンドル `data_cancer/processed/` を作ります。
既にバンドルがある場合は不要です。詳細は [README_data_cancer.md](README_data_cancer.md)。

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `graph_sif` | str | （必須） | PathwayCommons の SIF。同梱設定は `data_cancer/PathwayCommons12.All.hgnc.sif`（`data_tr/raw/PathwayCommons12.All.hgnc.sif` と同一内容。ヘッダ付きの `.txt` 版も可）。 |
| `hgnc_table` | str | （必須） | HGNC complete-set エクスポート。`Approved symbol` と `Ensembl gene ID` の両方を使う。 |
| `expression` | str | （必須） | recount2 のカウント行列（`counts_gene.tsv`）。 |
| `gene_ids` | str | （必須） | `expression` の**行の並び順**に対応する遺伝子 ID の一覧。第2列に `bp_length` を置くと `log1p_tpm` が使える。**同梱していない**。 |
| `metadata` | str | （必須） | `TCGA_ID.tsv`。列 UUID → 患者バーコードの対応。 |
| `clinical` | str | （必須） | `mmc1.csv`（TCGA-CDR）。 |
| `gene_sets` | list[str] | （必須） | がん関連遺伝子の母集合。`.gmt` は3列目以降、それ以外は1列目を遺伝子シンボルとして読み、和集合を取る。**同梱していない**。 |
| `ensembl_to_hgnc` | str | （任意） | `cancer-map-ids` の出力。`hgnc_table` より優先して参照される。 |
| `output_dir` | str | （必須） | `data_cancer/processed`。 |
| `years` | list[int] | `[1,2,3,4,5]` | 生成する検証年。 |
| `transform` | str | `log1p` | `log1p`: カウントの自然対数（公開バンドルと同じ）。`log1p_tpm`: TPM の自然対数（論文の記述。`gene_ids` に長さが必要）。 |
| `max_survival_days` | float\|null | `null` | 長期生存の除外閾値。`null` なら論文の規則（「中央値超の打ち切り例 ∪ 死亡例」の95パーセンタイル）から導出する。実データでは論文の 3,595 日を再現する。 |

### `cancer-prepare`（`configs/cancer/prepare.yaml`）

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `source_dir` | str | （必須） | `data_cancer`。実際に読むのは `<source_dir>/processed/`。 |
| `output_dir` | str | （必須） | |
| `years` | list[int] | `[1,2,3,4,5]` | 変換する検証年。年ごとに channel とタスクが1つずつできる。 |
| `num_genes` | int | `4448` | dense 行列の遺伝子数。実データと合わなければ失敗する。 |
| `strict_sample_counts` | bool | `true` | 年別サンプル数を `PAPER_SAMPLE_COUNTS` と照合し、不一致なら失敗する。`cancer-build-processed` で作り直したバンドルは数件ずれるので `false` にする（警告を出して続行）。 |

### `cdr-prepare`（`configs/cdr/prepare.yaml`）

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `source_dir` | str | （必須） | GraphCDRScan の処理済みバンドル（`data_cdr/processed/full_features`）。 |
| `output_dir` | str | （必須） | |
| `binary_mutations` | bool | `false` | `false`: `mutation` channel の値は遺伝子ごとの変異数。`true`: 変異があれば 1.0。 |
| `tasks` | list[str] | `[sensitive_drugwise, sensitive_global]` | 生成するタスク。`LN_IC50` の二値化基準が異なる（詳細は [README_data_cdr.md](README_data_cdr.md)）。 |

### `cancer-map-ids`（`configs/cancer/id_mapping.yaml`）

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `input_path` | str | （必須） | 順序付き Ensembl ID リスト。 |
| `output_path` | str | （必須） | HGNC ID への対応表。 |
| `cache_path` | str | （必須） | MyGene.info 応答の固定キャッシュ。**入力リストの SHA-256 が一致しないキャッシュは再利用を拒否**する。 |
| `species` | str | `human` | |
| `batch_size` | int | `1000` | MyGene.info への1回の問い合わせ件数。 |

---

## 9. レポートコマンド

3つのレポートはいずれも `pathwaygnn_datasets/document.py` を通して Markdown と単体 HTML を
同じ本文から生成するため、両者が食い違うことはありません。**`docs/` 以下は `docs/papers/` を
除きすべて生成物**です。編集するなら Markdown ではなくレポートモジュールを直してください。

### 共通

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `output_dir` | str | （必須） | TSV と PNG の出力先。 |
| `docs_dir` | str | `docs` | `<docs_dir>/<document>.{md,html}` と `<document>_assets/` の出力先。 |
| `pretrain_history` | str | データセット別 | `pretrain` の `history.json`。 |
| `ig_dir` | str | データセット別 | `ig` の出力ディレクトリ群の親。 |

### `tr-report` / `cdr-report`

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `cv_dir` | str | `outputs/<ds>/cv` | |
| `finetune_dir` | str | `outputs/<ds>/finetune` | `<finetune_dir>/<task>/metrics.json` を読む。 |
| `benchmark_dir` | str | `outputs/<ds>/benchmark` | `<benchmark_dir>/<task>/benchmark.json` を読む。 |
| `document` | str | `tr_report` / `cdr_report` | 生成する文書名。 |
| `top_k` | int | `20` | 表と図に載せる IG ランキング件数。 |
| `top_diseases` | int | `15` | **tr のみ**。疾患別 AUC 表に載せる件数。 |
| `top_sites` | int | `19` | **cdr のみ**。原発部位別 AUC 表に載せる件数。 |
| `gene_symbols` | str | `data_cdr/raw/EnsemblToHGNC.tsv` | **cdr のみ**。ノード名（HGNC 数値 ID）を承認シンボルに変換する対応表。存在しなければ ID のまま表示する。 |

### `cancer-report`

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `run_dir` | str | （必須） | Table 1 グリッドの `cv` 出力（`outputs/cancer/table1`）。 |
| `sweep_dir` | str | `outputs/cancer/pretraining_sweep` | 事前学習エポック別 CV 結果。**`outputs/cancer/pretrain_sweep`（チェックポイント側）と紛らわしいので注意**。 |

文書名は `cancer_reproduction` に固定で、`document` キーはありません。

---

## 10. 設定ファイル一覧

| ファイル | コマンド |
| --- | --- |
| `configs/tr/{dataset,prepare,pretrain,cv,report}.yaml` | tr の基本一式 |
| `configs/tr/{finetune,benchmark,ig}_{kd_inh,oe_act}.yaml` | tr のタスク別設定（`_oe_act` は `_kd_inh` を `defaults` で継承） |
| `configs/cancer/{dataset,build_processed,prepare,pretrain,pretrain_sweep,cv,ig,report,id_mapping}.yaml` | cancer 再現一式 |
| `configs/cdr/{dataset,prepare,pretrain,cv,report}.yaml` | cdr の基本一式 |
| `configs/cdr/{finetune,benchmark,ig}_{drugwise,global}.yaml` | cdr のタスク別設定（`_global` は `_drugwise` を継承） |
| `configs/cdr/upstream.json` | **YAML ではない**。`scripts/cdr/upstream/prepare_data.py` が読む GraphCDRScan 由来の JSON |

---

## 11. よくある落とし穴

- **キーのタイプミスは沈黙する。** スキーマ検証がないため、`traning:` と書いても
  エラーにならず、`training` の既定値が全部使われます。実行ログの1エポック目の値が
  想定と違うときはまずここを疑ってください。
- **`variants:` は部分上書きできない。** `defaults` で継承したうえで `variants:` を書くと、
  リスト全体が置き換わります。
- **`model.embedding_dim` はグラフ使用時には効かない。** encoder の `hidden_dim` が優先されます。
  埋め込み次元を変えたいなら `pretrain` からやり直す必要があります。
- **`cv` の再開はディレクトリの有無で決まる。** 設定を変えても fold ディレクトリが残っていれば
  古い結果が再利用されます。パラメータを変えたら該当ディレクトリを消してください。
- **`pretrained_checkpoint` はノード数・関係数を照合する。** データセットを作り直したら
  事前学習もやり直しです（`load_encoder` が明示的に失敗します）。
- **`benchmark` に `device:` を書いても無視される。** sklearn / XGBoost は CPU で走ります。
