# PathwayGNN

PathwayGNN は次の4実装を、現行 PyTorch / PyTorch Geometric API で再設計した統合プロジェクトです。

- `SLGCN-TR`: PathwayCommons と摂動・疾患発現シグネチャ、KD–inhibitory / OE–activatory 分類
- `SampleLevelGNN`: エッジ予測によるグラフ事前学習とサンプルレベル集約
- `DistributedGNN`: `torchrun`、`torch.distributed`、DDP による分散事前学習
- `GraphCDRScan`: Reactome機能相互作用グラフと GDSC/CCLP による細胞株×化合物の薬剤感受性予測

旧コードは依存関係として取り込まず、入出力と実験機能を参照して `src/pathwaygnn` にスクラッチ実装しています。

## 関連ドキュメント

| 文書 | 内容 |
| --- | --- |
| [README_config.md](README_config.md) | `configs/` の YAML 全項目のリファレンス（既定値・読むコマンド・落とし穴） |
| [README_data_tr.md](README_data_tr.md) | `data_tr`（target repositioning）の元データと前処理 |
| [README_data_cancer.md](README_data_cancer.md) | `data_cancer`（TCGA がん予後）の元データと前処理 |
| [README_data_cdr.md](README_data_cdr.md) | `data_cdr`（GDSC 薬剤感受性）の元データと前処理 |
| [docs/tr_report.md](docs/tr_report.md) / [docs/cancer_reproduction.md](docs/cancer_reproduction.md) / [docs/cdr_report.md](docs/cdr_report.md) | 各データセットの実行・評価結果（生成物） |

## 構成: 前処理と学習エンジンの分離

データセット固有の処理は学習エンジンから完全に分離しています。

| パッケージ | 役割 | CLI |
| --- | --- | --- |
| `pathwaygnn_datasets` | データセット固有。生データを読み、**汎用データ形式**を書き出す。論文固有のレポートも含む | `pathwaygnn-data` |
| `pathwaygnn` | データセット非依存の学習エンジン。汎用形式しか知らない | `pathwaygnn` |

前処理をすべて終えてから学習を開始します。

```bash
pathwaygnn-data tr-build-processed --config configs/tr/build_processed.yaml
                                                                      # data_tr/raw       -> data_tr/processed
pathwaygnn-data tr-prepare     --config configs/tr/prepare.yaml       # data_tr/processed -> data_tr/prepared
pathwaygnn-data cancer-build-processed --config configs/cancer/build_processed.yaml
                                                                      # rawdata_TCGA/     -> data_cancer/processed
pathwaygnn-data cancer-prepare --config configs/cancer/prepare.yaml   # data_cancer/...   -> data_cancer/prepared
pathwaygnn-data cdr-prepare    --config configs/cdr/prepare.yaml      # data_cdr/processed -> data_cdr/prepared
pathwaygnn pretrain --config configs/tr/pretrain.yaml                 # 以降は汎用形式のみを読む
```

### 3つのデータセットの切り替え

`data_tr`（target repositioning）、`data_cancer`（TCGA予後）、`data_cdr`（GDSC薬剤感受性）の
切り替えは、設定ファイルの
`dataset:` ブロックだけで決まります。ブロックは各データセットの `dataset.yaml` に一本化され、
実験設定は `defaults:` で取り込みます。

```yaml
# configs/cancer/dataset.yaml
dataset:
  name: cancer            # data_cancer/prepared/dataset.json と照合される
  dir: data_cancer/prepared
```

```yaml
# configs/cancer/cv.yaml
defaults:
  - dataset.yaml
dataset:
  tasks: [1year, 2year, 3year, 4year, 5year]
```

`name` は `dataset.json` と照合されるため、`dir` を取り違えた設定はデータを読んだ時点で失敗します。
出力も `outputs/tr/...`、`outputs/cancer/...`、`outputs/cdr/...` に分かれ、設定は `configs/tr/`、
`configs/cancer/`、`configs/cdr/`、スクリプトは `scripts/tr/`、`scripts/cancer/`、`scripts/cdr/`
に分かれています。

### 汎用データ形式

前処理の到達点は次の1形式です（詳細は `src/pathwaygnn/data/format.py`）。

```text
<root>/dataset.json                 マニフェスト。name がデータセットを識別する
<root>/graph.pt                     {"edge_index", "edge_type"}
<root>/nodes.json, relations.json   ノード名、関係名
<root>/channels/<channel>/          遺伝子-値テーブル（sparse=CSR / dense=memmap行列）
<root>/tasks/<task>/                labels.npy, groups.npy, covariates.npy, rows/<alias>.npy
```

- **channel** はサンプルの遺伝子-値表現（摂動シグネチャ、疾患シグネチャ、発現プロファイル）。
  データセット単位のテーブルなので、複数タスクが同じ表を共有できます。
- **task** は1つの二値予測問題。channel を局所名（alias）に束ね、サンプルから行への写像を持ちます。
  同じ問題を別サンプル集合に定義したタスク（1year…5year）は同じ alias を使うため、
  モデル設定がそのまま流用できます。
- **groups** はサンプル単位のグループ（がん種、対象疾患）。グループ別AUCや帰属の集計に使われます。
- **covariates** は密な共変量ベクトル（がん種one-hot）。

## 設計

1. PathwayCommons の13種の関係ごとに GINConv を適用し、関係別出力を加算する。
2. DistMult スコアと負例ノード置換によりエッジ予測を事前学習する。
3. 各 channel を遺伝子軸で集約し、共変量分岐と連結してサンプルごとに二値分類する。
4. accuracy、ROC-AUC、precision、recall、F1 を保存する。

サンプルレベルのヘッド `SampleLevelModel` は1つで両データセットを表現します。`block: paper`
（Linear-ELU-BN-Linear-ELU-BN）が論文のがん生存アーキテクチャ、`block: plain`
（Linear-ELU-Dropout-Linear）が target repositioning のブロックです。dense channel は
`reshape().sum()`、sparse channel は `index_add_` で集約し、いずれも遺伝子軸の総和になります。

グラフ事前学習では、各 DDP rank が同一の関係グラフ上で異なる正例・負例エッジをサンプリングし、勾配を all-reduce します。グローバルバッチサイズは
`WORLD_SIZE × training.batch_size` です。

PyTorch Lightning は使用していません。PyG の関係別全グラフ forward、rank 別エッジサンプリング、分散集約、rank 0 のチェックポイント保存を明示的に制御する方が、この用途では障害解析と再現性に優れるためです。サンプルレベル学習も同じネイティブ PyTorch ループに統一しています。

## 環境

既存の conda 環境 `gnn` を使います。現在の環境を更新する場合:

```bash
conda env update -n gnn -f environment.yml --prune
conda activate gnn
pip install -e .
```

要求バージョンは Python 3.11+、PyTorch 2.6+、PyG 2.6+ です。旧実装の PyTorch 1.x / PyG 1.x API は使用しません。CPU のみの場合は `environment.yml` の `pytorch-cuda` を除き、使用環境に合う公式 PyTorch パッケージを導入してください。

## データ準備

`data_tr` は3段階です。`raw`（公開ソース 22 GB）→ `processed`（中間バンドル）→ `prepared`（汎用形式）。
`processed` が既にあれば③だけで済みます。詳細は [README_data_tr.md](README_data_tr.md)。

```bash
python -m scripts.tr.upstream.download_raw_data   # ① 公開ソース取得 -> data_tr/raw
bash scripts/tr/build_processed.sh                # ② 要 .[tr-upstream] -> data_tr/processed
bash scripts/tr/prepare.sh                        # ③ -> data_tr/prepared
```

このスクリプトはリポジトリの場所に依存せずプロジェクトルートへ移動してから、次のCLIと同じ処理を実行します。

```bash
pathwaygnn-data tr-prepare --config configs/tr/prepare.yaml
```

`data_tr/processed` から以下を自動検出します。

```text
graph.tsv
disease_specific_signature.tsv
knockdown_signature.tsv
overexpression_signature.tsv
inhibitory_target_disease.tsv
activatory_target_disease.tsv
```

生成物は channel（`disease`, `perturbation_kd`, `perturbation_oe`）、task（`kd_inh`, `oe_act`）、
および件数と除外行を記録した `dataset.json` / `task.json` です。

## グラフ事前学習

単一 GPU / CPU:

```bash
pathwaygnn pretrain --config configs/tr/pretrain.yaml
```

単一ノード複数 GPU:

```bash
NPROC_PER_NODE=4 bash scripts/tr/pretrain_distributed.sh
```

複数ノードでは各ノードで同じ共有データを参照し、標準の `torchrun` 引数を使います。

```bash
torchrun \
  --nnodes=2 --nproc-per-node=4 \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
  -m pathwaygnn.cli pretrain --config configs/tr/pretrain.yaml
```

`outputs/tr/pretrain/best.pt` と `last.pt` は rank 0 のみが保存します。NCCL は GPU、Gloo は CPU で自動選択されます。

## サンプルレベルの学習

### 単一分割（train / validation / test）

```bash
pathwaygnn finetune --config configs/tr/finetune_kd_inh.yaml
pathwaygnn finetune --config configs/tr/finetune_oe_act.yaml
```

既定では事前学習済み encoder を凍結します。end-to-end 学習は設定を変更します。

```yaml
training:
  train_encoder: true
```

ラベルは層化して train / validation / test に分け、学習データのクラス比から `pos_weight` を設定します。validation ROC-AUC で早期終了し、`outputs/tr/finetune/<task>/metrics.json` に全履歴、test 指標、再現可能な分割 index を保存します。

### 層化k分割交差検証（`cv`）

`cv` はデータセット非依存です。`variants × tasks × folds` のグリッドを実行し、fold単位で再開でき、
グループ別AUCも出力します。がん論文の Table 1 と同じ仕組みで target repositioning も評価できます。

評価値は **ROC-AUC と、閾値 0.5 での accuracy / precision / recall / F1** を、エポック単位
（`history` の `test_*`）、fold 単位（`metrics.json`）、条件単位（`summary.json` の
`mean_*` / `std_*` / `fold_*`）で保存します。モデル選択は ROC-AUC のみで行います。
詳細は [README_config.md](README_config.md#保存される評価値) を参照してください。

```bash
pathwaygnn cv --config configs/tr/cv.yaml        # kd_inh と oe_act、グラフ有無の2 variant
pathwaygnn cv --config configs/cancer/cv.yaml    # 5年 × 4 variant（Table 1）
```

出力は `<output_dir>/<task>/<variant>/fold_<k>/{metrics.json,predictions.npz,model.pt}` と
条件ごとの `summary.json` です。

### Integrated Gradients（`ig`）

`ig` もデータセット非依存です。グラフノード埋め込み、各 channel の値、共変量に対する帰属を計算し、
データセット自身のノード名でランキングを書き出します。

```bash
pathwaygnn ig --config configs/tr/ig_kd_inh.yaml        # kd_inh の fold を帰属
pathwaygnn ig --config configs/cancer/ig.yaml    # 5年 gnn_dnn_cancer の fold 0
```

`top_graph_nodes.tsv`、`top_channel_<alias>.tsv`、`attributions.npz`、`ig_summary.json` を生成し、
`per_group_rankings: true` のときはグループ別のノードランキングも出力します（次数とIGの
Pearson相関も記録します）。

### レポート（`tr-report`）

上記すべての成果物（事前学習、cv、finetune、benchmark、ig）を1つの文書にまとめます。

```bash
pathwaygnn-data tr-report --config configs/tr/report.yaml
```

表は `outputs/tr/report/*.tsv`、図は `outputs/tr/report/*.png`、文書は
[`docs/tr_report.md`](docs/tr_report.md) と [`docs/tr_report.html`](docs/tr_report.html)
（図は `docs/tr_report_assets/`）に生成されます。データ監査、cv（グラフ有無のアブレーション）、
グラフなしベースラインとの比較、holdout、疾患別AUC、IGランキングを含みます。

## 比較実験

SLGCN-TR と同様に、グラフを使わない Logistic Regression、Random Forest、XGBoost を同じ疎な摂動・疾患シグネチャで層化5-fold比較できます。

```bash
pathwaygnn benchmark --config configs/tr/benchmark_kd_inh.yaml
pathwaygnn benchmark --config configs/tr/benchmark_oe_act.yaml
```

各 fold と平均の accuracy、ROC-AUC、precision、recall、F1 を `outputs/tr/benchmark/<task>/benchmark.json` に保存します。比較実験には `scikit-learn` と `xgboost` が必要です。

## Inoue et al. がん予後論文の再現

`data_cancer/` を用いた1～5年TCGA生存予測、4モデルの5-fold比較、
がん種別AUC、Integrated Gradientsを実装しています。

```bash
# 前処理、3 GPU分散事前学習、Table 1全20条件、IG、全レポート
bash scripts/cancer/reproduce_paper.sh full

# Table 1のみ（20条件を3 GPUへ分配し、fold単位で再開可能）
python scripts/cancer/reproduce_table1.py --gpus 0,1,2
```

結果は `outputs/cancer/report/` に、同一内容の再現文書は
[`docs/cancer_reproduction.md`](docs/cancer_reproduction.md) と
[`docs/cancer_reproduction.html`](docs/cancer_reproduction.html) に生成されます。
順序付きEnsembl IDリストが別途存在する場合は
`pathwaygnn-data cancer-map-ids --config configs/cancer/id_mapping.yaml` で、
MyGene.info応答を固定キャッシュしながらHGNC IDへ変換できます。

## がん薬剤感受性の統合（GraphCDRScan）

`data_cdr/` は GraphCDRScan の `data/` をそのまま持ち込んだものです。前処理も
`scripts/cdr/upstream/` に取り込んであり、PathwayGNN 内で完結します（データ根だけを
`data/` から `data_cdr/` に変更し、変換内容・エンコード・出力レイアウトは上流と同一です）。

```bash
# 上流ステージ: 公開データの取得と data_cdr/processed/ の再生成（既に生成済みなら不要）
# 専用の依存関係が必要: pip install -e '.[cdr-upstream]' と pdftotext / LibreOffice
python -m scripts.cdr.upstream.download_raw_data
python -m scripts.cdr.upstream.prepare_data --config configs/cdr/upstream.json

# 汎用形式への変換（numpy のみで完結し、上流の依存関係は不要）
bash scripts/cdr/prepare.sh          # data_cdr/processed/full_features -> data_cdr/prepared
```

サンプルは *(細胞株, 化合物)* の組で、汎用形式へは次のように写像されます。

- channel `mutation`（sparse）: 細胞株の Cancer Gene Census 遺伝子ごとの変異数。プロファイルは
  細胞株にしか依存しないため、107,418サンプルは760行を `rows/mutation.npy` 経由で共有します。
- covariates: GraphCDRScan のサンプル特徴量そのまま（96/78/83変異スペクトル、原発部位one-hot、
  3×1024ビット化合物フィンガープリント＝3,348次元）。
- groups: 原発部位（19種）。グループ別AUCはがん種別AUCになります。
- task: `LN_IC50` の二値化。`sensitive_drugwise`（同一化合物の中央値で分割＝細胞株のゲノムだけが
  手がかりになる）と `sensitive_global`（全体中央値で分割＝化合物の効力が支配的）。

学習・評価はすべて汎用の `pathwaygnn` で実行します。

```bash
bash scripts/cdr/reproduce.sh        # pretrain / cv / finetune / benchmark / ig / report
```

`cv` はグラフ有無 × 共変量有無の2因子アブレーション（`mlp`, `mlp_cov`, `gnn_mlp`, `gnn_mlp_cov`）
です。結果は `outputs/cdr/report/` と、同一内容の
[`docs/cdr_report.md`](docs/cdr_report.md) / [`docs/cdr_report.html`](docs/cdr_report.html)
に生成されます。

## 設定と再現性

YAML の `defaults` は現在の設定ファイルからの相対パスです。子設定は辞書を再帰的に上書きします。
**全項目の説明は [README_config.md](README_config.md)** にあります。主要設定は以下です。

- `dataset.name`, `dataset.dir`: 使用するデータセット（`dataset.json` と照合）
- `dataset.task` / `dataset.tasks`: 対象タスク（`kd_inh`, `oe_act`, `1year`…`5year`,
  `sensitive_drugwise`, `sensitive_global`）
- `seed`: Python / PyTorch と分割の乱数 seed
- `model.hidden_dim`, `model.num_layers`: グラフ encoder
- `model.block`, `model.batch_norm`: サンプルレベルヘッドの構造
- `training.batch_size`: rank ごとの事前学習エッジ数、またはサンプル数
- `training.train_encoder` / `training.end_to_end`: グラフ encoder も更新するか
- `variants[].use_graph`, `variants[].use_covariates`: `cv` のアブレーション条件

fold の seed は `seed + task.seed_offset * 1000 + fold + variant.seed_index * 100` です。
`seed_offset` はタスク（がんでは検証年）が持ち、`seed_index` は variant が持つため、
1条件だけを単独実行してもグリッド全体と同じ seed になります。

## テスト

```bash
conda run -n gnn python -m pytest
```

テストは小さな raw データと合成データセットを生成し、前処理、汎用形式の読み書き、関係別 GIN、
エッジ事前学習、dense/sparse channel の等価性、可変長サンプル集約、逆伝播、層化分割、
`cv` グリッドと fold 再開、`ig` の出力を検証します。

## Git

このディレクトリを独立リポジトリとして管理できます。

```bash
git init
git add .
git commit -m "Initial PathwayGNN implementation"
```

`prepared` データ、学習結果、チェックポイントは `.gitignore` の対象です。
`raw` と `processed` も容量のため対象外です（`data_tr` 21.5 GB、`data_cdr` 1.6 GB / 9.3 GB。
COSMIC のように登録が必要な入力もあります）。いずれも取得スクリプトと build コマンドで再生成します。
