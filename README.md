# PathwayGNN

パスウェイグラフを事前学習し、その埋め込みを使って
サンプル単位の予測を行う実装です。

> **まず動かす人へ:** 実データは容量とライセンスの都合で Git に入っていません。
> クローン直後に動くのは **`data_sample/`**（60サンプル・20遺伝子の合成データ）です。
> **§4 チュートリアル** を順に実行すると、
> 前処理から帰属解析まで **CPUで約2分**で一周できます。実データはその応用として **§7 応用** で説明します。

## ドキュメント一覧

| 文書 | 内容 |
| --- | --- |
| [このREADME](README.md)  | 全体像、チュートリアル、各データセットの使い方 |
| [README: config](README_config.md) | `configs/` の YAML 全項目のリファレンス（既定値・設定値の解説） |
| [README: data_sample](data_sample/README.md) | チュートリアル用データの詳細（生成規則、列レイアウト、汎用形式との対応） |
| [README: embedding_pc](embedding_pc/README.md) | PathwayCommons ノードの外部埋め込み（ESMC-6B / Uni-Mol）の作り方。使い方は §5.1 |
| [README: data_tr](README_data_tr.md) / [data_tr/README.md](data_tr/README.md) | target repositioning 応用例のための元データと前処理 |
| [README: data_cancer](README_data_cancer.md) / [data_cancer/README.md](data_cancer/README.md) | TCGA を使ったがん予後予測の応用例のための元データと前処理 |
| [README: data_cdr](README_data_cdr.md) / [data_cdr/README.md](data_cdr/README.md) | GDSC を使った薬剤感受性予測応用例のための元データと前処理 |
| [Report: tr](docs/tr_report.md)  | target repositioning 応用例の評価結果 |
| [Report: cancer](docs/cancer_reproduction.md)| TCGA を使ったがん予後予測の応用例の評価結果 |
| [Report: cdr](docs/cdr_report.md)  | GDSC を使った薬剤感受性予測応用例の評価結果 |
| [Report: dist](docs/dist_report.md) | グラフ分割の分散実行の評価結果 |

---

## 1. 全体像

### 1.1 前処理と学習エンジン

データセット固有の前処理部分と学習エンジンは分離されています。

| パッケージ | 役割 | CLIコマンド |
| --- | --- | --- |
| `pathwaygnn_datasets` | 1つのコーパスの事情（ファイル名、列レイアウト、ID規約、論文の参照値）とレポート | `pathwaygnn-data` |
| `pathwaygnn` | **汎用データ形式だけ**。データセット名すら本質的に知らない | `pathwaygnn` |
`pathwaygnn_datasets` は `pathwaygnn.data.format` を import しますが、エンジンは `pathwaygnn_datasets` を import しません。

```text
生データ ──前処理──▶ prepared/（汎用形式） ──▶ pretrain ─▶ cv / finetune / benchmark / ig ─▶ report
         pathwaygnn-data                        └────────── pathwaygnn ──────────┘   pathwaygnn-data
```

前処理は最後まで終わらせてから学習に入ります。学習コマンドは `prepared/` しか読みません。

### 1.2 4つのデータセット

| データセット | 内容 | ノード数 / 関係数 | タスク | サイズ | Git |
| --- | --- | --- | --- | --- | --- |
| `sample` | チュートリアル用の合成データ | 20 / 3 | `responder`（60）、`relapse`（48） | 12 KB | 本リポジトリ内 |
| `tr` | 摂動シグネチャ × 疾患シグネチャで創薬標的を分類 | 30,895 / 13 | `kd_inh`（61,101）、`oe_act`（3,465） | raw 21.5 GB | 外部 |
| `cancer` | TCGA 発現プロファイルから n 年生存を予測 | 30,918 / 13 | `1year`…`5year`（4,492〜9,484） | raw 2.6 GB | 外部 |
| `cdr` | GDSC の (細胞株, 化合物) の薬剤感受性 | 13,606 / 356 | `sensitive_drugwise`、`sensitive_global`（各107,418） | raw 1.6 GB | 外部 |

各データセットの設定は `dataset.yaml`内の `dataset:` ブロックに一本化され、各実験設定（例えば下は`configs/sample/cv.yaml`の例）の `defaults:` で取り込みます。
`name` はデータセット内の `prepared/dataset.json` と照合され、`dir` を取り違えた設定はデータを読んだ時点でエラーを出します。
出力も `outputs/sample/…`、`outputs/tr/…` のように分かれます。

```yaml
# configs/sample/dataset.yaml
dataset:
  name: sample                  # data_sample/prepared/dataset.json と照合される
  dir: data_sample/prepared
```

```yaml
# configs/sample/cv.yaml
defaults:
  - dataset.yaml
dataset:
  tasks: [responder, relapse]
```

---

## 2. セットアップ

```bash
conda env update -n gnn -f environment.yml --prune
conda activate gnn
pip install -e .                  # pathwaygnn と pathwaygnn-data の2コマンドが入る
pip install -e '.[benchmark]'     # 任意: scikit-learn / xgboost（cv とベースラインで使用）
pip install -e '.[hub]'           # 任意: huggingface-hub（`pathwaygnn hg` と hf:// 参照で使用）
```

Python 3.11+、PyTorch 2.6+、PyG 2.6+ が必要です。
CPU のみの場合は `environment.yml` の `pytorch-cuda` を外し、環境に合う公式 PyTorch を入れてください。
チュートリアルは GPU 不要での動作になっています（`configs/sample/*.yaml` は `device: cpu`）。

`pathwaygnn` / `pathwaygnn-data` コマンドは `python -m pathwaygnn.cli` / `python -m pathwaygnn_datasets.cli`
と等価です（`torchrun` から使うときは後者）。

---

## 3. 汎用データ形式

前処理後のデータは以下の形式です（定義は `src/pathwaygnn/data/format.py`、実例は [data_sample/README.md](data_sample/README.md) の §4を参照）。
（自前で用意する場合も以下ファイルを用意することで動作します）

```text
<root>/dataset.json                 マニフェスト。name がデータセットを識別する
<root>/graph.pt                     {"edge_index": int64[2,E], "edge_type": int64[E]}
<root>/nodes.json, relations.json   ノード名、関係名（index → 名前）
<root>/node_features/<name>/        遺伝子-値テーブル
                                      sparse: ptr/gene/value.npy（ノードindexのCSR）
                                      dense:  matrix.npy [rows,genes] + gene_index.npy
<root>/tasks/<task>/                labels.npy, groups.npy?, sample_features.npy?, rows/<alias>.npy?
```

複雑なデータセットの準備では以下の4つに気を付ける必要があります。

- **node-level feature**: サンプルの「遺伝子ごとの値」表現（発現プロファイル、摂動シグネチャ、変異数など）。データセット単位の表なので複数タスクで共有できます。
- **task**: 1つの(二値)予測問題。`task.json` の `node_features` が局所名（alias）→ データセットの表を対応づけます。同じ alias を使うタスク同士は
  モデル設定をそのまま流用できます（例：`cancer` の`1year`…`5year` が1つの設定で回る理由）。
- **rows/`<alias>`.npy**: サンプル → 表の行。ファイルが無ければ恒等写像。
  多数のサンプルが少数の行を共有する場合（例：`cdr` の107,418サンプル→760行）に使います。
- **groups** / **sample-level feature**: サンプル単位のグループ（がん種、生体組織、対象疾患）と、
  遺伝子に紐づかない密ベクトル（年齢、変異スペクトル、化合物フィンガープリント）。
  前者はグループ別AUC・グループ別帰属を、後者はモデルの sample-level の特徴に使用します。

書き込みは `DatasetWriter` 経由のみです（`finish()` がノード範囲・行範囲・形状・dense テーブルの完全性を検証します）。

---

## 4. チュートリアル: `data_sample` で全機能を一周する

```bash
bash scripts/sample/run_all.sh        # ①〜⑧をまとめて（CPU、約2分）
```

以下は各段が何をしているかの解説です。1段ずつ実行しても同じです。
数値はすべて実測値（1コアCPU）で、乱数の都合で ±0.02 程度ぶれます。

### 4.0 教材合成データ: 60サンプル・20遺伝子・27エッジ

`data_sample/raw/` に TSV が4つあります。

| ファイル | 内容 | 汎用形式では |
| --- | --- | --- |
| `graph.tsv` | 27行の3列 SIF（起点 / 関係 / 終点） | グラフ（20ノード、3関係） |
| `expression.tsv` | 60サンプル × 20遺伝子の発現量（横長） | **dense** node-level feature `expression` |
| `tissue_signature.tsv` | 生体組織3種 × マーカー8遺伝子（縦長） | **sparse** node-level feature `tissue_signature` |
| `samples.tsv` | 生体組織3種・年齢・性別・病期・喫煙・ラベル2列 | groups、sample-level features、task 2つ |

遺伝子名が役割を表しています。

| モジュール | 役割 |
| --- | --- |
| `GROWTH1`…`GROWTH7` | `responder` を**正**に決める |
| `IMMUNE1`…`IMMUNE7` | `responder` を**負**、`relapse` を**正**に決める |
| `NOISE1`…`NOISE6` | どのラベルにも効かない（帰属解析が拾ってはいけない遺伝子） |

ラベルは各モジュールの関数として以下の関数で作られています（人工的なラベル）。

```text
responder = 1  if  2.0*growth - 2.0*immune + 0.8*z(stage) + N(0,0.3²) > 中央値   （60サンプル、陽性30）
relapse   = 1  if  2.0*immune - 0.04*(age-62)            + N(0,0.3²) > 中央値   （48サンプル、陽性24）
```

生成規則・列レイアウト・再生成方法の詳細は
[data_sample/README.md](data_sample/README.md) にあります。

### 4.1 ① 前処理: 4つのTSV → 汎用形式

```bash
pathwaygnn-data sample-prepare --config configs/sample/prepare.yaml     # 5 秒
```

```json
{
  "format": "pathwaygnn/1", "name": "sample",
  "num_nodes": 20, "num_relations": 3, "num_edges": 54,
  "node_features": {
    "expression":       {"kind": "dense",  "num_rows": 60, "num_features": 20},
    "tissue_signature": {"kind": "sparse", "num_rows": 3,  "num_values": 24}
  },
  "tasks": ["responder", "relapse"]
}
```

読み方の要点は3つです。

1. **エッジが27→54**: 無向エッジを両方向に展開しています。ノード名・関係名は
   `sorted()` してから番号を振るので、同じ入力なら常に同じインデックスになります
   （事前学習の再現性がこれに依存します）。
2. **`tissue_signature` は3行しかない**: サンプル単位ではなく生体組織（今回は3種）単位の表で、
   60サンプルが `rows/tissue_signature.npy` を通して3行を共有します。
3. **`responder` には `rows/expression.npy` が無い**: 全60サンプルが発現表の60行と
   1対1（恒等写像）なので、ファイルを書きません。`relapse` は48サンプルなので持ちます。

```bash
find data_sample/prepared -type f | sort      # 92 KB、21ファイル
```

実装は `src/pathwaygnn_datasets/sample/prepare.py`（約240行、うち40行は解説）で、最小の前処理実装です。自分のデータを足すときの雛形になります（§4.10）。

### 4.2 ② グラフ事前学習（エッジ予測）

```bash
pathwaygnn pretrain --config configs/sample/pretrain.yaml               # 10 秒
```

各関係ごとの GINConv でノード埋め込みを作り、DistMult スコアで「実在エッジ vs 終点を差し替えた偽エッジ」を識別します。
ラベルは使いません。

```text
best epoch 45, loss 0.5887, edge-ranking accuracy 0.941
```

`outputs/sample/pretrain/` に `best.pt` / `last.pt` / `history.json` が出ます。
`best.pt` のチェックポイントには `model_config`（ノード数・関係数・次元・層数・dropout）が埋め込まれ、ノード数や関係数が違うデータセットに読み込もうとすると拒否されます。

### 4.3 ③ 層化k分割交差検証とアブレーション

```bash
pathwaygnn cv --config configs/sample/cv.yaml                          # 53 秒
```

`variants × tasks × folds` のグリッド（ここでは 4 × 2 × 3 = 24 fold）を回します。

| task | variant | グラフ | sample-level特徴 | ROC-AUC | accuracy | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| `responder` | `mlp` | – | – | 0.370 ± 0.108 | 0.400 | 0.287 |
| `responder` | `mlp_cov` | – | ✓ | 0.500 ± 0.029 | 0.533 | 0.586 |
| `responder` | `gnn_mlp` | ✓ | – | **0.973 ± 0.019** | 0.917 | 0.914 |
| `responder` | `gnn_mlp_cov` | ✓ | ✓ | **0.987 ± 0.012** | 0.917 | 0.917 |
| `relapse` | `mlp` | – | – | 0.750 ± 0.066 | 0.688 | 0.668 |
| `relapse` | `mlp_cov` | – | ✓ | 0.500 ± 0.193 | 0.500 | 0.441 |
| `relapse` | `gnn_mlp` | ✓ | – | **0.917 ± 0.007** | 0.854 | 0.838 |
| `relapse` | `gnn_mlp_cov` | ✓ | ✓ | **0.896 ± 0.037** | 0.750 | 0.770 |

**なぜ `mlp` が偶然（0.5）並みなのか**が、このアーキテクチャを理解する鍵です。
サンプルレベルヘッドは、遺伝子の値をスカラー1つずつ射影してから遺伝子軸で総和します。
`use_graph: false` ではノード埋め込みが足されないため、どの値がどの遺伝子のものか
区別できません（値の集合しか見ていない）。`GROWTH` の平均と `IMMUNE` の平均の差という
ラベル規則は、原理的に表現できません。
ノード埋め込みが遺伝子IDの役割も担っているのであって、グラフ構造の恩恵だけを測っているわけではない、ということです。
遺伝子ごとに重みを持つグラフなしベースラインが見たいときは §4.6 の `benchmark`（Logistic Regression など）を使います。

出力は `outputs/sample/cv/<task>/<variant>/fold_<k>/{metrics.json,predictions.npz,model.pt}` と条件ごとの `summary.json` です。
評価値は ROC-AUC と accuracy / precision / recall / F1 （閾値0.5）をエポック単位（`history` の `test_*`）、fold単位（`metrics.json`）、
条件単位（`summary.json` の `mean_*` / `std_*` / `fold_*`）で保存します。
モデル選択は ROC-AUC のみで行います。

**fold 単位で再開できます**。`metrics.json` / `predictions.npz` / `model.pt` が揃っていれば
その fold は再利用され、`fold_<k>/` を消すとそこだけ再計算されます。

### 4.4 ④ グループ別AUC

`groups`（ここでは生体組織を想定）を持つタスクでは、fold ごとにグループ別 AUC も記録されます。
追加のコマンドは不要で、`metrics.json` の `per_group_auc` に入っています。

```text
fold     TISSUE_A   TISSUE_B   TISSUE_C        （responder / gnn_mlp_cov）
0           1.000      1.000      1.000
1           1.000      1.000      1.000
2           1.000      1.000      1.000
```

実データではこれが**がん種別AUC**（`cancer`）、**原発部位別AUC**（`cdr`）、**疾患別AUC**（`tr`）になります。

### 4.5 ⑤ 単一分割（train / validation / test）

```bash
pathwaygnn finetune --config configs/sample/finetune.yaml               # 9 秒
```

層化して 36 / 12 / 12 に分け、validation AUC で早期終了し、test は最後に1回だけ評価します
（`cv` が held-out fold を毎エポック見るのとは別のプロトコル）。
クラス比`pos_weight` も学習データから設定されます。

```text
finetune (test split)  AUC 0.806  accuracy 0.750  F1 0.769   (38 epochs)
```

`outputs/sample/finetune/responder/metrics.json` に全履歴・test 指標・再現可能な分割 index が入ります。
test が12サンプルしかないので、`cv`（0.987）より値がぶれます。

### 4.6 ⑥ グラフなしベースライン

```bash
pathwaygnn benchmark --config configs/sample/benchmark.yaml             # 9 秒
```

全 node-level feature を `[samples, num_nodes]` に展開して sample-level 特徴と連結し、
同じ特徴量で Logistic Regression / Random Forest / XGBoost を同じ層化k分割で回します。

```text
logistic_regression    AUC 0.990   random_forest AUC 0.957   xgboost AUC 0.977
```

この教材データのラベルは遺伝子の線形和なので、線形モデルがほぼ上限（GNN の 0.987 と同等）です。

### 4.7 ⑦ Integrated Gradients で答え合わせ

```bash
pathwaygnn ig --config configs/sample/ig.yaml                           # 10 秒
```

`cv` が保存した fold のモデルを読み、**グラフ埋め込み行列**・**各 node-level feature の値**・
**sample-level feature** への帰属を計算します（held-out 20サンプル × 25積分ステップ）。

`outputs/sample/ig/responder_fold0/top_node_feature_expression.tsv` の全20行を並べると:

| 順位 | 遺伝子 | 符号付きIG | | 順位 | 遺伝子 | 符号付きIG |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | IMMUNE4 | −0.1912 | | 11 | **NOISE1** | +0.1306 |
| 2 | GROWTH4 | +0.1834 | | 12 | GROWTH7 | +0.1259 |
| 3 | IMMUNE7 | −0.1780 | | 13 | GROWTH6 | +0.1176 |
| 4 | IMMUNE2 | −0.1722 | | 14 | **NOISE4** | +0.1080 |
| 5 | IMMUNE1 | −0.1696 | | 15 | **NOISE2** | −0.1056 |
| 6 | IMMUNE3 | −0.1659 | | 16 | GROWTH3 | +0.0713 |
| 7 | IMMUNE6 | −0.1611 | | 17 | **NOISE3** | −0.0615 |
| 8 | GROWTH2 | +0.1435 | | 18 | GROWTH5 | +0.0437 |
| 9 | IMMUNE5 | −0.1394 | | 19 | **NOISE6** | +0.0059 |
| 10 | GROWTH1 | +0.1337 | | 20 | **NOISE5** | −0.0054 |

生成規則と照合すると:

- 上位10位がすべて因果遺伝子（`GROWTH` 3 + `IMMUNE` 7）
- `NOISE` の最上位は11位
- 符号が規則と一致します。`GROWTH` は7個すべて正、`IMMUNE` は7個すべて負
  （`responder = +growth − immune`）。`NOISE` は符号がばらつき、下位に集まります。
- `sample_feature_ig` の絶対値は小さい（`stage` −0.035、他は ≤0.002）。発現量だけでほぼ説明できてしまうためで、`gnn_mlp` と `gnn_mlp_cov` の AUC 差が小さいことと整合します。

同時に `top_graph_nodes.tsv`（グラフ埋め込みへの帰属。`per_group_rankings: true` なので生体組織別も3つ）、`attributions.npz`、`ig_summary.json` が出ます。
`ig_summary.json` の`degree_ig_pearson_r`（ここでは 0.53）は「帰属が単にノード次数を反映していないか」の健全性チェックです。

### 4.8 ⑧ 学習済みモデルで予測表を出す

```bash
pathwaygnn pred --config configs/sample/pred.yaml      # 数秒
```

```text
sample_index  probability  prediction  label  group     row_expression  row_tissue_signature
------------  -----------  ----------  -----  --------  --------------  --------------------
0             0.547536     1           1      TISSUE_A  0               0
1             0.332248     0           0      TISSUE_B  1               1
2             0.537786     1           0      TISSUE_C  2               2
3             0.755366     1           0      TISSUE_A  3               0
4             0.465206     0           1      TISSUE_B  4               1
... 55 more rows
```

`outputs/sample/pred/` に4つ出ます。
- `predictions.tsv`: 予測結果の表（上の全60行）
- `predictions_by_group.tsv` がグループ別集計
- `predictions.npz` が同じ内容の配列
- `summary.json` が件数・閾値・確率の分布・指標・チェックポイントの学習元です。

```text
group     samples  predicted_positive  predicted_positive_ratio  label_positive  accuracy
TISSUE_A  20       13                  0.6500                    10              0.6500
TISSUE_B  20       2                   0.1000                    9               0.6500
TISSUE_C  20       10                  0.5000                    11              0.5500
```

`row_<alias>` 列があるので、各予測がどの発現行・どの組織シグネチャ行から出たかを追えます。

`pathwaygnn pred` の本来の使い方は「別に用意した外部データ」を指定してテストすることです。
新しいサンプルを同じグラフの上で汎用形式に前処理し、`dataset.dir` をそのディレクトリに向ければ、設定の他の部分は変わりません。
このリポジトリには外部データがないので、同梱の設定は `data_sample/prepared` 自身を指しています。
そのため上の指標はモデルの評価ではなく動作確認です。
このチェックポイントの実際の評価値は §4.5 の test 側で、同じ行に `pred` を当てるとその値を完全に再現します（テストで固定しています）。

`checkpoint:` には §4.5 の `finetune` の `best.pt` でも §4.3 の `cv` の fold の `model.pt` でも渡せます。
キーの一覧は [README_config.md](README_config.md) §8 にあります。

### 4.9 結果を表にする / 片付ける

```bash
python scripts/sample/summarize.py         # 上記の表をまとめて表示（再学習しない）
rm -rf data_sample/prepared outputs/sample # 生成物を全消去（raw は残る）
```

実データでは、この `summarize.py` の役割を専用のレポートコマンド
（`pathwaygnn-data tr-report` など）が担い、Markdown と HTML と図まで生成します（§7.1）。

### 4.10 自分のデータを足す

`src/pathwaygnn_datasets/sample/prepare.py` をコピーして4手順を書き換えるだけです。

```python
writer = DatasetWriter(output_dir, "mydata")
writer.write_graph(edge_index, edge_type, node_names, relation_names)   # 1. グラフ
writer.dense_node_feature("expression", rows, genes)   # 2. サンプル×遺伝子の表
writer.sparse_node_feature("mutation", ptr, gene, value)  #    （dense / sparse どちらでも）
writer.write_task("label", labels, node_features={...}, groups=..., sample_features=...)  # 3. タスク
writer.finish()                                                        # 4. 検証して書き出し
```

あとは `configs/sample/` を `configs/mydata/` にコピーして `dataset.yaml` の
`name` / `dir` を書き換えれば、`pretrain` / `cv` / `finetune` / `benchmark` / `ig` がそのまま動きます（エンジン側の変更は不要）。
CLI に自分の前処理を登録する場合は `src/pathwaygnn_datasets/cli.py` に分岐を足す必要があります。

---

## 5. モデルと学習ループの設計

1. PathwayCommons の関係ごとに GINConv を適用し、関係別出力を加算する（`models/encoder.py`）。
2. DistMult スコアと負例ノード置換によりエッジ予測を事前学習する。
3. 各 node-level feature を遺伝子軸で集約し、sample-level feature の分岐と連結して
   サンプルごとに二値分類する（`models/predictor.py`）。
4. ROC-AUC、accuracy、precision、recall、F1 を保存する。

`RelationalGIN` は **1関係 × 1層ごとに GINConv と線形射影**を持ち、関係方向に和をとって
ELU + dropout、全層を連結して線形読み出しに渡します。ノード特徴は学習される
`nn.Embedding`（外部ベクトルを併用する設定は §5.1）なので、forward は常に**全グラフ**です
（近傍サンプリングをしません）。
Integrated Gradients が埋め込み行列をスケールできるように `forward_from_embedding` があり、2経路の等価性はテストで保証しています。

サンプルレベルヘッド `SampleLevelModel` は1つで全データセットを表現します。
node-level feature ごとに「値をスカラー射影 → （グラフ使用時）ノード埋め込みを加算 →
gene block → 遺伝子軸で総和 → aggregate block」を通し、全 node-level feature と
sample-level feature 分岐を連結して1ロジットにします。

- `block: paper` = Linear-ELU-[BN]-Linear-ELU-[BN]（がん生存の論文アーキテクチャ）
- `block: plain` = Linear-ELU-[Dropout]-Linear（target repositioning）
- dense は `reshape().sum()`、sparse は `index_add_` で集約します（数学的に等価。
  ただし加算順序が変わるため、再現性のために互いに置き換えません）。

学習ループは素の PyTorch です（PyTorch Lightning を使いません）。
全グラフ forward、rank別エッジサンプリング、分散集約、rank 0 のみのチェックポイント保存を明示的に制御する方が、この用途では障害解析と再現性に優れるためです。

- `training/pretrain.py` — 唯一の分散ループ。DDP の各 rank は同じグラフ上で異なる
  正例・負例エッジを引き、勾配を all-reduce します（グローバルバッチ = `WORLD_SIZE × batch_size`、
  seed = `seed + rank`）。`training.partition:` を書くと、全グラフ forward の代わりに
  METIS 分割 のバッチが張る部分グラフで1ステップを回すモードに切り替わり、
  1ステップのメモリがグラフのサイズから切り離されます（§6.1）。
- `training/cv.py` — 汎用グリッド、fold単位の再開、グループ別AUC。単一プロセスです
  （`cancer` のグリッドは DDP ではなく、GPU ごとに `cv` を1プロセス起動して分配します）。
- `training/ig.py` — 埋め込み行列・各 node-level feature の値・sample-level feature への帰属。
- `training/finetune.py` — 単一分割、validation AUC による早期終了、`pos_weight` を使う唯一のループ。
- `training/benchmark.py` — 同じ特徴量での sklearn / XGBoost ベースライン。

encoder を凍結する設定（`training.train_encoder: false`、または `end_to_end: false` の variant）では
埋め込みを1回だけ計算して再利用します。これが両ループの主な高速化レバーです。

### 5.1 外部ノード埋め込み（`embedding_pc/` の配列・構造ベクトルを encoder に入れる）

既定では encoder の入力は学習される `nn.Embedding` だけで、ノードは「グラフ上の位置」しか
持ちません。[`embedding_pc/`](embedding_pc/README.md) は PathwayCommons の participant に対して
**タンパク質は UniProt 配列を ESMC-6B で（2,560次元）、化合物は ChEBI の SMILES を Uni-Mol で
（512次元）** 埋め込んだ表 `processed/node_embeddings.npz`（同内容の `node_embeddings.json` も可）を作ります。
これを `model.node_embeddings:` に書くと、その表が名前を持つノードだけ、
**種別ごとのアダプタ（`nn.Linear(D → hidden_dim)`）を通した外部ベクトル**が encoder の入力に入ります。

```text
node_embeddings.npz       nodes.json と名前で突き合わせ         encoder への入力 [N, hidden_dim]
  protein 19,360 x 2560 ──┐                        ┌── Linear(2560 → H) ──┐
  chemical 10,236 x 512 ──┴─ 該当ノードだけ抽出 ───┴── Linear( 512 → H) ──┴─▶ 学習される
   + 別名 (HGNC ID など)                                                    nn.Embedding の行に加算
  表に載っていないノード ───────────────────────────────────────────────▶ nn.Embedding の行のまま
```

- **名前で突き合わせます。** 表のキーは SIF の participant 表記（タンパク質 = gene symbol、
  化合物 = `CHEBI:xxxx`）で、`prepared/nodes.json` と一致した行だけが使われます。
  `data_tr` は 30,895 ノード中 **28,735（93.0%、protein 18,501 + chemical 10,234）** が一致します。
- **ノード名が gene symbol でないコーパスには「別名」を使います。**
  `embedding_pc/scripts/build_gene_ids.py`（HGNC complete set から作成）があると、
  `build_node_embeddings.py` は **HGNC ID / Entrez ID / Ensembl ID → 行番号** の対応を
  ベクトルを複製せずに表へ入れます。`data_cancer` はノード名が数値 HGNC ID なので、
  `aliases: [hgnc_id]` を足すと被覆が **10,646（34.4%）→ 29,566（95.6%、うち 18,920 が別名経由）**
  になります（`configs/cancer/pretrain_pc_embedding.yaml`）。
  **どの体系かは自動判定しません**: HGNC ID も Entrez ID も裸の数値で、`5` は HGNC:5 = A1BG とも
  Entrez 5 = 別遺伝子とも読めるためです。指定しなければ別名は一切使いません。
  `data_cdr` も数値 HGNC ID で、`aliases: [hgnc_id]` にすると 13,606 中 **13,433（98.7%）** が当たります
  （どちらも未指定だと1つも当たらずエラーになります）。
- **一致しなかったノードは今までどおり `nn.Embedding`** です。両者は排他で、
  同じノードが2つの種別に載っていれば読み込み時にエラーにします。
- **アダプタだけが学習し、外部ベクトルは凍結**します。ベクトルは `persistent=False` の buffer なので
  チェックポイントには入らず（+2 MB、`hidden_dim: 64` でアダプタは 196,736 パラメータ）、
  代わりに `model_config.node_embeddings` に**読み込み元のパスと各種設定が記録**されます。
  `cv` / `finetune` / `ig` / `pred` はその記録を見て自分で表を読み直すので、
  **事前学習以外のコマンドの設定は書き換え不要**です（表を移動したときだけ
  `model.node_embeddings.path` で上書きします）。
- **`training.partition:` とも併用できます。** 部分グラフのステップでも同じ行が使われます。
- Integrated Gradients の帰属先も「実際に畳み込まれた行列」になります
  （外部ベクトル由来の行を含む `[N, hidden_dim]`）。

```bash
# configs/tr/pretrain_pc_embedding.yaml = configs/tr/pretrain.yaml に下の設定を足しただけ
pathwaygnn pretrain --config configs/tr/pretrain_pc_embedding.yaml
#   -> outputs/tr/pretrain_pc_embedding/best.pt

# 以降は pretrained_checkpoint をその best.pt に向けるだけでよく、
# それらの設定に model.node_embeddings を書く必要はない（checkpoint に記録済み）
pathwaygnn cv --config configs/tr/cv.yaml
```

```yaml
model:
  node_embeddings:
    path: embedding_pc/processed/node_embeddings.npz
    normalize: l2       # l2（既定） / standardize / none
    combine: add        # add（既定） / replace
    aliases: []         # 既定は空。data_cancer なら [hgnc_id]
```

`normalize` は**何を残すか**、`combine` は**学習される行との関係**を決めます（詳細と既定値は
[README_config.md §3](README_config.md)）。アダプタの初期化は種別ごとの入力 RMS で割ってあるので、
`normalize` を変えても初期スケールは変わりません（`init_std`、既定は `add` で 0.01 = 学習行の1/10、
`replace` で 0.1 = 学習行と同じ）。

**測定値（`data_tr`、RTX PRO 6000 1枚、100 epoch、それ以外は `configs/tr/pretrain.yaml` と同一。
CV は `configs/tr/cv.yaml` の `oe_act` / `gnn_mlp` だけを、両方の checkpoint に対して
同じ日に走らせた対の実測値です。[docs/tr_report.md](docs/tr_report.md) の 0.713 とは
別の run なので、比べるのは同じ表の中だけにしてください）**

| 事前学習 | epoch 1 loss | epoch 100 loss | epoch 100 accuracy | `oe_act` の 5-fold CV AUC（`gnn_mlp`） |
| --- | --- | --- | --- | --- |
| 外部埋め込みなし（既定） | 12.91 | **0.967** | **0.867** | **0.735 ± 0.039** |
| `l2` + `add`（既定） | 45.85 | 1.273 | 0.785 | 0.723 ± 0.036 |
| `l2` + `replace` | 4597 | 2.059 | 0.735 | — |
| `standardize` + `add` | 19.32 | 8.487 | 0.657 | — |

`data_cancer`（`aliases: [hgnc_id]`、`configs/cancer/pretrain.yaml` の 50 epoch）でも同じ傾向で、
best epoch の loss / accuracy は **1.477 / 0.704 → 1.524 / 0.648**（外部埋め込みなし → あり）です。

読み方: **エッジ予測の当てはまりは外部ベクトルを入れると必ず悪くなります**。
93% のノードの表現が凍結ベクトルの線形像に縛られる以上これは当然で、
`combine: replace`（学習行を完全に置き換える）や `normalize: standardize`
（ESMC ベクトルのほぼ定数の次元まで等分散に引き伸ばす）ほど悪化が大きくなります。
下流の `oe_act` でも 0.723 ± 0.036 と、ベースライン 0.735 ± 0.039 に対して
**fold 間のばらつきの内側**で、この設定では改善は観測できていません。
既定値（`l2` + `add`）は「この中では最も素直に学習が進む組み合わせ」という意味で、
**外部埋め込みが効くという主張ではありません**。他のコーパス・他の埋め込みモデルで試すための
配線として用意してあります。

---

## 6. 分散事前学習

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

`best.pt` / `last.pt` は rank 0 のみが保存します。
NCCL は GPU、Gloo は CPU で自動選択されます。
教材データは20ノードしかないので分散の意味がありません（`configs/sample/pretrain.yaml` は`device: cpu`）。

### 6.1 グラフ分割による分散実行（大規模グラフ）

上の DDP は**各 rank がグラフ全体を持ちます**。
1ステップの各rankの必要メモリは　`ノード数 × hidden_dim × 関係数` で増えるので、rank を増やしてもグラフが大きくなればどこかで載らなくなります。
実測では `data_cdr`（13,606ノード・**356関係**）の全グラフforward+backward が **14.1 GiB** です。

グラフ分割モードは Cluster-GCN の考え方でこの上限を外します。
グラフを METIS で`num_parts` 個に切っておき、1ステップは**そのうち `parts_per_batch` 個が張る部分グラフだけ**を
計算します。メモリはグラフのサイズではなくバッチで決まり、rank には**互いに素なパーティション**が配られます（`DistributedSampler`）。

```bash
# ① グラフを1回だけ切る（グラフ全体をメモリに載せる唯一の工程）
pathwaygnn partition --config configs/tr/pretrain_partitioned.yaml

# ② 分割ファイルだけを読んで分散学習する（①②をまとめたのが下のスクリプト）
NPROC_PER_NODE=4 bash scripts/tr/pretrain_partitioned.sh
```

実測（`data_cdr`、`num_parts: 64`、`hidden_dim: 64`、RTX PRO 6000。
ピークメモリはパラメータ・勾配・optimizer 状態が載った状態での値、つまり「載るかどうか」を決める値）:

| モード | 1ステップのノード数 | ピークメモリ | 1ステップ | 1周で見えるエッジ | 1周の時間 |
| --- | --- | --- | --- | --- | --- |
| 全グラフ | 13,606 | 14.11 GiB | 509 ms | 100% | 0.51 s |
| 分割 `parts_per_batch: 1` | 213 | **0.40 GiB** | 470 ms | 55.6% | 30.2 s |
| 分割 `parts_per_batch: 4` | 850 | 1.06 GiB | 465 ms | 58.0% | 7.5 s |
| 分割 `parts_per_batch: 8` | 1,701 | 1.93 GiB | 468 ms | 59.6% | 3.8 s |
| 分割 `parts_per_batch: 16` | 3,402 | 3.67 GiB | 493 ms | 67.0% | 2.0 s |

**この表は「メモリを時間と忠実度で買う」ことを示しています**
1ステップの時間はほとんど減りません（関係ごとに GINConv を1つ走らせるので、部分グラフを小さくしても356回のカーネル起動は消えない）。
したがって**1周あたりの時間は分割数とともに増えます**。
分割は「載らないグラフを載せる」ための手段で、載るグラフを速くするものではありません。

参考に `data_tr`（30,895ノード、**13関係**）では全グラフでも 1.47 GiB / 30 ms しかかからないので、この設定では**分割は不要**です。
効くのは関係数の多い `cdr` 型、あるいはこれより大きなグラフ・大きな `hidden_dim` です。
分割数 × バッチ幅の全組み合わせ（68条件）の実測は [docs/dist_report.md](docs/dist_report.md) にあります（`pathwaygnn dist-benchmark` の生成物）。

```bash
pathwaygnn dist-benchmark   --config configs/dist/benchmark.yaml   # 計測
pathwaygnn-data dist-report --config configs/dist/report.yaml      # 文書化
```

教材データでも仕組みだけは10秒で通せます（20ノードを4分割するので分割自体に意味はありません）。

```bash
pathwaygnn partition --config configs/sample/pretrain_partitioned.yaml
pathwaygnn pretrain  --config configs/sample/pretrain_partitioned.yaml
```

分割は `pretrain` から切り離された別コマンドです。
グラフ全体をメモリに載せるのがそこだけなので、載る計算機で1回作り、学習側は分割ファイルしか読みません
（`training.partition.create: false` にすると、グラフを読むフォールバックを禁止できます）。

#### 分割の注意事項: 
現状の実装では、分割で切られたエッジはそのステップに現れず、負例の破壊先も部分グラフ内のノードに限られるので、**全グラフループと同じ数値にはなりません**。
PathwayCommons 由来のグラフはハブが多く密で、`data_tr` を256分割すると1周で見えるエッジは 13.4%、`parts_per_batch: 8` でも16.0%です（`shuffle: true` で毎エポック組み合わせが変わるため、切られたエッジも学習の過程では登場します）。
キーごとの実測値は
[README_config.md](README_config.md) §3 の `training.partition:` にあります。
大規模グラフを扱うための手段であり、既存の再現結果（cancer の公開数値など）は全グラフループのものです。
`training.partition` を書かなければ従来どおりで、損失は一致します。

なお分割が効くのは事前学習（グラフ側のループ）です。`cv` / `finetune` /　`ig` のサンプルレベルヘッドは全遺伝子の埋め込みを同時に必要とするため、分割の対象外です。
分割学習で作った `best.pt` は、チェックポイントがグラフ全体のノード数を持つのでこれらのコマンドからそのまま使えます。

---

## 6.2 HuggingFace Hub への公開と再開

事前学習後・finetune後のチェックポイントを公開し、公開されたモデルから再開できます。

```bash
# 公開する（既定は dry_run: true = マシンの外に何も出さず内容を組み立てるだけ）
pathwaygnn hg --config configs/tr/hub.yaml
```

`checkpoint:` の種別は自動判別されます。`pretrain` の出力なら **encoder**、
`finetune` / `cv` の出力なら **head** として、`model.pt`（バイト一致のコピー）、
`pathwaygnn.json`（機械可読なマニフェスト）、`README.md`（モデルカード）を組み立てます。
モデルカードの Usage 節には**そのまま貼れる設定行**が入ります。

モデル利用側に新しいコマンドは不要で、チェックポイントを指すキーはすべて`hf://` 参照を受け付けて自動で利用可能です。

```yaml
# configs/tr/cv.yaml など（`pretrained_checkpoint` を持つ全コマンド）
pretrained_checkpoint: hf://your-account/pathwaygnn-tr-encoder/model.pt@v1

# configs/tr/pred_oe_act.yaml（head を公開した場合）
checkpoint: hf://your-account/pathwaygnn-tr-oe-act/model.pt

# configs/tr/pretrain.yaml — 事前学習を続ける
resume_from: hf://your-account/pathwaygnn-tr-encoder/model.pt
```

`resume_from` は**本当の再開**です。
重み・optimizer のモーメント・エポック番号に加えて**エッジサンプラの乱数ストリームまで復元**するので、**中断せずに回した場合とビット一致します**
（`training.epochs` は「ここから追加で回すエポック数」）。
公開モデルの形状が手元の `model:` ブロックより優先されるため、静かに作り替えられることもありません。

既定は `private: true`、`dry_run: true` です（公開は明示的な選択）。
認証は`hf auth login` か `HF_TOKEN` を推奨します（設定ファイルにトークンを書かないでください）。
キーの一覧は [README_config.md](README_config.md) §9 にあります。

---

## 7. 応用: 3つの実データセット

チュートリアルと**同じコマンド・同じ汎用形式**です。違うのは、①生データが巨大・雑多なので
`raw → processed → prepared` の**2段**になること、②論文の参照値と比較するレポートがあること、
③データセットごとの流儀（クセ）が設定に明記されていることだけです。

### 7.1 共通: 教材データとの差分

| | `sample`（教材） | 実データ |
| --- | --- | --- |
| 前処理 | 1段（`sample-prepare`） | 2段（`*-build-processed` → `*-prepare`） |
| 生データ | コミット済み 12 KB | Git 管理外（1.6〜21.5 GB、取得スクリプトで再生成） |
| レポート | `scripts/sample/summarize.py`（標準出力のみ） | `pathwaygnn-data {tr,cancer,cdr}-report` → `docs/*.md` + `*.html` + 図 |
| 計算資源 | CPU 2分 | GPU 前提（`cancer` の Table 1 全20条件で3 GPU × 約50分） |
| 実行 | `bash scripts/sample/run_all.sh` | `bash scripts/cancer/reproduce_paper.sh full` など |

`docs/` 配下（`docs/papers/` を除く）は**すべて生成物**です。Markdown と HTML は同一の
`pathwaygnn_datasets/document.py` から作られるので食い違いません。編集するのは
レポートモジュールの側です。

### 7.2 `data_tr` — 創薬標的の再配置（SLGCN-TR）

「ある遺伝子に摂動（KD/OE）を与えたときの発現変化」と「ある疾患の発現シグネチャ」を
突き合わせ、その遺伝子がその疾患の阻害性／活性化性標的かを分類します。
実体は LINCS L1000 Level 5 + CREEDS + PathwayCommons。詳細は
[README_data_tr.md](README_data_tr.md) / [data_tr/README.md](data_tr/README.md)。

```bash
python -m scripts.tr.upstream.download_raw_data   # ① 公開ソース取得（21.5 GB）
bash scripts/tr/build_processed.sh                # ② 要 .[tr-upstream]（h5py）
bash scripts/tr/prepare.sh                        # ③ -> data_tr/prepared
pathwaygnn pretrain  --config configs/tr/pretrain.yaml
pathwaygnn cv        --config configs/tr/cv.yaml            # kd_inh / oe_act × グラフ有無
pathwaygnn finetune  --config configs/tr/finetune_kd_inh.yaml
pathwaygnn benchmark --config configs/tr/benchmark_kd_inh.yaml
pathwaygnn ig        --config configs/tr/ig_kd_inh.yaml
pathwaygnn-data tr-report --config configs/tr/report.yaml   # -> docs/tr_report.md
```

教材データとの対応: node-level feature は `disease`（**2タスクで共有**）と
`perturbation_kd` / `perturbation_oe`（タスクごと）で、alias は両タスクとも
`perturbation` / `disease`。groups は対象疾患177種。**sample-level feature が無い**ため
`cv` のアブレーションは2条件（`mlp` / `gnn_mlp`）です。ラベルは陽性8.2%と不均衡なので
`pos_weight: auto` を使います（教材データは均衡なので不要でした）。

### 7.3 `data_cancer` — TCGA がん予後

TCGA 発現プロファイルから診断後 n 年（1〜5年）の生存を二値予測します。
論文の Table 1（4モデル × 5年 = 20条件）、がん種別AUC、Integrated Gradients を再現します。
詳細は [README_data_cancer.md](README_data_cancer.md) / [data_cancer/README.md](data_cancer/README.md)。

```bash
bash scripts/cancer/reproduce_paper.sh full          # 前処理〜3GPU事前学習〜Table1〜IG〜レポート
python scripts/cancer/reproduce_table1.py --gpus 0,1,2   # Table 1のみ（fold単位で再開可能）
```

教材データとの対応: node-level feature は `expression`（dense、4,448遺伝子）1つ、
task は `1year`…`5year` で**同じ alias を公開**するため1つのモデル設定で回ります
（§3 の設計がそのまま効く例）。sample-level feature はがん種 one-hot、
groups はがん種。**論文の数値がコード内に定数として入っており**
（`pathwaygnn_datasets/cancer/paper.py`）、`cancer-prepare` が年ごとのサンプル数を検証し、
`cancer-report` が 論文値 / 再現値 / 差分の表を出します。

このデータセットには再現性のために次の設定が指定されています（`selection: final_epoch`、`loss_clip: 0.01`、`loss_reduction: sum`、`grad_clip_value: 10.0`、`shuffle: false`、`num_layers: 2`）。
汎用の既定値は素直な側（`mean`、clipなし、`final_epoch`）で、がんの設定が明示的に上書きしています（変更すると再現性が失われる可能性があります）。

### 7.4 `data_cdr` — GDSC 薬剤感受性（GraphCDRScan）

サンプルが *(細胞株, 化合物)* の組である例です。詳細は
[README_data_cdr.md](README_data_cdr.md) / [data_cdr/README.md](data_cdr/README.md)。

```bash
python -m scripts.cdr.upstream.download_raw_data          # ① 要 .[cdr-upstream]
python -m scripts.cdr.upstream.prepare_data --config configs/cdr/upstream.json   # ②
bash scripts/cdr/prepare.sh                               # ③ numpy のみで完結
bash scripts/cdr/reproduce.sh                             # pretrain / cv / finetune / benchmark / ig / report
```

教材データとの対応が最も分かりやすい例です。

- node-level feature `mutation`（sparse）は細胞株にしか依存しないため、
  107,418サンプルが760行を共有します（`rows/mutation.npy`)。教材データで60サンプルが3行の生体組織シグネチャを共有していたのと同じ仕組みです。
- sample-level feature は3,348次元（変異スペクトル96/78/83 + 原発部位one-hot + 化合物
  フィンガープリント3×1024）。教材データの4次元（年齢・性別・病期・喫煙）の実物版です。
- groups は原発部位19種 → グループ別AUCがそのまま「がん種別AUC」。
- task は `LN_IC50` の二値化2種。`sensitive_drugwise`（同一化合物の中央値で分割＝細胞株の
  ゲノムだけが手がかり）と `sensitive_global`（全体中央値＝化合物の効力が支配的）。
- **関係が356種**あるため encoder は 9.8 M パラメータで、1 forward+backward が
  RTX PRO 6000 で約0.5秒かかります。`configs/cdr/cv.yaml` は `end_to_end: false`
  （fold ごとに埋め込みを1回計算）を維持してください。

---

## 8. 設定と再現性

YAML の `defaults` は**その設定ファイルからの相対パス**で、辞書は再帰的にマージ、
スカラーとリストは上書きです。CLI からの上書きやスキーマはありません
（各値は使用箇所で `.get(key, default)` されるので、**実効的な既定値は学習モジュール側**にあります）。
**全項目の説明は [README_config.md](README_config.md)** にあります。主要なものは以下です。

- `dataset.name`, `dataset.dir`: 使用するデータセット（`dataset.json` と照合）
- `dataset.task` / `dataset.tasks`: 対象タスク
- `seed`: Python / PyTorch と分割の乱数 seed
- `model.hidden_dim`, `model.num_layers`: グラフ encoder
- `model.block`, `model.batch_norm`, `model.dropout`: サンプルレベルヘッドの構造
- `training.batch_size`: 事前学習では rank ごとのエッジ数、それ以外はサンプル数
- `training.train_encoder` / `training.end_to_end`: グラフ encoder も更新するか
- `training.selection`: `final_epoch`（論文プロトコル）か `best_test_auc`（公開コード互換、
  held-out で選ぶのでリークあり）
- `variants[].use_graph`, `variants[].use_sample_features`: `cv` のアブレーション条件

fold の seed は **位置に依存しません**。

```text
seed + task.seed_offset * 1000 + fold + variant.seed_index * 100
```

`seed_offset` はタスク（がんでは検証年）が、`seed_index` は variant が持つため、
1条件だけを単独実行してもグリッド全体と同じ seed になります
（`scripts/cancer/reproduce_table1.py` がグリッドを条件別設定に分割できる理由）。

**GPU の非決定性は仕様です。** `end_to_end: true` では encoder の scatter-add が毎ステップ走るため、
同一設定の2回の実行は150エポックで数 1e-3 程度ずれます。リファクタの正しさは
**1エポック目の学習損失がビット一致するか**で判断してください（最終 AUC で判断しない）。

---

## 9. テスト

```bash
conda run -n gnn python -m pytest          # 131件、数秒、CPU のみ
```

小さな raw データと合成データセットを `tmp_path` に作り、前処理、汎用形式の読み書き、
関係別 GIN、エッジ事前学習、dense/sparse node-level feature の等価性、可変長サンプル集約、
逆伝播、層化分割、`cv` グリッドと fold 再開、`ig` の出力を検証します。
- `tests/test_tensor_shapes.py` は、モデル周辺の関数が受け渡すテンソルの次元と dtype が
docstring/コメントの記述どおりであることを検証します（記述が陳腐化しないようにするため）。
- `tests/test_predict.py` は `pred` の表の列と行数、閾値の効き方、**`finetune` が報告した
test 指標を同じ行に対して完全再現すること**、別データセット（同じグラフ）を採点できること、
ラベルが1クラスなら指標を出さないことを検証します。
- `tests/test_hub.py` は `hf://` 参照の解析、staging された payload の自己記述性、
dry run が何も送信しないこと、`resume_from` が中断なしの学習とビット一致すること、
公開モデルの形状が手元の設定より優先されることを検証します
（`huggingface_hub` は入れず、`sys.modules` のスタブで代替します）。
- `tests/test_partition.py` は METIS 分割が全ノード・全エッジをちょうど1回覆うこと、
パーティションのバッチがその集合の誘導部分グラフと厳密に一致すること（分割をまたぐ
エッジを落とさないこと）、rank への割り当てが互いに素かつ同ステップ数になること
（DDP の all-reduce が要求します）、`graph.pt` を消しても分割だけで学習できること、
そして `training.partition` を書かなければ損失がビット一致することを検証します。
- `data_tr/` `data_cancer/` `data_cdr/` `outputs/` には触れません。
- `tests/test_sample_prepare.py` だけは例外的にコミット済みのデータセット `data_sample/raw` を読み、`scripts/sample/make_raw_data.py` がそれをバイト単位で再生成できることも確認します
（書き込みは `tmp_path` のみ）。

---

## 10. リポジトリの構成

```text
src/pathwaygnn/            学習エンジン（データセット非依存）
  data/format.py             汎用データ形式の定義と DatasetWriter
  data/samples.py            タスク → バッチ（dense/sparse の可変長集約）
  data/partition.py          METIS グラフ分割と、その上の Cluster-GCN ローダ
  data/node_embeddings.py    外部ノード埋め込み表の読み込みと nodes.json との突き合わせ
  hub.py                     HuggingFace Hub への公開と hf:// 参照の解決
  models/encoder.py          RelationalGIN、GraphPretrainer、load_encoder
  models/predictor.py        SampleLevelModel（両データセット共通のヘッド）
  training/                  pretrain / cv / finetune / benchmark / ig / pred / metrics / distributed
                             dist_benchmark（分割数 × バッチ幅の時間・メモリ計測）
src/pathwaygnn_datasets/   コーパスごとの前処理とレポート
  sample/prepare.py          最小の実装例（約240行）
  tr/ cancer/ cdr/           各コーパスの build / prepare / report
  dist/report.py             グラフ分割ベンチマークの文書化（データセット非依存）
configs/{sample,tr,cancer,cdr}/   実験設定（dataset.yaml を defaults で取り込む）
scripts/{sample,tr,cancer,cdr}/   取得・再現スクリプト
embedding_pc/              PathwayCommons ノードの外部埋め込み（配列・構造）の作成スクリプト
data_sample/               教材データ（raw 12 KB）
data_{tr,cancer,cdr}/      実データ（README.md 以外は Git 管理外）
docs/                      生成されるレポート（docs/papers/ 以外はすべて生成物）
                           <名前>.md と <名前>.html は同一ソースから生成。html は
                           スタイル内蔵・印刷対応で、同階層の <名前>_assets/ だけを
                           伴えばそのまま配布できます
outputs/                   学習結果（Git 管理外）
tests/                     pytest（合成データのみで完結）
```

`prepared` データ、学習結果、チェックポイントは `.gitignore` 対象です。
`raw` と `processed` も容量のため対象外です（`data_tr` 21.5 GB、`data_cdr` 1.6 GB / 9.3 GB。COSMIC のように登録が必要な入力もあります）。
いずれも取得スクリプトと build コマンドで再生成します。
 `data_sample/raw/`** は、例外的にこれはクローン直後にチュートリアルを動かすためにリポジトリに含まれています。

ライセンスは [LICENSE](LICENSE)（MIT）です。各コーパスの元データはそれぞれの
配布条件に従ってください（`data_*/README.md` に出典を記載しています）。
