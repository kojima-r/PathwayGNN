# README_data_tr — target repositioning（`data_tr`）

創薬標的の再配置（target repositioning）データセットです。「ある遺伝子に摂動を与えた際の
発現変化」と「ある疾患の発現シグネチャ」を突き合わせ、その遺伝子がその疾患の
**阻害性標的（inhibitory target）** あるいは **活性化性標的（activatory target）** かを
二値分類します。由来は `SLGCN-TR`。

- 設定項目の説明: [README_config.md](README_config.md)
- 実行結果: [`docs/tr_report.md`](docs/tr_report.md) / [`docs/tr_report.html`](docs/tr_report.html)

---

## 0. 最初に読む: リポジトリをクローンした直後の状態

**3つのデータセットのうち、生データがリポジトリに同梱されているのは `data_tr` だけです**
（合計 150 MB、`SHA256SUMS` つき）。したがってクローン直後にそのまま実行できます。
外部からのダウンロードも、ライセンス手続きも、追加の依存関係も必要ありません。

```bash
conda activate gnn
pip install -e .
bash scripts/tr/prepare.sh                              # data_tr/raw -> data_tr/prepared
pathwaygnn pretrain --config configs/tr/pretrain.yaml   # 以降は §4 参照
```

---

## 1. ディレクトリ構成

```text
data_tr/
├── raw/                      Git 管理対象（SHA256SUMS つき）
│   ├── PathwayCommons12.All.hgnc.sif
│   ├── disease_specific_signature.tsv
│   ├── knockdown_signature_sample.tsv
│   ├── overexpression_signature_sample.tsv
│   ├── inhibitory_target_disease.tsv
│   ├── activatory_target_disease.tsv
│   ├── Target_repositioning_データ完全版説明.txt
│   ├── SHA256SUMS
│   └── 推論用データ/                 未使用（本パイプラインは読まない）
│       ├── pred_data_inhibit.tsv
│       └── pred_data_activate.tsv
└── prepared/                 生成物（.gitignore 対象）
```

`raw/` は3つのデータセットの中で唯一 Git 管理下にあります（合計 150 MB 程度）。
`SHA256SUMS` で各ファイルのハッシュを固定しているので、前処理の再現性が検証できます。

---

## 2. 元データ

### 2.1 `PathwayCommons12.All.hgnc.sif` — パスウェイグラフ

PathwayCommons v12 の SIF 形式。3列 TSV で **1,884,849 行**、ヘッダなし。

```text
A1BG	controls-expression-of	A2M
A1BG	interacts-with	ABCC6
```

| 列 | 内容 |
| --- | --- |
| 1 | 起点ノード（HGNC 承認シンボル、または `CHEBI:*` などの化学実体） |
| 2 | 関係タイプ（13種） |
| 3 | 終点ノード |

関係タイプ13種:
`catalysis-precedes`, `chemical-affects`, `consumption-controlled-by`,
`controls-expression-of`, `controls-phosphorylation-of`, `controls-production-of`,
`controls-state-change-of`, `controls-transport-of`, `controls-transport-of-chemical`,
`in-complex-with`, `interacts-with`, `reacts-with`, `used-to-produce`。

> 旧 README の `PathwayCommons12.All.hgnc.sif.tsv` という名前でも自動検出されます。

### 2.2 `disease_specific_signature.tsv` — 疾患特異的発現シグネチャ

ヘッダ付き3列 TSV、**251,041 行**。

```text
do_id	human_gene_name	expression
DOID:0050156	A2M	0.007530039642006159
```

| 列 | 内容 |
| --- | --- |
| `do_id` | Disease Ontology ID（235疾患） |
| `human_gene_name` | 遺伝子シンボル |
| `expression` | 疾患特異的な発現変化量（実数、正負あり） |

### 2.3 `knockdown_signature_sample.tsv` / `overexpression_signature_sample.tsv` — 摂動シグネチャ

L1000 の遺伝子発現プロファイル。**行 = 摂動を与えた遺伝子、列 = 測定した L1000 遺伝子（978個）**
のワイド形式で、先頭列が `pert_iname`（＝ 979 列）。

```text
pert_iname	PSME1	ATF1	RHEB	FOXO3	RHOA	...
61E3.4	0.42161217	-0.46110147	0.25424212	0.04707861	-0.07255195	...
```

| ファイル | 摂動 | 行数（ヘッダ除く） |
| --- | --- | --- |
| `knockdown_signature_sample.tsv` | ノックダウン | 4,345 |
| `overexpression_signature_sample.tsv` | 過剰発現 | 3,114 |

### 2.4 `inhibitory_target_disease.tsv` / `activatory_target_disease.tsv` — 正解ラベル

ヘッダ付き3列 TSV。`label` は 1 が「関係あり」、0 が「関係なし」。

```text
gene	doid	label
ABL1	DOID:8552	1
```

| ファイル | 対応タスク | 行数（ヘッダ除く） |
| --- | --- | --- |
| `inhibitory_target_disease.tsv` | `kd_inh`（ノックダウン × 阻害性標的） | 7,168 |
| `activatory_target_disease.tsv` | `oe_act`（過剰発現 × 活性化性標的） | 450 |

### 2.5 `推論用データ/`

学習・評価には使いません。本パイプラインは読み込まないため、
`prepare_tr_dataset` の対象外です。

---

## 3. 前処理（`pathwaygnn-data tr-prepare`）

```bash
bash scripts/tr/prepare.sh
# = pathwaygnn-data tr-prepare --config configs/tr/prepare.yaml
```

実装は `src/pathwaygnn_datasets/tr/prepare.py`。処理内容は次の通りです。

1. **グラフ構築** — SIF の各エッジを **両方向に追加して対称化**し、`(src, relation, dst)` の
   集合で重複を除去、`sorted()` してから整数 ID を振ります。ノード名・関係名も
   `sorted()` するため、`PYTHONHASHSEED` に依存せず決定的です。
2. **疾患シグネチャ** — グラフに存在しない遺伝子の行は捨て（`disease_rows_skipped` に記録）、
   絶対値が `cutoff`（既定 `1e-7`）未満の値も落として CSR 疎行列にします。
3. **摂動シグネチャ** — ヘッダの L1000 列のうちグラフに存在するものだけを使い
   （`signature_genes_skipped` に記録）、同じく `cutoff` で疎化します。
4. **ラベル** — `gene` が摂動テーブルに、`doid` が疾患テーブルに存在しない行は捨てます
   （`label_rows_skipped` に記録）。
5. すべての件数と除外行数を `dataset.json` / `task.json` に記録します。

### 生成される汎用形式

| 項目 | 値 |
| --- | --- |
| ノード数 | 30,918 |
| 関係タイプ数 | 13 |
| 有向エッジ数 | 3,673,654（対称化・重複除去後） |
| 疾患数 | 235 |
| 除外された疾患シグネチャ行 | 11,053 |

**channel（データセット単位の遺伝子-値テーブル、すべて sparse）**

| channel | 行数 | 非ゼロ値数 | 内容 |
| --- | --- | --- | --- |
| `disease` | 235 | 239,173 | 疾患シグネチャ。**2タスクで共有** |
| `perturbation_kd` | 4,345 | 4,166,696 | ノックダウンシグネチャ |
| `perturbation_oe` | 3,114 | 2,985,387 | 過剰発現シグネチャ |

**task**

| task | サンプル数 | 陽性数 | alias → channel | 除外ラベル行 |
| --- | --- | --- | --- | --- |
| `kd_inh` | 6,944 | 567 | `perturbation`→`perturbation_kd`, `disease`→`disease` | 224 |
| `oe_act` | 450 | 37 | `perturbation`→`perturbation_oe`, `disease`→`disease` | 0 |

両タスクとも **alias は `perturbation` と `disease` で共通**なので、モデル設定がそのまま
流用できます（これが汎用形式の alias 機構の意図です）。`groups` はサンプルが対象とする
疾患（235種）で、疾患別 AUC と疾患別帰属の集計に使われます。共変量はありません
（したがって `use_covariates: true` の variant は使えません）。

---

## 4. 実行

```bash
conda activate gnn
bash scripts/tr/prepare.sh
pathwaygnn pretrain  --config configs/tr/pretrain.yaml      # 単一 GPU
bash scripts/tr/pretrain_distributed.sh                     # または torchrun で分散
pathwaygnn cv        --config configs/tr/cv.yaml
pathwaygnn finetune  --config configs/tr/finetune_kd_inh.yaml
pathwaygnn finetune  --config configs/tr/finetune_oe_act.yaml
pathwaygnn benchmark --config configs/tr/benchmark_kd_inh.yaml
pathwaygnn benchmark --config configs/tr/benchmark_oe_act.yaml
pathwaygnn ig        --config configs/tr/ig_kd_inh.yaml
pathwaygnn ig        --config configs/tr/ig_oe_act.yaml
pathwaygnn-data tr-report --config configs/tr/report.yaml
```

`cv` のアブレーションは共変量がないため2条件（`mlp` / `gnn_mlp`）です。

---

## 5. 注意点

- **`oe_act` は小さい**（450サンプル・陽性37件）。5-fold の分散が大きく、平均値の
  推定精度は高くありません。`finetune` の validation 分割は数十件しかないため、
  `best_valid_auc` は held-out 性能を過大評価します。
- **`kd_inh` では木系ベースラインの方が強い。** 同一 fold で比較して、グラフパイプラインが
  素の特徴量モデルに劣ります。詳細は `docs/tr_report.md` の "Interpretation scope" を参照。
- **IG のランキングは次数に強く相関します。** 上位は `CHEBI:*` の化学実体が占めがちで、
  疾患特異的な機序の証拠としては読めません。
- グラフノードは遺伝子シンボルだけでなく化学実体も含むため、ノード数（30,918）は
  ヒト遺伝子数より多くなります。
