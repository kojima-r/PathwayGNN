# README_data_cdr — がん薬剤感受性（`data_cdr`）

がん細胞株と化合物の組に対して薬剤感受性（GDSC の `LN_IC50`）を予測します。
由来は `GraphCDRScan`（CDRscan の Reactome グラフ版）。`data_cdr/` は GraphCDRScan の
`data/` をそのまま持ち込んだもので、前処理も `scripts/cdr/upstream/` に取り込んであるため
PathwayGNN 内で完結します。

- 設定項目の説明: [README_config.md](README_config.md)
- 実行・評価結果: [`docs/cdr_report.md`](docs/cdr_report.md) / [`docs/cdr_report.html`](docs/cdr_report.html)

---

## 1. ディレクトリ構成

```text
data_cdr/                     全体が .gitignore 対象（容量とライセンスのため）
├── raw/                      2.4 GB — 上流ステージの入力
│   ├── sources/              取得した公開ファイルの原本（監査用に保持）
│   ├── CosmicCLP_MutantExport.tsv     312 MB
│   ├── v17_fitted_dose_response.csv    41 MB
│   ├── Screened_Compounds.csv          42 KB
│   ├── used_cell_lines.csv            6.5 KB
│   ├── used_compounds.csv             5.3 KB
│   ├── cancer_gene_census.csv         5.3 KB
│   ├── reactome_rev2.graph.tsv        7.0 MB
│   ├── EnsemblToHGNC.tsv              1.5 MB
│   ├── hg38.2bit / hg19.2bit          835 MB / 816 MB
│   ├── fingerprints.csv               1.4 MB（生成物）
│   ├── maf.csv                        130 MB（生成物・キャッシュ）
│   └── SHA256SUMS
├── processed/full_features/  9.3 GB — 上流ステージの出力＝cdr-prepare の入力
│   ├── graph.tsv                      7.7 MB
│   ├── vertices_dic.tsv
│   ├── relationships_dic.tsv
│   ├── node_features.tsv              9.1 GB
│   ├── sample_features.tsv            729 MB
│   └── labels.tsv                     3.6 MB
└── prepared/                 2.7 GB — 生成物。pathwaygnn が読む汎用形式
```

`data_tr/raw` と違い Git 管理外です。容量（2.4 GB / 9.3 GB）に加えて、
COSMIC Cancer Gene Census が登録済みユーザーの手動ダウンロードを要求するためです。

---

## 2. パイプラインの2段階

```text
公開データ  ──①──▶  data_cdr/raw  ──②──▶  data_cdr/processed  ──③──▶  data_cdr/prepared
           download_raw_data      prepare_data              cdr-prepare
           （scripts/cdr/upstream/）                        （pathwaygnn_datasets/cdr/）
```

| 段階 | 実行コマンド | 依存関係 |
| --- | --- | --- |
| ① 取得・互換変換 | `python -m scripts.cdr.upstream.download_raw_data` | `.[cdr-upstream]` + `pdftotext`(Poppler) + LibreOffice |
| ② 特徴量生成 | `python -m scripts.cdr.upstream.prepare_data --config configs/cdr/upstream.json` | 同上 |
| ③ 汎用形式へ変換 | `bash scripts/cdr/prepare.sh` | **numpy のみ** |

**①②は既に完了済みなので通常は不要です。** これらの依存関係（pandas / rdkit /
twobitreader / signatureanalyzer）は `gnn` 環境には入っていません。③以降は numpy だけで
動くよう意図的に書かれており、パイプライン本体が上流の依存関係を持ち込むことはありません。

`scripts/cdr/upstream/` は GraphCDRScan の `scripts/` ツリーの写しです。
**変更点はデータ根のみ**（`data/` → `data_cdr/`、`prepare_data.py` の `--data-root`）で、
互換変換・エンコード・出力レイアウトは上流と同一です。「モダン化」しないでください
──上流のバンドルをそのまま再現するために存在します。

---

## 3. 元データ（`raw/`）

CDRscan 論文（2018）が使った COSMIC/CCLP v82 + GDSC 6.0 と、当時の Google Drive 上の
Reactome グラフはいずれも入手できません。`download_raw_data.py` は現行の公開ソースを
取得して、レガシーなファイル名・スキーマへ変換します。

| `raw/` のファイル | 現在の取得元と変換内容 |
| --- | --- |
| `CosmicCLP_MutantExport.tsv` | Cell Model Passports の `mutations_all_latest.csv.gz` + `model_list_latest.csv.gz`。Sanger モデル ID を COSMIC ID に逆写像し、列をレガシーリーダに合わせる |
| `Screened_Compounds.csv` | Cell Model Passports / GDSC Compound Annotation release 8.5 |
| `v17_fitted_dose_response.csv` | 現行の GDSC1 fitted-response ワークブック（`GDSC1_fitted_dose_response_27Oct23.xlsx`）を CSV 化し COSMIC ID に写像。ファイル名だけレガシーを踏襲 |
| `used_cell_lines.csv`, `used_compounds.csv` | CDRscan 論文の CC BY 補足 PDF の Table S2 / S3。旧表記の化合物名4件を現行 GDSC 名に正規化 |
| `reactome_rev2.graph.tsv` | Reactome Functional Interactions 2025。承認シンボルを数値 HGNC ID に写像し3列グラフ形式へ変換。消滅した Google Drive 成果物の代替 |
| `EnsemblToHGNC.tsv` | HGNC 公開バケットの現行 complete TSV |
| `hg38.2bit` | UCSC `goldenPath/hg38/bigZips/hg38.2bit` |
| `cancer_gene_census.csv` | COSMIC Cancer Gene Census v104（GRCh38）。**認証付き手動ダウンロードが必要** |

COSMIC の CGC は次のように手動で渡します。

```bash
python -m scripts.cdr.upstream.download_raw_data --no-download \
  --cosmic-cgc ./Cosmic_CancerGeneCensus_Tsv_v104_GRCh38.tar
```

CGC からは `GENE_SYMBOL` 列しか読まず、v104 の GRCh37 版と GRCh38 版は同一の 768 シンボルを
同一順序で持つため、どちらの archive からでも `cancer_gene_census.csv` はバイト単位で同じになります。

### ゲノムアセンブリは hg38

変異座標の出所が Cell Model Passports（**GRCh38**）に変わったため、
変異シグネチャの参照塩基取得も hg38 でなければなりません（`HG2BIT: raw/hg38.2bit`）。
検証結果は、サンプリングした SNV 20,000 件中 20,000 件が hg38 と一致、hg19 では
19,925 件中 4,874 件（24.5%＝偶然一致の水準）で、約 17k の座標が hg19 の染色体長を超えます。
`hg19.2bit` は検証の痕跡として残っているだけで、パイプラインは使いません。

`cancer_gene_census.csv` は整合性のため GRCh38 版から取っていますが、座標は一切読まれません。

### 生成物（`raw/` に置かれるが取得物ではない）

- **`fingerprints.csv`** — 化合物フィンガープリント。原著者は PaDEL-Descriptor GUI で
  3種 × 1024 ビットを生成しましたが、PaDEL は保守停止・Java 専用・API なしのため
  RDKit に置き換えています（`scripts/cdr/upstream/create_fingerprints.py`）。
  PubChem CID から SMILES を取得し、`Name` 列（GDSC Drug ID）＋ 3×1024 ビットという
  PaDEL と同じ 3,073 列の形で書き出します。使用する3種は
  Morgan/ECFP4（radius 2）、RDKit topological、hashed atom pair。
  **PaDEL の再実装ではなく別の記述子**なので、フィンガープリント由来の結果は論文と直接比較できません。
- **`maf.csv`** — 変異シグネチャ計算用の MAF キャッシュ。存在すれば必ず再利用されます。
  **変異ソースを変えたら `prepare_data.py` の再実行前に必ず削除してください**
  （indel のアレルはキャッシュ作成時のゲノムから読まれているため）。

---

## 4. 上流ステージ ②（`prepare_data.py`）

`configs/cdr/upstream.json` が入力設定です（GraphCDRScan の `config.json` から
前処理キーだけを残したもの。パスは `--data-root`＝`data_cdr` からの相対）。

主な処理:

1. **グラフ生成** — Reactome FI を読み、重複除去 → **両方向化** → 自己ループ除去。
   Ensembl ID を HGNC ID に変換し、ノード集合と関係集合を **`sorted()` してから**
   整数 ID を振ります（Python の文字列ハッシュがプロセスごとに変わるため、
   ソートしないと実行のたびにエンコードが変わっていました）。
2. **化合物データ** — 使用化合物リスト × Screened Compounds × フィンガープリント ×
   用量反応データを結合し、`COSMIC_ID` / `LN_IC50` / `fingerprints` にまとめます。
3. **変異シグネチャ** — hg38 から参照文脈を読み、96/78/83 コンテキストの変異スペクトルを
   細胞株ごとに算出します（`signatureanalyzer`）。
4. **変異データ** — 変異タイプ one-hot を付与し、Cancer Gene Census の遺伝子と
   使用細胞株で絞り込み、変異座標のエンコードを付けます。
5. `node_features` / `sample_features` / `labels` を書き出します。

`config.json` のフラグ（すべて既定 `true`）: `VARIANT_TYPE`, `MUTATION_FEATURES`,
`CANCER_TYPE`, `SPECTRA96`, `SPECTRA78`, `SPECTRA83`。`FOLDER: full_features` が
出力サブディレクトリ名です。

---

## 5. 上流バンドル（`processed/full_features/`）

すべてヘッダなし TSV。**これが `cdr-prepare` の入力**です。

### `vertices_dic.tsv` / `relationships_dic.tsv`

`名前<TAB>ID` の2列。ノード名は **HGNC の数値 ID**（遺伝子シンボルではありません）。

| ファイル | 件数 |
| --- | --- |
| `vertices_dic.tsv` | 13,606 |
| `relationships_dic.tsv` | 356 |

関係タイプが 356 種もあるのは、Reactome FI のアノテーションが組み合わせとして
文字列化されているためです（`activate`, `activate; activated by`,
`activate; activated by; catalyze`, …）。

### `graph.tsv` — 3列（起点 ID / 関係 ID / 終点 ID）、536,274 行

上流で対称化・自己ループ除去済みです。

### `labels.tsv` — 3列、107,418 行

| 列 | 内容 |
| --- | --- |
| 1 | サンプル ID（0〜107,417） |
| 2 | `LN_IC50`（実数。範囲 −9.80 〜 12.35、中央値 2.73） |
| 3 | `IC50` = `exp(LN_IC50)` |

### `sample_features.tsv` — 3,350 列、107,418 行

| 列（0始まり） | 内容 |
| --- | --- |
| 0 | サンプル ID |
| 1 | がん種コード（出現順。`groups` の元） |
| 2〜258 | 変異スペクトル 257 次元（96 + 78 + 83 の順） |
| 259〜277 | 原発部位 one-hot 19 次元 |
| 278〜3349 | 化合物フィンガープリント 3,072 ビット（3 × 1024） |

**原発部位 one-hot は `pd.get_dummies` の出力なのでアルファベット順**です
（Bladder, Bone, Breast, Central Nervous System, Cervix, Esophagus,
Haematopoietic and Lymphoid, Head and Neck, Kidney, Large Intestine, Liver, Lung,
Ovary, Pancreas, Peripheral Nervous System, Skin, Stomach, Thyroid, Vulva）。
この事実は `raw/CosmicCLP_MutantExport.tsv` を `used_cell_lines.csv` で絞った結果と
照合して検証済みで、`cdr/prepare.py` の `PRIMARY_SITES` に定数として置いてあります。

### `node_features.tsv` — 23 列、27,309,391 行（9.1 GB）

**1行 = 1変異**です。同じ (サンプル, 遺伝子) の組が複数行に現れます。

| 列（0始まり） | 内容 |
| --- | --- |
| 0 | サンプル ID |
| 1 | グラフのノード ID |
| 2〜4 | 変異タイプ one-hot 3 次元（アルファベット順に Deletion / Insertion / Substitution） |
| 5〜16 | 変異開始座標の位置エンコード 12 次元（下位から3桁ずつ3ブロック × 4 次元） |
| 17〜19 | 染色体番号の位置エンコード 3 次元 |
| 20〜22 | 変異長（終端−始端）の位置エンコード 3 次元 |

サンプル ID 昇順にグループ化されて並んでいます。1サンプルあたりの
**ユニークな**変異遺伝子数は中央値 126、95 パーセンタイル 352、最大 638。

---

## 6. 汎用形式への変換（`pathwaygnn-data cdr-prepare`）

```bash
bash scripts/cdr/prepare.sh
# = pathwaygnn-data cdr-prepare --config configs/cdr/prepare.yaml
```

実装は `src/pathwaygnn_datasets/cdr/prepare.py`。numpy のみで 9.1 GB を1回だけ走査します。

### 6.1 グラフ

`vertices_dic` / `relationships_dic` の ID が 0 始まりの連番であることを検証し、
`graph.tsv` を重複除去・ソートして決定的な `edge_index` / `edge_type` にします。
上流で既に対称化されているため**ここでは対称化しません**が、対称性が保たれているかを
検査して `dataset.json` の `graph_symmetric` に記録します（実データでは `true`、
自己ループ・重複ともに 0 件）。

### 6.2 channel `mutation`（sparse）

1行1変異の `node_features.tsv` を **(サンプル, 遺伝子) ごとの変異数**に集約します。
GDSC のサンプルは *(細胞株, 化合物)* の組なので、同じ細胞株に対して試された化合物の数だけ
同一の変異プロファイルが繰り返されます。そこで**プロファイルの内容でハッシュして重複排除**し、
`rows/mutation.npy` でサンプル → 行を写像します。

| | 変換前（`node_features.tsv`） | 変換後（channel `mutation`） |
| --- | --- | --- |
| 行の意味 | 1変異（サンプルあたり平均 254 行） | 1変異プロファイル |
| 行数 | 27,309,391 | **760**（＝細胞株数） |
| 非ゼロ値 | 16,605,297 の (サンプル, 遺伝子) 組 | **117,880**（プロファイルあたり平均 155 遺伝子） |
| サイズ | 9.1 GB | 約 1 MB |

`binary_mutations: true` にすると値が変異数ではなく 1.0 になります。

> **設計上の割り切り**: サンプルレベルヘッドは遺伝子ごとに **スカラー1つ**しか射影しないため、
> 上流の21次元ノード特徴（変異タイプ、座標エンコード）は変異数に縮約されます。
> 変異タイプ・座標の情報はサンプルレベルの変異スペクトル（共変量側）に部分的に残ります。

### 6.3 covariates

`sample_features.tsv` の**サンプル ID とがん種コードを除いた残り全部**、
すなわち 3,348 次元（スペクトル 257 + 部位 one-hot 19 + フィンガープリント 3,072）を
そのまま共変量にします。名前は `spectra96_*` / `spectra78_*` / `spectra83_*` /
`site_<部位名>` / `fingerprint_*` と付き、IG の出力がそのまま読めます。

書き出しはタスクあたり 1.4 GB になるため、一時 memmap 経由でストリーム書き込みします。

### 6.4 groups

がん種コードを one-hot の位置（＝アルファベット順）に振り直したものを `groups` にします。
上流のコードは「出現順」なので、そのままでは部位名と対応しません。
サンプルごとに「コード」と「one-hot 位置」の両方が入っているため対応関係は
データから復元でき、矛盾があれば失敗します。

| 部位 | サンプル数 | | 部位 | サンプル数 |
| --- | --- | --- | --- | --- |
| Lung | 21,696 | | Bladder | 2,576 |
| Haematopoietic and Lymphoid | 17,092 | | Thyroid | 2,290 |
| Skin | 7,841 | | Liver | 2,180 |
| Central Nervous System | 7,420 | | Cervix | 1,828 |
| Breast | 7,103 | | Vulva | 141 |
| Large Intestine | 6,422 | | Bone | 137 |
| Head and Neck | 5,487 | | | |
| Esophagus | 4,947 | | | |
| Peripheral Nervous System | 4,703 | | | |
| Kidney | 4,403 | | | |
| Ovary | 4,156 | | | |
| Pancreas | 3,887 | | | |
| Stomach | 3,109 | | | |

### 6.5 task（`LN_IC50` の二値化）

`pathwaygnn` は二値分類しか扱わないため、**閾値の定義は前処理の一部**であり
`task.json` の `source` に記録されます。ラベル 1 = 感受性（`LN_IC50` が基準中央値未満）。

| task | 基準 | サンプル数 | 陽性数 | 陽性率 |
| --- | --- | --- | --- | --- |
| `sensitive_drugwise` | **同一化合物**の `LN_IC50` 中央値 | 107,418 | 53,669 | 49.96% |
| `sensitive_global` | 全サンプルの `LN_IC50` 中央値 | 107,418 | 53,709 | 50.00% |

`sensitive_drugwise` が本命です。化合物ごとに中央値で切るので、化合物固有の効力は
ラベルから消え、**細胞株のゲノムだけが手がかり**になります。
`sensitive_global` は逆に化合物の同定でほぼ説明がついてしまいます。

### 6.6 復元した情報

上流バンドルは `DRUG_ID` と `COSMIC_ID` を落としてしまっています。`cdr-prepare` は
これらを推論で復元します（推測ではなく検証付き）。

- **化合物** — フィンガープリント 3,072 ビットが化合物を一意に識別する → 168 化合物
- **細胞株** — がん種コード + 変異スペクトル 257 次元が細胞株を識別する → 760 細胞株

### 生成される汎用形式のまとめ

| 項目 | 値 |
| --- | --- |
| ノード数 | 13,606 |
| 関係タイプ数 | 356 |
| 有向エッジ数 | 536,274 |
| サンプル数 | 107,418（760 細胞株 × 168 化合物の一部） |
| channel | `mutation`（sparse、760 行 / 117,880 値） |
| covariates | 3,348 次元 |
| groups | 原発部位 19 種 |
| tasks | `sensitive_drugwise`, `sensitive_global` |

---

## 7. 実行

```bash
conda activate gnn
bash scripts/cdr/reproduce.sh
```

内訳（個別実行も可）:

```bash
pathwaygnn-data cdr-prepare --config configs/cdr/prepare.yaml
pathwaygnn pretrain  --config configs/cdr/pretrain.yaml
python scripts/cdr/run_cv.py --gpus 0,1,2      # 8条件 (2 task × 4 variant) を GPU に分配
pathwaygnn finetune  --config configs/cdr/finetune_drugwise.yaml
pathwaygnn finetune  --config configs/cdr/finetune_global.yaml
pathwaygnn benchmark --config configs/cdr/benchmark_drugwise.yaml
pathwaygnn benchmark --config configs/cdr/benchmark_global.yaml
pathwaygnn ig        --config configs/cdr/ig_drugwise.yaml
pathwaygnn ig        --config configs/cdr/ig_global.yaml
pathwaygnn-data cdr-report --config configs/cdr/report.yaml
```

`run_cv.py` は `configs/cdr/cv.yaml` を (task, variant) ごとの設定に分割し、
1条件 = 1プロセスとして GPU スロットに割り当てます。`cv` は fold 単位で再開するので、
再実行は出力ディレクトリのない fold だけを計算します。単純に順番に回したいなら
`pathwaygnn cv --config configs/cdr/cv.yaml` で同じ 40 fold を1プロセスで実行できます。

---

## 8. 計算コストの注意点

**関係タイプが 356 種あることがコストを支配します。** `RelationalGIN` は
「層 × 関係」ごとに GINConv を作るため、encoder は 9.8 M パラメータになり、
全グラフ forward + backward が RTX PRO 6000 1枚で約 0.5 秒（`no_grad` で 0.31 秒、
勾配ありのピークメモリ 15 GB）かかります。

- `configs/cdr/cv.yaml` は **`end_to_end: false`** を維持してください。埋め込みが
  fold ごとに1回だけ計算され、ステップごとには計算されなくなります。
- 事前学習では `training.steps_per_epoch` を小さく保ってください。コストはサンプル数ではなく
  ステップ数に比例します（同梱設定は 100 エポック × 5 ステップ ≒ 4 分）。
- `ig` は 1サンプル × 1積分ステップごとにこの 0.5 秒を払います。同梱設定が
  40 サンプル × 25 ステップなのはそのためです（1タスク約 8 分）。

`benchmark` は 107,418 × 16,954 の疎行列（1行あたり非ゼロ約 1,550）を扱うため、
Random Forest は木数 60・深さ 12 に制限してあります。これは調整済みモデルではなく
**参照点**です（`LogisticRegression` は `max_iter=1000` に達しても収束しません）。

---

## 9. 結果の要約

5-fold 交差検証の平均 ROC-AUC（詳細は [`docs/cdr_report.md`](docs/cdr_report.md)）:

| task | `mlp` | `mlp_cov` | `gnn_mlp` | `gnn_mlp_cov` | 最良ベースライン |
| --- | --- | --- | --- | --- | --- |
| `sensitive_drugwise` | 0.516 | 0.718 | 0.551 | 0.723 | xgboost 0.758 |
| `sensitive_global` | 0.502 | 0.924 | 0.507 | 0.924 | xgboost 0.933 |

読み方の注意:

- **グラフの寄与は共変量なしの条件でしか見えません。** `sensitive_drugwise` で
  +0.035（0.516 → 0.551）、共変量を入れると +0.004 に縮みます。
- **2つのタスクは構造的に難易度が違います。** 互いの AUC を比べても、モデルではなく
  ラベル定義の話にしかなりません。
- **fold はサンプル単位のランダム分割で、細胞株単位でも化合物単位でもありません。**
  同じ細胞株が別の化合物として学習側にも held-out 側にも現れるため、これは
  「部分観測の応答行列を埋める」課題であり、未知の細胞株への汎化ではありません。
- **木系ベースラインの方が強い**（両タスクとも xgboost が上）。
- **IG のノードランキングは次数に強く相関します**（r ≈ 0.75〜0.76）。上位は Reactome FI の
  ハブであるユビキチン／リボソーム遺伝子（UBC, UBB, UBA52, RPS27A）が占めます。
- **これは 2018 年の CDRscan 実験ではありません。** GDSC1（2023年10月）、GRCh38 上の
  Cell Model Passports 変異、RDKit フィンガープリントで代替しているため、
  公開されている CDRscan の数値とは比較できません。
