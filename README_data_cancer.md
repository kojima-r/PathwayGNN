# README_data_cancer — TCGA がん予後予測（`data_cancer`）

TCGA の遺伝子発現プロファイルから、診断後 n 年（n = 1〜5）時点の生存を二値予測します。
Inoue et al. の論文（`docs/papers/Arxiv_INOUE_0629.pdf`）の再現対象データセットで、
由来は `SampleLevelGNN` / `DistributedGNN`。

- 設定項目の説明: [README_config.md](README_config.md)
- 再現結果: [`docs/cancer_reproduction.md`](docs/cancer_reproduction.md) /
  [`docs/cancer_reproduction.html`](docs/cancer_reproduction.html)

---

## 1. パイプラインの3段階

生データから学習可能な形式まで、2つのコマンドで到達します。

```text
rawdata_TCGA/ + PathwayCommons SIF
        │
        │ ①  pathwaygnn-data cancer-build-processed
        ▼
data_cancer/processed/          論文コードが公開したのと同じレイアウトの中間バンドル
        │
        │ ②  pathwaygnn-data cancer-prepare
        ▼
data_cancer/prepared/           pathwaygnn が読む汎用形式
```

| 段階 | コマンド | 実装 |
| --- | --- | --- |
| ① バンドル構築 | `pathwaygnn-data cancer-build-processed --config configs/cancer/build_processed.yaml` | `src/pathwaygnn_datasets/cancer/build.py` |
| ② 汎用形式へ変換 | `pathwaygnn-data cancer-prepare --config configs/cancer/prepare.yaml` | `src/pathwaygnn_datasets/cancer/prepare.py` |

```bash
conda activate gnn
bash scripts/cancer/reproduce_paper.sh build-processed   # ①（生データから作り直す場合のみ）
bash scripts/cancer/reproduce_paper.sh prepare           # ②
```

**`data_cancer/processed/` が既にある場合、①は不要です**（②から始めてください）。
①は後述する2つの入力が別途必要です。

---

## 2. ディレクトリ構成

```text
data_cancer/
├── rawdata_TCGA/             生データ
│   ├── counts_gene.tsv                2.7 GB  recount2 の TCGA 遺伝子カウント
│   ├── TCGA_ID.tsv                     50 MB  recount2 のサンプルメタデータ
│   ├── mmc1.csv                       3.1 MB  TCGA-CDR 臨床情報（Liu et al. 2018）
│   ├── ensembl_gene_ids.txt           ← 別途用意（§4）
│   └── msigdb.gmt / LM22.txt          ← 別途用意（§4）
├── PathwayCommons12.All.hgnc.sif      58 MB  パスウェイグラフ（§3.1）
├── processed/                ①の出力＝②の入力
└── prepared/                 ②の出力（.gitignore 対象）
```

グラフの入力は `data_cancer/PathwayCommons12.All.hgnc.sif` です（`data_tr/raw/` にある
**Git 管理下**の同名ファイルと SHA-256 が一致する複製で、どちらを指しても結果は同じです）。
HGNC 対応表は `data_cdr/raw/EnsemblToHGNC.tsv` を既定で使います（HGNC の
complete-set エクスポートなら何でも構いません）。

---

## 3. 元データ

### 3.1 グラフ: PathwayCommons

論文はナレッジグラフとして Pathway Commons を使い、13種の関係タイプを持つと記述しています。
**公開バンドルのグラフは PathwayCommons 12 と一致します**。次の3点を照合して確認済みです。

| | 公開バンドル | `PathwayCommons12.All.hgnc.sif` から再構築 |
| --- | --- | --- |
| ノード数 | 30,918 | **30,918** |
| 関係タイプ数 | 13 | **13** |
| 有向エッジ数 | 3,673,654 | **3,673,654** |

形式は3列 SIF（`.txt` 版はヘッダと注釈列つきで、どちらも読めます）。

```text
A1BG	controls-expression-of	A2M
A1BG	interacts-with	ABCC6
```

`cancer-build-processed` は、各エッジを両方向に追加して対称化し、遺伝子シンボルを
**数値 HGNC ID** に置換して（対応がないもの＝`CHEBI:*` などはそのまま）整数エンコードします。
`data_tr` がシンボルのままノード名にするのに対し、`data_cancer` は数値 ID を使う、というのが
両者の唯一の違いです。

### 3.2 発現量: `rawdata_TCGA/counts_gene.tsv`

recount2 が TCGA の RNA-Seq から推定した遺伝子カウント行列。

| | |
| --- | --- |
| 行 | 58,037 遺伝子。**行 ID の列を持たない**（§4） |
| 列 | 11,285 サンプル。列名は GDC のファイル UUID |
| 値 | カウント（例: 586401） |

### 3.3 サンプル対応: `rawdata_TCGA/TCGA_ID.tsv`

recount2 のメタデータ。横長ですが、使うのは3列だけです。

| 列 | 用途 |
| --- | --- |
| `gdc_file_id` | `counts_gene.tsv` の列名（UUID、大文字小文字違い）と対応 |
| `gdc_cases.submitter_id` | TCGA 患者バーコード（`mmc1.csv` と対応） |
| `gdc_cases.project.project_id` | `TCGA-LIHC` などのがん種 |

> UTF-8 として不正なバイトを含むため、`latin-1` で読みます。

### 3.4 臨床情報: `rawdata_TCGA/mmc1.csv`

TCGA Pan-Cancer Clinical Data Resource（Liu et al. 2018）。11,160 症例。

| 列 | 用途 |
| --- | --- |
| `bcr_patient_barcode` | 患者バーコード |
| `type` | がん種略号（33種。`ACC`〜`UVM`） |
| `vital_status` | `Alive` / `Dead` |
| `last_contact_days_to` | 生存例の最終確認日数 |
| `death_days_to` | 死亡例の死亡日数 |

### 3.5 論文由来の定数

`src/pathwaygnn_datasets/cancer/paper.py` に検証済み定数がチェックインされています。

| 定数 | 内容 |
| --- | --- |
| `CANCER_TYPES` | 33 のがん種略号（アルファベット順。one-hot の並びと一致） |
| `PAPER_SAMPLE_COUNTS` | 年別サンプル数 `{1: 9484, 2: 7308, 3: 5915, 4: 5036, 5: 4492}` |
| `PAPER_TABLE1` | 論文 Table 1 の AUC 4条件 × 5年 |
| `PAPER_VARIANTS` | 4条件（`dnn`, `dnn_cancer`, `gnn_dnn`, `gnn_dnn_cancer`）と `seed_index` |

---

## 4. 別途用意が必要な2つの入力

**これらはライセンスと入手方法の都合でリポジトリに同梱できません。**
①を実行するときだけ必要で、②以降には不要です。

### 4.1 `ensembl_gene_ids.txt` — 発現行列の行 ID

`counts_gene.tsv` は行名を持たないため、**58,037 行の並び順に対応する遺伝子 ID の一覧**が
別途必要です。recount2 の遺伝子アノテーション（Gencode v25）がその並び順を定義しています。

形式は1行1 ID。第2列に `bp_length` を置くと `transform: log1p_tpm` が使えます。

```text
ENSG00000000003.14	4536
ENSG00000000005.6	1476
```

Ensembl ID 以外に、遺伝子シンボルや数値 HGNC ID でも受け付けます。
`cancer-map-ids` で MyGene.info 経由の HGNC 対応表を別に作り、`ensembl_to_hgnc:` として
渡すこともできます（応答をキャッシュし、**入力リストの SHA-256 が一致しないキャッシュは
再利用を拒否**するため、ID 変換は監査可能かつオフラインで再現できます）。

```bash
pathwaygnn-data cancer-map-ids --config configs/cancer/id_mapping.yaml
```

### 4.2 `gene_sets` — がん関連遺伝子の母集合

論文 2.2 節: 「MSigDB と LM22 免疫遺伝子シグネチャに掲載された遺伝子を選択し、
ナレッジグラフのノードに存在しない遺伝子を除外した。4,448 遺伝子の発現量を使用した」。

- **MSigDB**（Liberzon et al. 2015）— 登録が必要。`.gmt` 形式をそのまま渡せます。
- **LM22**（Newman et al. 2015、CIBERSORT）— 登録が必要。1列目に遺伝子シンボルを持つ
  TSV をそのまま渡せます。

`.gmt` は3列目以降を、それ以外は1列目を遺伝子シンボルとして読みます。複数指定すると和集合です。
最終的な選択は **(gene_sets の和集合) ∩ (グラフのノード)** で、論文の 4,448 遺伝子に対応します。

---

## 5. ①`cancer-build-processed` の処理内容

設定は `configs/cancer/build_processed.yaml`。

1. **グラフ** — SIF を対称化 → シンボルを HGNC ID に置換 → **ソートしてから**整数 ID を付与し、
   `graph.tsv` / `vertices_dic.tsv` / `relationships_dic.tsv` を書き出します。
2. **サンプル選択**（論文 2.3 節をそのまま実装）
   - `counts_gene.tsv` の各列を `TCGA_ID.tsv` 経由で患者に、`mmc1.csv` 経由で生存情報に紐づける。
   - 打ち切り（生存中）サンプルの生存日数の中央値を求める。
   - 「中央値超の打ち切り例 ∪ 死亡例」の生存日数の95パーセンタイルを超えるサンプルを除外する。
   - 年 n について、n×365 日以内に打ち切られたサンプルは生死を判定できないので除外し、
     n 年到達なら label 1、未到達なら label 0。
3. **遺伝子選択** — §4.2 の通り。
4. **発現量の変換** — `transform: log1p`（既定）で `ln(count + 1)`。
5. `<n>years_{labels,sample,node_input}.tsv` と `build_manifest.json` を書き出します。

### 実データでの検証結果

`rawdata_TCGA/` に対して実行すると、論文が明記した中間値をすべて再現します。

| 項目 | 論文 | 再構築 |
| --- | --- | --- |
| 打ち切り例の生存日数の中央値 | 819 日 | **819 日** |
| 長期生存の除外閾値 | 3,595 日 | **3,595 日**（規則から導出） |
| 閾値超で除外 | 364 件 | 363 件 |
| 除外後のサンプル数 | 10,823 | **10,823** |

年別サンプル数:

| 年 | 論文 | 再構築 | 差 |
| --- | --- | --- | --- |
| 1 | 9,484 | 9,479 | −5 |
| 2 | 7,308 | **7,308** | 0 |
| 3 | 5,915 | 5,916 | +1 |
| 4 | 5,036 | **5,036** | 0 |
| 5 | 4,492 | **4,492** | 0 |

差分の原因は、`counts_gene.tsv` の 11,285 列のうち 1 列が `TCGA_ID.tsv` に無いこと、および
`vital_status` が `#N/A` / `[Discrepancy]` の症例（5件）と臨床情報を引けない列（39件）の
扱いです。合計は 0.05% 未満で、②の `strict_sample_counts: false` で受け入れられます。

### 公開バンドルとの意図的な差異

`build_manifest.json` にも記録されます。

1. **ノード・関係の整数エンコードがソート順である。** 公開バンドルは Python の集合の
   反復順で番号を振っており（`relationships_dic.tsv` がアルファベット順でないのはこのため）、
   プロセスごとに変わるため再現できません。**再構築したバンドルは内部的には整合していますが
   公開バンドルとは互換ではなく、片方で事前学習した encoder はもう片方では使えません。**
2. **`<n>years_sample.tsv` の先頭にヘッダ行がない。** §6 参照。
3. **発現量の変換。** 公開バンドルの値は最大 20.2 で `ln(10⁶)` ≈ 13.8 を超えるため、
   論文が記述する TPM ではありえません。既定は `log1p`（カウントの自然対数）です。
   論文の記述に合わせたい場合は `gene_ids` に `bp_length` を与えて `transform: log1p_tpm`
   を指定してください。

---

## 6. 中間バンドル（`processed/`）の形式

すべてヘッダなし TSV。ID は整数エンコード済みです。

#### `graph.tsv` — 3列（起点 ID / 関係 ID / 終点 ID）

**行の並びはそのまま使われます**（`data_tr` と違い、②では再ソートも対称化もしません）。
事前学習の再現性がこの順序に依存しているためです。

#### `vertices_dic.tsv` / `relationships_dic.tsv` — `名前<TAB>ID`

ID が 0 始まりの連番でなければ②は失敗します。ノード名は数値 HGNC ID（と `CHEBI:*`）です。

#### `<n>years_node_input.tsv` — 発現プロファイル（3列ロング形式）

| 列 | 内容 |
| --- | --- |
| 1 | サンプル ID |
| 2 | グラフのノード ID（＝遺伝子） |
| 3 | 発現値 |

**1サンプルあたり 4,448 行**が同じ遺伝子順で並びます。1〜5年分で合計 4.2 GB。

#### `<n>years_sample.tsv` — サンプル属性（35列）

| 列 | 内容 |
| --- | --- |
| 1 | サンプル ID |
| 2 | がん種コード（0〜32。`CANCER_TYPES` のインデックス） |
| 3〜35 | がん種の one-hot（33種。コードと同じ位置が 1） |

#### `<n>years_labels.tsv` — ラベル（2列）

| 列 | 内容 |
| --- | --- |
| 1 | サンプル ID |
| 2 | ラベル。**1 = 生存、0 = 死亡** |

### 公開バンドルの既知のレガシー仕様

公開バンドルの `<n>years_sample.tsv` の**行 0 は `0,1,2,…,34` というシリアライズされた
ヘッダ**で、データ行ではありません。削除すると公開コードとサンプル対応がずれるため、
②はそのまま1サンプルとして保持します（`dataset.json` の `known_legacy_issue` と各
`task.json` の `legacy_header_row_retained` に記録。3年のみ `false`）。
**「バグに見えるから直す」ことはしないでください。**
①で作り直したバンドルにはこの行がありません。

---

## 7. ②`cancer-prepare` の処理内容

```bash
bash scripts/cancer/reproduce_paper.sh prepare
# = pathwaygnn-data cancer-prepare --config configs/cancer/prepare.yaml
```

1. **グラフ** — `graph.tsv` の行順をそのまま `edge_index` / `edge_type` にします。
2. **発現行列の変換** — 3列ロング形式の TSV を 64 MB ずつ読みながら
   `matrix.npy`（`float32[num_samples, 4448]`、memmap）へ流し込みます。合計 4.2 GB を
   約1分半で処理します。この間、
   - サンプル ID が `0,0,…,0,1,1,…` の順に並んでいるか
   - 遺伝子 ID の並びが全サンプルで同一か

   を検証し、崩れていればその位置を示して失敗します。
   **既存の `matrix.npy` の形状が一致していれば変換をスキップ**するので、再実行は安価です。
3. **タスク生成** — 年ごとに `labels` と `sample` をサンプル ID でソートしてから、
   ラベル・sample-level feature（がん種 one-hot 33次元）・グループ（がん種コード）を書き出します。
4. `seed_offset` にはその年（1〜5）を入れます。これにより
   「何年分を含む実行か」に依存せず fold seed が決まります。

`strict_sample_counts`（既定 `true`）が有効なとき、年別サンプル数が `PAPER_SAMPLE_COUNTS`
と一致しなければ失敗します。①で作り直したバンドルを使う場合は `false` にしてください
（警告を出したうえで続行します）。

### 生成される汎用形式

| 項目 | 値 |
| --- | --- |
| ノード数 | 30,918 |
| 関係タイプ数 | 13 |
| 有向エッジ数 | 3,673,654 |
| 遺伝子数（dense 行列の列） | 4,448 |

**node-level feature（すべて dense、memmap）**

| node-level feature | 行数 | 列数 |
| --- | --- | --- |
| `expression_1year` | 9,484 | 4,448 |
| `expression_2year` | 7,308 | 4,448 |
| `expression_3year` | 5,915 | 4,448 |
| `expression_4year` | 5,036 | 4,448 |
| `expression_5year` | 4,492 | 4,448 |

**task**

| task | サンプル数 | 生存（陽性） | 死亡 | alias → node-level feature |
| --- | --- | --- | --- | --- |
| `1year` | 9,484 | 8,408 | 1,076 | `expression`→`expression_1year` |
| `2year` | 7,308 | 5,357 | 1,951 | `expression`→`expression_2year` |
| `3year` | 5,915 | 3,490 | 2,425 | `expression`→`expression_3year` |
| `4year` | 5,036 | 2,323 | 2,713 | `expression`→`expression_4year` |
| `5year` | 4,492 | 1,570 | 2,922 | `expression`→`expression_5year` |

5タスクとも **alias は `expression` で共通**なので、モデル設定が相互に流用できます。
sample-level featureはがん種 one-hot（33次元）、グループもがん種です。
年が進むほど陽性率が下がり（1年 88.7% → 5年 34.9%）、タスクの難易度が変わります。

---

## 8. 学習の実行

```bash
conda activate gnn

# 前処理 → 3 GPU 分散事前学習 → Table 1 全20条件 → Figure 2 → IG → レポート
bash scripts/cancer/reproduce_paper.sh full

# 段階ごとに実行する場合
bash scripts/cancer/reproduce_paper.sh build-processed  # ①（full には含まれない）
bash scripts/cancer/reproduce_paper.sh prepare
bash scripts/cancer/reproduce_paper.sh pretrain      # torchrun、既存 best.pt があれば再利用
bash scripts/cancer/reproduce_paper.sh table1        # 20条件を 3 GPU に分配
bash scripts/cancer/reproduce_paper.sh figure2       # 事前学習エポック別スイープ
bash scripts/cancer/reproduce_paper.sh ig
bash scripts/cancer/reproduce_paper.sh report
```

環境変数 `NPROC_PER_NODE`（既定 3）、`TABLE1_GPUS`（既定 `0,1,2`）、`JOBS_PER_GPU`（既定 1）、
`FORCE_PRETRAIN=1`（事前学習の再利用を無効化）で制御できます。

---

## 9. 再現上の注意点

- **`configs/cancer/cv.yaml` の設定は参照実装の癖をそのまま写しています。**
  `loss_clip: 0.01`、`loss_reduction: sum`、`grad_clip_value: 10.0`（ヘッドのみ）、
  `shuffle: false`、`num_layers: 2`（論文本文は3層）。コード側の汎用既定値は
  いずれも「素直な方」（`mean`、クリップなし）で、cancer 設定だけが明示的に癖を宣言しています。
  **プロトコルを変える意図がない限りこれらを触らないでください。**
- **`training.selection`** は `final_epoch`（論文プロトコル）が既定です。
  `best_test_auc` は held-out fold で選択する＝リークするため、公開コード互換のためだけに
  存在し、レポートでもその旨が明示されます。
- **fold 単位で再開できます。** 1 fold は RTX PRO 6000 1枚で 60〜110 秒、
  20条件のグリッド全体で 3 GPU 約50分。**再実行を強制するには fold ディレクトリを削除**します。
  リファクタ前の `model.pt` に対する後方互換シムは意図的に用意していません
  （`ig` がどの fold を再実行すべきか明示して失敗します）。
- **`ig` は高価です。** 同梱設定は held-out 全件（`max_samples: null`、899サンプル × 50ステップ
  ≒ 1 GPU で 2時間）を対象にします。サンプルループが終わるまで一切書き込まないため、
  中断しても前回の出力は無事です。変更の検証時は `max_samples` / `steps` を下げてください。
- **紛らわしい2つの出力ディレクトリ。**
  `outputs/cancer/pretrain_sweep/` はエポック別の encoder チェックポイント
  （`epoch_{0,10,…,50}.pt`）、`outputs/cancer/pretraining_sweep/` はそれらを使った CV 結果です。
  レポートは後者を `sweep_dir` として読みます。
- **GPU の非決定性は想定内です。** `end_to_end: true` では encoder の scatter-add が
  毎ステップ走るため、同一設定の2回の実行は 150 エポックの間に fold AUC で 1e-3 程度ずれます。
  リファクタの妥当性は最終 AUC ではなく **1エポック目の学習損失がビット一致するか**で判断してください。
- **バンドルを作り直したら事前学習もやり直しです。** ①のエンコードは公開バンドルと異なるため、
  `outputs/cancer/pretrain_50/best.pt` は流用できません（`load_encoder` はノード数・関係数しか
  照合しないので、**この場合エラーにならず黙って別のグラフを学習した encoder を使ってしまいます**。
  作り直したときは `outputs/cancer/` を消してください）。
