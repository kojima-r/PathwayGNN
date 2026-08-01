# README_data_cancer — TCGA がん予後予測（`data_cancer`）

TCGA の遺伝子発現プロファイルから、診断後 n 年（n = 1〜5）時点の生存を二値予測します。
Inoue et al. の論文（`docs/papers/Arxiv_INOUE_0629.pdf`）の再現対象データセットで、
由来は `SampleLevelGNN` / `DistributedGNN`。

- 設定項目の説明: [README_config.md](README_config.md)
- 再現結果: [`docs/cancer_reproduction.md`](docs/cancer_reproduction.md) /
  [`docs/cancer_reproduction.html`](docs/cancer_reproduction.html)

---

## 1. ディレクトリ構成

```text
data_cancer/
├── processed/                論文コードが公開した「上流バンドル」＝前処理の入力
│   ├── graph.tsv                      52 MB
│   ├── vertices_dic.tsv
│   ├── relationships_dic.tsv
│   ├── <n>years_node_input.tsv        1〜5年分で計 4.2 GB
│   ├── <n>years_sample.tsv
│   └── <n>years_labels.tsv
├── rawdata_TCGA/             さらに上流の素材（本パイプラインは直接読まない）
│   ├── counts_gene.tsv                2.7 GB
│   ├── TCGA_ID.tsv                     50 MB
│   └── mmc1.csv                       3.1 MB
├── PathwayCommons13.All.hgnc.txt      1.0 GB（参考。グラフは processed/ 側を使う）
├── artifacts/                リファクタ前のレイアウト。もう読まれない
└── prepared/                 生成物（.gitignore 対象）
```

**`processed/` は出力ではなく入力です。** 論文コードが公開した名前をそのまま維持しており、
`data_cdr/processed/` と同じ位置づけです（`data_tr` だけが `raw/` から始まります）。

---

## 2. 元データ

### 2.1 上流バンドル `processed/`

こちらが `cancer-prepare` が実際に読むファイル群です。すべてヘッダなし TSV で、
ID は整数エンコード済みです。

#### `graph.tsv` — PathwayCommons グラフ

```text
18832	12	14768
18832	8	10104
```

| 列 | 内容 |
| --- | --- |
| 1 | 起点ノード ID |
| 2 | 関係タイプ ID |
| 3 | 終点ノード ID |

**行の並びはそのまま使われます**（`tr` と違って再ソートも対称化もしません）。事前学習の
再現性がこの順序に依存しているためです。

#### `vertices_dic.tsv` / `relationships_dic.tsv` — 整数エンコードの辞書

`名前<TAB>ID` の2列。ID が 0 始まりの連番でなければ前処理は失敗します。
関係タイプは 13 種で、`tr` と同じ PathwayCommons の語彙です。

```text
controls-transport-of-chemical	0
reacts-with	1
...
controls-expression-of	12
```

#### `<n>years_node_input.tsv` — 発現プロファイル（3列ロング形式）

```text
0	25898	0.0
```

| 列 | 内容 |
| --- | --- |
| 1 | サンプル ID |
| 2 | グラフのノード ID（＝遺伝子） |
| 3 | 発現値 |

**1サンプルあたり 4,448 行**が同じ遺伝子順で並びます。1〜5年分で合計 4.2 GB。
先頭行が `0\t1\t2` というシリアライズ済みヘッダの場合があり、その場合はスキップされます。

#### `<n>years_sample.tsv` — サンプル属性（35列）

| 列 | 内容 |
| --- | --- |
| 1 | サンプル ID |
| 2 | がん種コード（0〜32） |
| 3〜35 | がん種の one-hot（33種） |

#### `<n>years_labels.tsv` — ラベル（2列）

| 列 | 内容 |
| --- | --- |
| 1 | サンプル ID |
| 2 | ラベル。**1 = 生存、0 = 死亡** |

### 2.2 `rawdata_TCGA/`（参考素材）

`processed/` を作った元の公開データです。本パイプラインは読み込みません。

| ファイル | 内容 |
| --- | --- |
| `counts_gene.tsv` | recount2 由来の遺伝子カウント行列。58,037 遺伝子（行）× 11,285 サンプル（列）。列名は GDC の UUID、**行名の列を持たない** |
| `TCGA_ID.tsv` | サンプルのメタデータ（GDC / SRA の各種 ID、臨床情報を含む横長テーブル） |
| `mmc1.csv` | TCGA Pan-Cancer Clinical Data Resource（11,160 症例。`vital_status`、`death_days_to`、`last_contact_days_to` など） |

`counts_gene.tsv` は行（遺伝子）の ID 列を持たないため、順序付き Ensembl ID リストが別途必要です。
それがある場合は次のコマンドで HGNC ID に変換できます。

```bash
pathwaygnn-data cancer-map-ids --config configs/cancer/id_mapping.yaml
```

MyGene.info の応答を丸ごとキャッシュし、**入力リストの SHA-256 が一致しないキャッシュは
再利用を拒否**するため、ID 変換は監査可能かつオフラインで再現できます。

### 2.3 論文由来の定数

`src/pathwaygnn_datasets/cancer/paper.py` に検証済み定数がチェックインされています。

| 定数 | 内容 |
| --- | --- |
| `CANCER_TYPES` | 33 のがん種略号（ACC, BLCA, …, UVM） |
| `PAPER_SAMPLE_COUNTS` | 年別サンプル数 `{1: 9484, 2: 7308, 3: 5915, 4: 5036, 5: 4492}` |
| `PAPER_TABLE1` | 論文 Table 1 の AUC 4条件 × 5年 |
| `PAPER_VARIANTS` | 4条件（`dnn`, `dnn_cancer`, `gnn_dnn`, `gnn_dnn_cancer`）と `seed_index` |

`cancer-prepare` は年別サンプル数を `PAPER_SAMPLE_COUNTS` と照合し、合わなければ失敗します。
`cancer-report` は「論文値 / 再現値 / 差分」の表を出力します。

---

## 3. 前処理（`pathwaygnn-data cancer-prepare`）

```bash
bash scripts/cancer/reproduce_paper.sh prepare
# = pathwaygnn-data cancer-prepare --config configs/cancer/prepare.yaml
```

実装は `src/pathwaygnn_datasets/cancer/prepare.py`。処理内容は次の通りです。

1. **グラフ** — `graph.tsv` の行順をそのまま `edge_index` / `edge_type` にします。
2. **発現行列の変換** — 3列ロング形式の TSV を 64 MB ずつ読みながら
   `matrix.npy`（`float32[num_samples, 4448]`、memmap）へ流し込みます。合計 4.2 GB を
   約1分半で処理します。この間、
   - サンプル ID が `0,0,…,0,1,1,…` の順に並んでいるか
   - 遺伝子 ID の並びが全サンプルで同一か

   を検証し、崩れていればその位置を示して失敗します。
   **既存の `matrix.npy` の形状が一致していれば変換をスキップ**するので、再実行は安価です。
3. **タスク生成** — 年ごとに `labels` と `sample` をサンプル ID でソートしてから、
   ラベル・共変量（がん種 one-hot 33次元）・グループ（がん種コード）を書き出します。
4. `seed_offset` にはその年（1〜5）を入れます。これにより
   「何年分を含む実行か」に依存せず fold seed が決まります。

### 既知のレガシー仕様（意図的に残しています）

`<n>years_sample.tsv` の**行 0 は `0,1,2,…,34` というシリアライズされたヘッダ**で、
データ行ではありません。これを削除すると公開コードとサンプル対応がずれるため、
そのまま1サンプルとして保持しています。`dataset.json` の `known_legacy_issue` と
各 `task.json` の `legacy_header_row_retained` に記録されています（3年のみ `false`）。
**「バグに見えるから直す」ことはしないでください。**

### 生成される汎用形式

| 項目 | 値 |
| --- | --- |
| ノード数 | 30,918 |
| 関係タイプ数 | 13 |
| 有向エッジ数 | 3,673,654 |
| 遺伝子数（dense 行列の列） | 4,448 |

**channel（すべて dense、memmap）**

| channel | 行数 | 列数 |
| --- | --- | --- |
| `expression_1year` | 9,484 | 4,448 |
| `expression_2year` | 7,308 | 4,448 |
| `expression_3year` | 5,915 | 4,448 |
| `expression_4year` | 5,036 | 4,448 |
| `expression_5year` | 4,492 | 4,448 |

**task**

| task | サンプル数 | 生存（陽性） | 死亡 | alias → channel |
| --- | --- | --- | --- | --- |
| `1year` | 9,484 | 8,408 | 1,076 | `expression`→`expression_1year` |
| `2year` | 7,308 | 5,357 | 1,951 | `expression`→`expression_2year` |
| `3year` | 5,915 | 3,490 | 2,425 | `expression`→`expression_3year` |
| `4year` | 5,036 | 2,323 | 2,713 | `expression`→`expression_4year` |
| `5year` | 4,492 | 1,570 | 2,922 | `expression`→`expression_5year` |

5タスクとも **alias は `expression` で共通**なので、モデル設定が相互に流用できます。
共変量はがん種 one-hot（33次元）、グループもがん種です。
年が進むほど陽性率が下がり（1年 88.7% → 5年 34.9%）、タスクの難易度が変わります。

---

## 4. 実行

```bash
conda activate gnn

# 前処理 → 3 GPU 分散事前学習 → Table 1 全20条件 → Figure 2 → IG → レポート
bash scripts/cancer/reproduce_paper.sh full

# 段階ごとに実行する場合
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

## 5. 再現上の注意点

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
