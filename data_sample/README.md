# data_sample — チュートリアル用の超小型データセット（合成）

**60サンプル・20遺伝子・27エッジの完全な合成データ**です。実測値ではありません。
このリポジトリの仕組み（グラフ事前学習 → サンプルレベル予測 → 交差検証 → 帰属解析）を
**CPUで合計2分**で一周し、`data_tr` / `data_cancer` / `data_cdr` を読む前に
「汎用データ形式とは何か」を理解するために置いています。

- チュートリアル本体: [`../README.md`](../README.md) の §4
- 設定ファイルの全項目: [`../README_config.md`](../README_config.md)

**このディレクトリの `raw/` は Git にコミットされています**（12 KB）。他の3コーパスは
サイズとライセンスの都合で Git 管理外なので、クローン直後に動かせるのはここだけです。
`prepared/` は生成物なので `.gitignore` 対象です。

---

## 1. ディレクトリ構成

```text
data_sample/
├── README.md                  この文書
├── raw/                       12 KB — Git 管理下。人が読める4つの TSV（§3）
│   ├── graph.tsv               926 B  パスウェイグラフ（3列 SIF 形式）
│   ├── expression.tsv          7.3 KB 発現量（60行 × 20遺伝子）
│   ├── tissue_signature.tsv    574 B  組織シグネチャ（長形式）
│   ├── samples.tsv             1.6 KB サンプル属性とラベル
│   └── manifest.json           1.2 KB 生成規則と件数（§2）
└── prepared/                  92 KB — 生成物（.gitignore 対象）。エンジンが読む汎用形式（§4）
```

パイプラインは**1段**だけです。他の3コーパスは生データが巨大・雑多なため
`raw → processed → prepared` の2段になっています。

```text
raw/ ──① pathwaygnn-data sample-prepare──▶ prepared/ ──▶ pathwaygnn pretrain / cv / ig / ...
```

---

## 2. データの作り方（＝正解が分かっている）

`raw/` の4ファイルは `scripts/sample/make_raw_data.py`（numpy のみ、seed 固定）が
生成したものです。**コミット済みなので実行は不要**ですが、規則を知っておくと
学習結果と帰属結果を「答え合わせ」できます。

```bash
python scripts/sample/make_raw_data.py            # data_sample/raw を再生成（同一バイト列）
python scripts/sample/make_raw_data.py --num-samples 200 --seed 1 --output-dir /tmp/raw
```

### 2.1 遺伝子は3つのモジュールに分かれている

| モジュール | 遺伝子 | 役割 |
| --- | --- | --- |
| `GROWTH1`…`GROWTH7` | 7 | `responder` を**正**方向に決める |
| `IMMUNE1`…`IMMUNE7` | 7 | `responder` を**負**方向、`relapse` を**正**方向に決める |
| `NOISE1`…`NOISE6` | 6 | **どのラベルにも効かない**（帰属解析が「拾ってはいけない」遺伝子） |

サンプルごとにモジュール活性 `activity`（3値）を引き、各遺伝子はその**ノイズ付き観測**です。

```text
expression[gene] = max(0, 4.0 + activity[module(gene)] + tissue_shift[tissue, gene] + N(0, 0.5²))
```

- `4.0` のオフセットは値を**非負**にするためです。実データの `log2(TPM+1)` と同じ性質で、
  これは飾りではありません。Integrated Gradients は「値 × 勾配」を**符号付きで**サンプル方向に
  足すため、平均0の行列だと精度の高いモデルではサンプル間で寄与が打ち消し合ってしまいます。
- 同一モジュールの遺伝子は連動します。グラフ（同一モジュール内を密に結んである）が
  効きうる唯一の構造がこれです。

### 2.2 ラベルは活性の関数

```text
responder = 1  if  2.0*growth - 2.0*immune + 0.8*z(stage) + N(0, 0.3²)  > その中央値
relapse   = 1  if  2.0*immune - 0.04*(age - 62)           + N(0, 0.3²)  > その中央値
                   （追跡できた48サンプルのみ）
```

- 中央値で切っているので**ラベルは厳密に均衡**です（30/30 と 24/24）。`pos_weight` は不要です。
- **2タスクで効くモジュールが違います**（`responder` は GROWTH と IMMUNE、`relapse` は
  IMMUNE と年齢のみ）。帰属解析をタスク間で比べる意味がここにあります。
- サンプルレベル特徴量のうち効くのは `responder` では `stage`、`relapse` では `age` だけで、
  `sex_female` と `smoker` は無関係です。

---

## 3. `raw/` の4ファイル

### 3.1 `graph.tsv` — パスウェイグラフ

ヘッダなし3列（起点 / 関係 / 終点）。**`data_cancer` と `data_tr` が読む
PathwayCommons の SIF と同じ形式**で、関係名も PathwayCommons の語彙から採っています。

```text
GROWTH1	controls-expression-of	GROWTH2
GROWTH1	in-complex-with	GROWTH3
GROWTH4	interacts-with	IMMUNE1
```

- 27行（無向エッジ27本）。前処理が逆向きを補って **54有向エッジ**になります。
- 関係3種: `controls-expression-of` / `in-complex-with` / `interacts-with`
  （実データは13種、`data_cdr` は356種）。
- モジュール内を鎖状＋ハブ（`GROWTH1` が4本）で結び、モジュール間は3本だけ。
  グラフ全体は連結です。

### 3.2 `expression.tsv` — 発現量（dense な node-level feature になる）

1行1サンプル、1列1遺伝子の**横長**の表。全サンプルに全遺伝子の値があるので dense です。

```text
sample_id	GROWTH1	GROWTH2	...	NOISE6
S01	4.262	3.338	...	3.874
```

### 3.3 `tissue_signature.tsv` — 組織シグネチャ（sparse な node-level feature になる）

`(キー, 遺伝子, 値)` の**縦長**の表。キーはサンプルではなく**組織**で、
組織あたりマーカー8遺伝子だけを持ちます（3組織 × 8 = 24行）。

```text
tissue	gene	value
TISSUE_A	GROWTH7	0.522
TISSUE_A	IMMUNE1	0.379
```

**60サンプルが3行を共有する**のがポイントです。`data_tr` が1つの疾患シグネチャ表を
2タスクで共有し、`data_cdr` が107,418サンプルを760行の変異プロファイルに畳むのと
同じ仕組み（`rows/<alias>.npy`）を、最小構成で見せています。

### 3.4 `samples.tsv` — 属性とラベル

| 列 | 汎用形式での役割 |
| --- | --- |
| `sample_id` | サンプルの順序（`expression.tsv` と一致していること） |
| `tissue` | **groups**（グループ別AUC・グループ別帰属の単位） |
| `age`, `sex_female`, `stage`, `smoker` | **sample-level features**（遺伝子に紐づかない密ベクトル） |
| `responder` | **task** `responder` のラベル（60サンプル） |
| `relapse` | **task** `relapse` のラベル。`NA` は「そのタスクの対象外」（12件） |

```text
sample_id	tissue	age	sex_female	stage	smoker	responder	relapse
S01	TISSUE_A	70	1	4	1	1	0
S02	TISSUE_B	63	0	3	0	1	NA
```

ラベル列を増やせばタスクが増えます（`src/pathwaygnn_datasets/sample/prepare.py` の
`LABEL_COLUMNS`）。

---

## 4. `prepared/` — 汎用形式との対応

`pathwaygnn-data sample-prepare --config configs/sample/prepare.yaml` の出力（92 KB）。
形式の定義は `src/pathwaygnn/data/format.py` です。

```text
prepared/
├── dataset.json                              マニフェスト。name=sample, 20ノード/3関係/54エッジ
├── graph.pt                                  {"edge_index": int64[2,54], "edge_type": int64[54]}
├── nodes.json                                20遺伝子名（ソート済み＝インデックスの定義）
├── relations.json                            3関係名（ソート済み）
├── node_features/
│   ├── expression/                           kind=dense, 60行 × 20遺伝子
│   │   ├── matrix.npy                        float32[60, 20]
│   │   └── gene_index.npy                    int64[20]  各列がどのノードか
│   └── tissue_signature/                     kind=sparse, 3行 / 値24個（CSR）
│       ├── ptr.npy                           int64[4]   行の境界
│       ├── gene.npy                          int64[24]  ノードインデックス
│       └── value.npy                         float32[24]
└── tasks/
    ├── responder/                            60サンプル（陽性30）
    │   ├── labels.npy, groups.npy, sample_features.npy   float32[60, 4]
    │   ├── rows/tissue_signature.npy          サンプル→組織行（60→3）
    │   └── task.json                          alias→テーブルの対応、グループ名、特徴量名
    └── relapse/                              48サンプル（陽性24）
        ├── rows/expression.npy                サンプル→発現行（48→60行のうち48）
        └── rows/tissue_signature.npy
```

この構造だけで、形式の全概念が1つずつ登場します。

| 概念 | ここでの実体 | 実データでの対応 |
| --- | --- | --- |
| dense node-level feature | `expression`（全遺伝子に値） | `cancer` の発現プロファイル |
| sparse node-level feature | `tissue_signature`（8遺伝子だけ） | `tr` の摂動/疾患シグネチャ、`cdr` の変異数 |
| 行マップ `rows/` | 60サンプル→3組織行、48サンプル→60行の部分集合 | `cdr` の107,418→760、`tr` の疾患行共有 |
| alias | 両タスクが `expression` / `tissue_signature` を同名で公開 | `cancer` の `1year`…`5year` が同一モデル設定を使える理由 |
| groups | 組織3種 | がん種、原発部位、対象疾患 |
| sample-level features | 年齢・性別・病期・喫煙 | がん種one-hot、変異スペクトル、化合物フィンガープリント |

**`responder` には `rows/expression.npy` がありません。** 全60サンプルが発現表の
60行と1対1なので、恒等写像＝ファイルなしで表現します。`relapse` は48サンプルなので
明示的に持ちます。どちらの分岐も1つのデータセットの中で見られるようにしてあります。

---

## 5. 実行と実測値

```bash
bash scripts/sample/run_all.sh        # 全段（合計 約2分、CPU）
python scripts/sample/summarize.py    # 結果を表にして表示（再実行不要）
```

段ごとの実測（1コアCPU、`conda run` の起動時間込み）:

| 段 | コマンド | 時間 |
| --- | --- | --- |
| ① 前処理 | `pathwaygnn-data sample-prepare` | 5 s |
| ② グラフ事前学習（50 epoch） | `pathwaygnn pretrain` | 10 s |
| ③ 交差検証（2タスク × 4条件 × 3fold = 24 fold） | `pathwaygnn cv` | 53 s |
| ④ 単一分割 | `pathwaygnn finetune` | 9 s |
| ⑤ グラフなしベースライン | `pathwaygnn benchmark` | 9 s |
| ⑥ Integrated Gradients（20サンプル × 25ステップ） | `pathwaygnn ig` | 10 s |

得られる数値と読み方は [`../README.md`](../README.md) の §4 に
まとめてあります。乱数の都合で AUC は環境により ±0.02 程度ぶれます。

---

## 6. 自分のデータで同じことをする

`src/pathwaygnn_datasets/sample/prepare.py` は約240行（うち40行は解説コメント）で、このリポジトリで
**最小の前処理実装**です。これをコピーして、

1. グラフ（`(起点, 関係, 終点)` の列挙）を `DatasetWriter.write_graph` に渡す
2. サンプル×遺伝子の表を `dense_node_feature` か `sparse_node_feature` で書く
3. ラベルを `write_task` に渡し、必要なら `rows` / `groups` / `sample_features` を添える
4. `finish()` を呼ぶ

の4手順を自分の入力に合わせて書けば、`pathwaygnn` の全コマンド（`pretrain` / `cv` /
`finetune` / `benchmark` / `ig`）がそのまま動きます。**エンジン側の変更は不要**です。
`configs/sample/` を `configs/<自分のデータ>/` にコピーし、`dataset.yaml` の
`name` / `dir` を書き換えてください。
