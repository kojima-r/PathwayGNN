# data_tr — 創薬標的の再配置（target repositioning）

「ある遺伝子に摂動（ノックダウン／過剰発現）を与えたときの発現変化」と「ある疾患の
発現シグネチャ」を突き合わせ、その遺伝子がその疾患の **阻害性標的** あるいは
**活性化性標的** かを二値分類するデータセットです。由来は `SLGCN-TR`。

- 前処理・学習の詳細: [`../README_data_tr.md`](../README_data_tr.md)
- 実行結果: [`../docs/tr_report.md`](../docs/tr_report.md)

**3つのデータセットのうち、生データが Git に同梱されているのは `data_tr` だけです**
（`raw/` 合計 174 MB、`SHA256SUMS` つき）。クローン直後にダウンロード不要で実行できます。

---

## 1. ディレクトリ構成

```text
data_tr/
├── raw/                      174 MB — Git 管理対象
│   ├── PathwayCommons12.All.hgnc.sif        58 MB
│   ├── disease_specific_signature.tsv      9.0 MB
│   ├── knockdown_signature_sample.tsv       46 MB
│   ├── overexpression_signature_sample.tsv  33 MB
│   ├── inhibitory_target_disease.tsv       127 KB
│   ├── activatory_target_disease.tsv       8.0 KB
│   ├── Target_repositioning_データ完全版説明.txt
│   ├── SHA256SUMS
│   └── 推論用データ/            未使用（前処理は読まない）
│       ├── pred_data_inhibit.tsv
│       └── pred_data_activate.tsv
└── prepared/                 170 MB — 生成物（.gitignore 対象）
```

---

## 2. 生データの内訳と入手元

| ファイル | 内容 | 規模 | 入手元 |
| --- | --- | --- | --- |
| `PathwayCommons12.All.hgnc.sif` | パスウェイグラフ（3列 SIF、関係13種） | 1,884,849 行 | Pathway Commons v12（§2.1） |
| `disease_specific_signature.tsv` | 疾患特異的発現シグネチャ（`do_id` / `human_gene_name` / `expression`） | 251,041 行・235 疾患 | **出所未記録**（§2.2） |
| `knockdown_signature_sample.tsv` | ノックダウン摂動シグネチャ（行 = 摂動遺伝子、列 = L1000 遺伝子 978 個） | 4,345 行 × 979 列 | **出所未記録**（§2.2） |
| `overexpression_signature_sample.tsv` | 過剰発現摂動シグネチャ（同上） | 3,114 行 × 979 列 | **出所未記録**（§2.2） |
| `inhibitory_target_disease.tsv` | 正解ラベル（`gene` / `doid` / `label`）。タスク `kd_inh` 用 | 7,168 行 | **出所未記録**（§2.2） |
| `activatory_target_disease.tsv` | 正解ラベル。タスク `oe_act` 用 | 450 行 | **出所未記録**（§2.2） |

各ファイルの列レイアウトと値の例は [`../README_data_tr.md` §2](../README_data_tr.md) にあります。

### 2.1 `PathwayCommons12.All.hgnc.sif`

Pathway Commons Release 12 の「全ソース統合・HGNC シンボル・3列 SIF」ファイルです。
配布物は gzip 圧縮されているので、展開してこの名前で置きます。

```bash
curl -O https://download.baderlab.org/PathwayCommons/PC2/v12/PathwayCommons12.All.hgnc.sif.gz
gunzip PathwayCommons12.All.hgnc.sif.gz
```

- アーカイブ一覧: <https://download.baderlab.org/PathwayCommons/PC2/v12/>
  （`https://www.pathwaycommons.org/archives/PC2/` はここへリダイレクトされます）
- プロジェクト: <https://www.pathwaycommons.org/>
- ライセンス: 統合元のデータソースごとに条件が異なります。配布ディレクトリの
  `datasources.txt` / `LICENSE` を確認してください。

> ミラーに現存するのは v2〜v12 と v14 で、**v13 は取得できません**
> （`data_cancer/PathwayCommons13.All.hgnc.txt` はその名残です。
> [`../data_cancer/README.md`](../data_cancer/README.md) 参照）。

### 2.2 出所が記録されていないファイル（シグネチャとラベル）

`raw/Target_repositioning_データ完全版説明.txt` は各ファイルの **意味** を説明していますが、
**取得元 URL・バージョン・抽出条件は記録されていません**。共同研究者から受領した
`SLGCN-TR/data/raw` をそのまま持ち込んだためです。将来再取得が必要になったときの
手掛かりとして、データの形から読み取れることを記しておきます
（**以下は推定であり、検証されていません**）。

- 摂動シグネチャの列数 978 は **LINCS L1000 のランドマーク遺伝子数**と一致します。
  L1000 の一次データは GEO（`GSE92742` / `GSE70138`）および <https://clue.io/> で
  公開されています。行が「摂動を与えた遺伝子」なので、遺伝子ごとに集約された
  コンセンサスシグネチャに相当すると考えられます。
- 疾患 ID は **Disease Ontology**（`DOID:*`、<https://disease-ontology.org/>）です。
- ラベルは既知の治療標的－疾患関係を陽性（`1`）、それ以外を陰性（`0`）としたもので、
  陽性率は前処理後で `kd_inh` 8.2%（567 / 6,944）、`oe_act` 8.2%（37 / 450）です。

**再取得はこのリポジトリの前処理には不要です。** `raw/` は Git 管理下にあり
`SHA256SUMS` でハッシュが固定されているので、前処理の再現性は保たれます。

### 2.3 `推論用データ/`

学習・評価には使いません。`prepare_tr_dataset` は読み込まないため、受領したバンドルの
一部としてそのまま保持しているだけです。

---

## 3. 整合性の検証

```bash
(cd data_tr/raw && sha256sum -c SHA256SUMS)
```

---

## 4. 前処理

```bash
conda activate gnn
bash scripts/tr/prepare.sh    # = pathwaygnn-data tr-prepare --config configs/tr/prepare.yaml
```

`raw/` → `prepared/`（汎用形式: `dataset.json`, `graph.pt`, `nodes.json`,
`relations.json`, `channels/`, `tasks/`。定義は `src/pathwaygnn/data/format.py`）。
`prepared/` は `raw/` から完全に再生成できるため `.gitignore` 対象です。入力統計と
除外行数は `prepared/dataset.json` / `prepared/tasks/*/task.json` に記録されます。

生成結果の要約: ノード 30,918 / 関係タイプ 13 / 有向エッジ 3,673,654、
channel は `disease`（235 行）・`perturbation_kd`（4,345 行）・`perturbation_oe`（3,114 行）、
task は `kd_inh`（6,944 サンプル・陽性 567）と `oe_act`（450 サンプル・陽性 37）。
処理内容の詳細は [`../README_data_tr.md` §3](../README_data_tr.md) を参照してください。
