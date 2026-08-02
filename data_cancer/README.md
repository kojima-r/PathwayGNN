# data_cancer — TCGA がん予後予測（Inoue et al.）

TCGA の遺伝子発現プロファイルから、診断後 n 年（n = 1〜5）時点の生存を二値予測します。
由来は `SampleLevelGNN` / `DistributedGNN`、再現対象は `docs/papers/Arxiv_INOUE_0629.pdf`。

- 前処理・学習の詳細: [`../README_data_cancer.md`](../README_data_cancer.md)
- 再現結果: [`../docs/cancer_reproduction.md`](../docs/cancer_reproduction.md)

**`data_cancer/` は Git 管理外です**（`.gitignore` 対象は `prepared/`、
それ以外の実データもサイズの都合でコミットされていません）。

---

## 1. ディレクトリ構成

```text
data_cancer/
├── rawdata_TCGA/             2.6 GB — 生データ（§2）
│   ├── counts_gene.tsv               2.7 GB  recount2 の TCGA 遺伝子カウント
│   ├── TCGA_ID.tsv                    50 MB  recount2 のサンプルメタデータ
│   ├── mmc1.csv                      3.1 MB  TCGA-CDR 臨床情報（Liu et al. 2018）
│   ├── ensembl_gene_ids.txt          ← 別途用意が必要（§3.1）※未同梱
│   ├── msigdb.gmt / LM22.txt         ← 別途用意が必要（§3.2）※未同梱
│   └── ._*                           macOS の AppleDouble。無視してよい
├── PathwayCommons12.All.hgnc.sif      58 MB  パスウェイグラフ（§2.3）
├── processed/                4.0 GB — ①の出力＝②の入力（論文コードの公開バンドル）
└── prepared/                 637 MB — ②の出力（.gitignore 対象）
```

パイプラインは2段階です。

```text
rawdata_TCGA/ + PathwayCommons SIF
   │ ① pathwaygnn-data cancer-build-processed
   ▼
processed/            論文コードが公開したのと同じレイアウトの中間バンドル
   │ ② pathwaygnn-data cancer-prepare
   ▼
prepared/             pathwaygnn が読む汎用形式
```

**`processed/` が既にある場合、①は不要です**（②から始めてください）。
①を回すときだけ §3 の2入力が追加で必要になります。

---

## 2. 生データの入手元

| ファイル | 内容 | 入手元 | 登録 |
| --- | --- | --- | --- |
| `counts_gene.tsv` | recount2 が TCGA の RNA-Seq から推定した遺伝子カウント行列。58,037 遺伝子 × 11,285 サンプル。**行 ID 列を持たない** | recount2（§2.1） | 不要 |
| `TCGA_ID.tsv` | recount2 の TCGA サンプルメタデータ（`gdc_file_id`, `gdc_cases.submitter_id`, `gdc_cases.project.project_id` を使用） | recount2（§2.1） | 不要 |
| `mmc1.csv` | TCGA Pan-Cancer Clinical Data Resource（TCGA-CDR）。11,160 症例の生存情報 | Liu et al. 2018（§2.2） | 不要 |
| `PathwayCommons12.All.hgnc.sif` | パスウェイグラフ（3列 SIF、関係13種）。1,884,849 行 | Pathway Commons v12（§2.3） | 不要 |
| HGNC 対応表 | シンボル → 数値 HGNC ID。既定では `data_cdr/raw/EnsemblToHGNC.tsv` | HGNC（§2.4） | 不要 |
| `ensembl_gene_ids.txt` | `counts_gene.tsv` の 58,037 行に対応する遺伝子 ID の並び | recount2 の遺伝子アノテーション（§3.1） | 不要 |
| `msigdb.gmt`, `LM22.txt` | がん関連遺伝子の母集合 | MSigDB / CIBERSORT（§3.2） | **必要** |

### 2.1 recount2（`counts_gene.tsv`, `TCGA_ID.tsv`）

recount2 は TCGA を含む公開 RNA-Seq を統一パイプライン（Rail-RNA、アノテーションは
Gencode v25）で再定量したリソースです。

- 入口: <https://jhubiostatistics.shinyapps.io/recount/>
- 論文: Collado-Torres et al., *Nature Biotechnology* 35, 319–321 (2017).
  <https://doi.org/10.1038/nbt.3838>
- 直接ダウンロード（`recount` パッケージの `download_study()` が使う URL 形式）:

```bash
curl -O http://duffel.rail.bio/recount/v2/TCGA/counts_gene.tsv.gz   # -> counts_gene.tsv
curl -O http://duffel.rail.bio/recount/v2/TCGA/TCGA.tsv             # -> TCGA_ID.tsv にリネーム
```

- Bioconductor 経由（推奨。ミラーの生死に左右されません）:

```r
BiocManager::install("recount")
recount::download_study("TCGA", type = "counts-gene")
```

> `TCGA_ID.tsv` は recount2 のメタデータ（元名 `TCGA.tsv`）をローカルでリネームした
> ものです。UTF-8 として不正なバイトを含むため、前処理は `latin-1` で読みます。

### 2.2 TCGA-CDR（`mmc1.csv`）

Liu et al., "An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive
High-Quality Survival Outcome Analytics", *Cell* 173(2):400–416 (2018).
<https://doi.org/10.1016/j.cell.2018.02.052>

`mmc1.csv` は同論文の Supplemental Table S1（`mmc1.xlsx`）を CSV 化したものです。
同じ表は GDC PanCanAtlas からも配布されています。

```bash
curl -L -o TCGA-CDR-SupplementalTableS1.xlsx \
  https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81
# 1枚目のシート（TCGA-CDR）を CSV として書き出し mmc1.csv とする
```

- 一覧ページ: <https://gdc.cancer.gov/about-data/publications/pancanatlas>

### 2.3 グラフ: `PathwayCommons12.All.hgnc.sif`

Pathway Commons Release 12 の「全ソース統合・HGNC シンボル・3列 SIF」（1,884,849 行、
関係13種）。`configs/cancer/build_processed.yaml` の `graph_sif` がこれを指します。

```bash
curl -o data_cancer/PathwayCommons12.All.hgnc.sif.gz \
  https://download.baderlab.org/PathwayCommons/PC2/v12/PathwayCommons12.All.hgnc.sif.gz
gunzip data_cancer/PathwayCommons12.All.hgnc.sif.gz
```

- アーカイブ一覧: <https://download.baderlab.org/PathwayCommons/PC2/v12/>
- 列レイアウトの詳細: [`../data_tr/README.md` §2.1](../data_tr/README.md)

**`data_tr/raw/PathwayCommons12.All.hgnc.sif` と同一内容です**（SHA-256 一致:
`b6cbb006…2e6bc1`）。そちらは Git 管理下にあるので、この複製が無い環境では
`graph_sif` をそちらに向ければ再ダウンロード不要です。

論文はグラフを「Pathway Commons（関係13種）」と記述しており、公開バンドルのグラフは
**PathwayCommons 12 と一致することを確認済み**です（ノード 30,918 / 関係 13 /
有向エッジ 3,673,654 の3点が一致）。

> 以前ここに置かれていた `PathwayCommons13.All.hgnc.txt`（v13 の7列拡張 SIF、963 MB）は
> **削除し、実際に使う v12 の SIF に置き換えました**。参考素材でパイプラインは読んで
> いませんでした。なお **v13 は現在のミラーから消えており再取得できません**
> （<https://download.baderlab.org/PathwayCommons/PC2/> にあるのは v2〜v12 と v14）。
> 新しい版が必要なら v14 が後継ですが、ノード・関係・エッジ数が変わるため
> 事前学習からやり直しになります。

### 2.4 HGNC 対応表

既定では `data_cdr/raw/EnsemblToHGNC.tsv`（`data_cdr` の取得スクリプトが生成）を
使いますが、`Approved symbol` と `Ensembl gene ID` を持つ HGNC の complete-set
エクスポートなら何でも構いません。

```bash
curl -O https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt
```

- HGNC ダウンロード: <https://www.genenames.org/download/>

---

## 3. 同梱できない2つの入力（①を回すときだけ必要）

### 3.1 `ensembl_gene_ids.txt` — 発現行列の行 ID

`counts_gene.tsv` は行名を持たないため、**58,037 行の並び順に対応する遺伝子 ID の一覧**が
別途必要です。この並び順は recount2 の遺伝子アノテーション（Gencode v25）が定義しています。

```r
library(recount)
data(recount_genes)                     # 58,037 遺伝子、counts_gene.tsv の行順
write.table(data.frame(id = names(recount_genes), bp_length = recount_genes$bp_length),
            "ensembl_gene_ids.txt", sep = "\t", quote = FALSE,
            row.names = FALSE, col.names = FALSE)
```

形式は1行1 ID。第2列に `bp_length` を置くと `transform: log1p_tpm` が使えます。

```text
ENSG00000000003.14	4536
ENSG00000000005.6	1476
```

Ensembl ID のほか、遺伝子シンボルや数値 HGNC ID でも受け付けます。
MyGene.info 経由の対応表を別に作って `ensembl_to_hgnc:` に渡すこともできます
（応答をキャッシュし、入力リストの SHA-256 が一致しないキャッシュは再利用を拒否するため、
ID 変換は監査可能かつオフラインで再現できます）。

```bash
pathwaygnn-data cancer-map-ids --config configs/cancer/id_mapping.yaml
```

### 3.2 `gene_sets` — がん関連遺伝子の母集合（**登録が必要**）

論文 2.2 節: 「MSigDB と LM22 免疫遺伝子シグネチャに掲載された遺伝子を選択し、
ナレッジグラフのノードに存在しない遺伝子を除外した。4,448 遺伝子の発現量を使用した」。

| データ | 出典 | 入手元 |
| --- | --- | --- |
| **MSigDB**（`.gmt`） | Liberzon et al., *Cell Systems* 1(6):417–425 (2015). <https://doi.org/10.1016/j.cels.2015.12.004> | <https://www.gsea-msigdb.org/gsea/msigdb/> — 無料だがアカウント登録が必要 |
| **LM22**（TSV） | Newman et al., *Nature Methods* 12:453–457 (2015). <https://doi.org/10.1038/nmeth.3337> | CIBERSORT/CIBERSORTx <https://cibersortx.stanford.edu/> — 登録が必要。同論文の Supplementary Table 1 にも同じ 22 細胞種 × 547 遺伝子の行列が掲載 |

`.gmt` は3列目以降を、それ以外は1列目を遺伝子シンボルとして読みます。複数指定すると和集合です。
最終的な選択は **(gene_sets の和集合) ∩ (グラフのノード)** で、論文の 4,448 遺伝子に対応します。

---

## 4. `processed/`（中間バンドル）

論文コードが公開したのと同じレイアウトです。**これが `cancer-prepare` の入力**で、
`data_cancer/processed/` が手元にあるなら §2・§3 のダウンロードは一切不要です。

```text
processed/
├── graph.tsv                   53 MB   3列（起点 ID / 関係 ID / 終点 ID）。行順をそのまま使う
├── vertices_dic.tsv            452 KB  名前<TAB>ID。名前は数値 HGNC ID（と CHEBI:*）
├── relationships_dic.tsv       315 B   名前<TAB>ID（13種）
├── <n>years_node_input.tsv     計 4.2 GB  3列ロング形式の発現プロファイル
├── <n>years_sample.tsv         35列（サンプル ID / がん種コード / がん種 one-hot 33）
└── <n>years_labels.tsv         2列（サンプル ID / ラベル。1 = 生存、0 = 死亡）
```

公開バンドルの `<n>years_sample.tsv` は**行 0 が `0,1,2,…,34` というシリアライズされた
ヘッダ**というレガシー仕様を持ちます。削除すると公開コードとサンプル対応がずれるため、
②はそのまま1サンプルとして保持します。**「バグに見えるから直す」ことはしないでください。**

①で作り直したバンドルは内部的には整合していますが、整数エンコードがソート順になるため
**公開バンドルとは互換ではありません**（片方で事前学習した encoder はもう片方では使えず、
`load_encoder` はノード数・関係数しか照合しないので黙って通ってしまいます）。
作り直したら `outputs/cancer/` を消してください。詳細は
[`../README_data_cancer.md` §5](../README_data_cancer.md)。

---

## 5. 実行

```bash
conda activate gnn
bash scripts/cancer/reproduce_paper.sh build-processed   # ①（生データから作り直す場合のみ）
bash scripts/cancer/reproduce_paper.sh prepare           # ②
```

②は3列ロング形式の TSV（合計 4.2 GB、約1分半）を `prepared/channels/expression_<n>year/` の
memmap 行列へ流し込み、年ごとに `prepared/tasks/<n>year/` を作り、年別サンプル数を
論文の Supplementary Table 1 と照合します。既存行列の形状が一致していれば変換をスキップします。

生成結果: ノード 30,918 / 関係 13 / 有向エッジ 3,673,654、dense channel
`expression_1year`〜`expression_5year`（各 4,448 遺伝子）、task `1year`〜`5year`
（9,484 / 7,308 / 5,915 / 5,036 / 4,492 サンプル）。共変量とグループはがん種（33種）。

①で作り直したバンドルを使う場合は、年別サンプル数が数件ずれるため
`configs/cancer/prepare.yaml` の `strict_sample_counts` を `false` にしてください。

> リファクタ前のレイアウトだった `artifacts/` は削除済みです（`prepared/` が後継）。
