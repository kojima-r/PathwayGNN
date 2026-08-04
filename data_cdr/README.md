# data_cdr — がん細胞株の薬剤感受性（GDSC / CDRscan）

がん細胞株と化合物の組に対して薬剤感受性（GDSC の `LN_IC50`）を予測します。
由来は `GraphCDRScan`（CDRscan の Reactome グラフ版）。

- 前処理・学習の詳細: [`../README_data_cdr.md`](../README_data_cdr.md)
- 実行結果: [`../docs/cdr_report.md`](../docs/cdr_report.md)

**`data_cdr/raw/` と `data_cdr/processed/` は Git 管理外です**（合計 11 GB、かつ
COSMIC Cancer Gene Census が登録済みユーザーの手動ダウンロードを要求するため）。
新規クローンでは §4 の手順で作る必要があります。

---

## 1. ディレクトリ構成

```text
data_cdr/
├── raw/                      1.6 GB — 上流ステージの入力
│   ├── sources/              取得した公開ファイルの原本（監査用に保持）
│   ├── CosmicCLP_MutantExport.tsv    312 MB
│   ├── v17_fitted_dose_response.csv   41 MB
│   ├── Screened_Compounds.csv         42 KB
│   ├── used_cell_lines.csv           6.5 KB
│   ├── used_compounds.csv            5.3 KB
│   ├── cancer_gene_census.csv        5.3 KB
│   ├── reactome_rev2.graph.tsv       7.0 MB
│   ├── EnsemblToHGNC.tsv             1.5 MB
│   ├── hg38.2bit                     835 MB
│   ├── fingerprints.csv              1.4 MB（生成物）
│   ├── maf.csv                       130 MB（生成物・キャッシュ）
│   └── SHA256SUMS
├── processed/full_features/  9.3 GB — 上流ステージの出力＝cdr-prepare の入力
│   ├── graph.tsv                     7.7 MB
│   ├── vertices_dic.tsv / relationships_dic.tsv
│   ├── node_features.tsv             9.1 GB
│   ├── sample_features.tsv           729 MB
│   └── labels.tsv                    3.6 MB
└── prepared/                 2.7 GB — 生成物。pathwaygnn が読む汎用形式
```

パイプラインは3段階です。

```text
公開データ ──①──▶ raw/ ──②──▶ processed/full_features ──③──▶ prepared/
        download_raw_data   prepare_data              cdr-prepare
       （scripts/cdr/upstream/、要 .[cdr-upstream]）  （numpy のみ）
```

`data_cdr/processed/full_features` を他所から持ち込める場合、①②は不要です
（③以降は `gnn` 環境の numpy だけで動くよう意図的に書かれています）。

---

## 2. 元データの入手元

CDRscan 論文（2018）が使った **COSMIC/CCLP v82 + GDSC 6.0** と、当時 Google Drive に
置かれていた Reactome グラフはいずれも**現在入手できません**。
`scripts/cdr/upstream/download_raw_data.py` が現行の公開ソースを取得し、
レガシーなファイル名・スキーマへ変換します（`raw/sources/` に原本を残します）。

### 2.1 自動ダウンロードされるもの（認証不要）

| `raw/sources/` の原本 | URL | 由来と用途 |
| --- | --- | --- |
| `mutations_all_latest.csv.gz` (295 MB) | <https://cog.sanger.ac.uk/cmp/download/mutations_all_latest.csv.gz> | Cell Model Passports の細胞株変異一覧 → `CosmicCLP_MutantExport.tsv` |
| `model_list_latest.csv.gz` | <https://cog.sanger.ac.uk/cmp/download/model_list_latest.csv.gz> | Cell Model Passports のモデル一覧。Sanger モデル ID → COSMIC ID の逆写像に使う |
| `screened_compounds_rel_8.5.csv` | <https://cmp.cog.sanger.ac.uk/download/screened_compounds_rel_8.5.csv> | GDSC Compound Annotation release 8.5 → `Screened_Compounds.csv` |
| `GDSC1_fitted_dose_response_27Oct23.xlsx` (29 MB) | <https://cmp.cog.sanger.ac.uk/download/GDSC1_fitted_dose_response_27Oct23.xlsx> | GDSC1 用量反応（2023年10月版）→ `v17_fitted_dose_response.csv`（**名前だけレガシー踏襲**） |
| `hgnc_complete_set.txt` (16 MB) | <https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt> | HGNC 公開バケットの現行 complete set → `EnsemblToHGNC.tsv` |
| `FIsInGene_04142025_with_annotations.txt.zip` | <https://reactome.org/download/tools/ReactomeFIs/FIsInGene_04142025_with_annotations.txt.zip> | Reactome Functional Interactions 2025 → `reactome_rev2.graph.tsv`（消滅した Google Drive 成果物の代替） |
| `CDRscan_supplementary_information.pdf` (5.3 MB) | <https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-018-27214-6/MediaObjects/41598_2018_27214_MOESM1_ESM.pdf> | CDRscan 論文の CC BY 補足資料。Table S2 / S3 から `used_cell_lines.csv`（787株）と `used_compounds.csv`（229化合物）を抽出 |
| `hg38.2bit` (835 MB) | <https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit> | UCSC のゲノム配列。変異シグネチャ算出時の参照塩基取得に使う（`raw/` 直下に置かれる） |

出典（引用すべき論文・リソース）:

- **Cell Model Passports** — van der Meer et al., *Nucleic Acids Research* 47:D923–D929 (2019).
  <https://cellmodelpassports.sanger.ac.uk/>
- **GDSC** — Yang et al., *Nucleic Acids Research* 41:D955–D961 (2013).
  <https://www.cancerrxgene.org/>
- **Reactome FI** — <https://reactome.org/tools/reactome-fiviz>
- **CDRscan** — Chang et al., *Scientific Reports* 8:8857 (2018).
  <https://doi.org/10.1038/s41598-018-27214-6>
- **HGNC** — <https://www.genenames.org/download/>
- **UCSC Genome Browser** — <https://hgdownload.soe.ucsc.edu/downloads.html>

### 2.2 手動ダウンロードが必要なもの（**要ログイン**）

| `raw/` のファイル | 入手元 |
| --- | --- |
| `cancer_gene_census.csv` | COSMIC Cancer Gene Census **v104（GRCh38）**。<https://cancer.sanger.ac.uk/cosmic/download> でアカウント登録のうえ `Cosmic_CancerGeneCensus_Tsv_v104_GRCh38.tar` を取得 |

取得したアーカイブは次のように渡します（`--no-download` で再ダウンロードを抑止）。

```bash
python -m scripts.cdr.upstream.download_raw_data --no-download \
  --cosmic-cgc ./Cosmic_CancerGeneCensus_Tsv_v104_GRCh38.tar
```

CGC からは `GENE_SYMBOL` 列しか読まず、v104 の GRCh37 版と GRCh38 版は同一の 768 シンボルを
同一順序で持つため、どちらの archive からでも `cancer_gene_census.csv` はバイト単位で同じになります。

- 出典: Sondka et al., *Nature Reviews Cancer* 18:696–705 (2018).

### 2.3 生成物（`raw/` に置かれるが取得物ではない）

| ファイル | 生成方法 |
| --- | --- |
| `fingerprints.csv` | `scripts/cdr/upstream/create_fingerprints.py`。PubChem CID から SMILES を取得（PUG REST: <https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/ConnectivitySMILES/TXT>）し、RDKit で 3種 × 1024 ビットを計算。原著は PaDEL-Descriptor GUI を使っていましたが、保守停止・Java 専用・API なしのため **RDKit で置き換え**ています（Morgan/ECFP4 radius 2、RDKit topological、hashed atom pair）。**PaDEL の再実装ではないので、フィンガープリント由来の結果は論文と直接比較できません** |
| `maf.csv` | 変異シグネチャ計算用の MAF キャッシュ。存在すれば必ず再利用されます。**変異ソースを変えたら `prepare_data.py` 再実行前に必ず削除**してください（indel のアレルがキャッシュ作成時のゲノムから読まれているため） |

### 2.4 ゲノムアセンブリは hg38

変異座標の出所が Cell Model Passports（**GRCh38**）に変わったため、参照塩基の取得も
hg38 でなければなりません（`configs/cdr/upstream.json` の `HG2BIT: raw/hg38.2bit`）。
検証結果は、サンプリングした SNV 20,000 件が hg38 と 100% 一致、hg19 では 24.5%
（＝偶然一致の水準）で、約 17k の座標が hg19 の染色体長を超えます。

> 検証に使った `raw/hg19.2bit`（<https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.2bit>、
> 816 MB）は削除済みです。パイプラインは読まないので再取得は不要で、
> `download_raw_data.py` も hg38 しか取得しません。

---

## 3. `processed/full_features/`（上流バンドル）

すべてヘッダなし TSV で、**これが `cdr-prepare` の入力**です。

| ファイル | 形式 | 規模 |
| --- | --- | --- |
| `vertices_dic.tsv` | 名前<TAB>ID。名前は **HGNC の数値 ID** | 13,606 ノード |
| `relationships_dic.tsv` | 名前<TAB>ID。Reactome FI のアノテーション文字列の組み合わせ | 356 関係タイプ |
| `graph.tsv` | 3列（起点 ID / 関係 ID / 終点 ID）。対称化・自己ループ除去済み | 536,274 行 |
| `node_features.tsv` | 23列。**1行 = 1変異**（サンプル ID / ノード ID / 変異タイプ one-hot / 座標エンコード） | 27,309,391 行（9.1 GB） |
| `sample_features.tsv` | 3,350列（サンプル ID / がん種コード / 変異スペクトル 257 / 原発部位 one-hot 19 / フィンガープリント 3,072） | 107,418 行 |
| `labels.tsv` | 3列（サンプル ID / `LN_IC50` / `IC50`） | 107,418 行 |

列レイアウトの詳細は [`../README_data_cdr.md` §5](../README_data_cdr.md) を参照してください。

---

## 4. 構築手順

```bash
conda activate gnn
pip install -e '.[cdr-upstream]'     # ①② 専用の依存（pandas / rdkit / twobitreader / signatureanalyzer）
# さらに pdftotext (Poppler) と LibreOffice が PATH に必要

# ① 公開データの取得と互換変換 -> data_cdr/raw
python -m scripts.cdr.upstream.download_raw_data
python -m scripts.cdr.upstream.download_raw_data --no-download \
  --cosmic-cgc ./Cosmic_CancerGeneCensus_Tsv_v104_GRCh38.tar   # COSMIC CGC のみ手動（§2.2）

# ② 特徴量生成 -> data_cdr/processed/full_features
python -m scripts.cdr.upstream.prepare_data --config configs/cdr/upstream.json

# ③ 汎用形式へ変換 -> data_cdr/prepared
bash scripts/cdr/prepare.sh

# ④ 学習・評価・レポート
bash scripts/cdr/reproduce.sh
```

所要時間の目安: ①は回線次第（約 1.2 GB のダウンロード、`hg38.2bit` が支配的）、
②は変異シグネチャの算出が重く数時間、③は約4分、④は 3 GPU で約1時間。

①は最後に `raw/SHA256SUMS` を書き出します。検証はこちら:

```bash
(cd data_cdr/raw && sha256sum -c SHA256SUMS)
```

> `scripts/cdr/upstream/` は GraphCDRScan の `scripts/` ツリーの写しです。
> **変更点はデータ根のみ**（`data/` → `data_cdr/`、`prepare_data.py` の `--data-root`）。
> 上流バンドルをそのまま再現するために存在するので「モダン化」しないでください。

---

## 5. 生成される汎用形式

| 項目 | 値 |
| --- | --- |
| ノード数 | 13,606 |
| 関係タイプ数 | 356 |
| 有向エッジ数 | 536,274 |
| サンプル数 | 107,418（760 細胞株 × 168 化合物の一部） |
| node-level feature | `mutation`（sparse、760 行 / 117,880 値。同一プロファイルを重複排除し `rows/mutation.npy` で写像） |
| sample-level features | 3,348 次元（スペクトル 257 + 部位 one-hot 19 + フィンガープリント 3,072） |
| groups | 原発部位 19 種 |
| tasks | `sensitive_drugwise`（化合物ごとの `LN_IC50` 中央値で二値化）、`sensitive_global`（全体中央値） |

---

## 6. 注意点

- **これは 2018 年の CDRscan 実験ではありません。** GDSC1（2023年10月）、GRCh38 上の
  Cell Model Passports 変異、RDKit フィンガープリントで代替しているため、
  公開されている CDRscan の数値とは比較できません。
- **関係タイプ 356 種が計算コストを支配します。** `configs/cdr/cv.yaml` の
  `end_to_end: false` を維持してください（詳細は
  [`../README_data_cdr.md` §8](../README_data_cdr.md)）。
- ライセンスはソースごとに異なります。COSMIC は学術利用でも登録が必要、
  GDSC / Cell Model Passports / Reactome / UCSC / HGNC はそれぞれの利用条件に従ってください。
