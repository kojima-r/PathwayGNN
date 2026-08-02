# data_tr — 創薬標的の再配置（target repositioning）

「ある遺伝子に摂動（ノックダウン／過剰発現）を与えたときの発現変化」と「ある疾患の
発現シグネチャ」を突き合わせ、その遺伝子がその疾患の **阻害性標的** あるいは
**活性化性標的** かを二値分類するデータセットです。摂動シグネチャの実体は
**LINCS L1000（GSE92742）Level 5** です。

- 前処理・学習の詳細: [`../README_data_tr.md`](../README_data_tr.md)
- 実行結果: [`../docs/tr_report.md`](../docs/tr_report.md)

**`data_tr/raw/` と `data_tr/processed/` は Git 管理外です**（`raw/` は 22 GB、
うち LINCS の Level 5 行列だけで 23.4 GB のうちの大半）。新規クローンでは §4 の手順で作ります。

---

## 1. ディレクトリ構成

```text
data_tr/
├── raw/                      22 GB — 公開ソース（§2）
│   ├── GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx  23.4 GB（展開後）
│   ├── GSE92742_Broad_LINCS_sig_info.txt                          105 MB
│   ├── GSE92742_Broad_LINCS_gene_info.txt                         622 KB
│   ├── disease_signatures-v1.0.json                                17 MB
│   ├── hgnc_complete_set.txt                                       17 MB
│   ├── PathwayCommons12.All.hgnc.sif                               58 MB
│   ├── kegg_disease_omim.list                                     220 KB
│   ├── HumanDO.obo                                                6.6 MB
│   ├── data_tr__target_disease.zip                                 30 KB
│   ├── inhibitory_target_disease.tsv                              127 KB（zip から展開）
│   ├── activatory_target_disease.tsv                              8.0 KB（同上）
│   └── SHA256SUMS
├── processed/                697 MB — 中間バンドル（§3）。tr-prepare の入力
│   ├── graph.tsv                      61 MB
│   ├── knockdown_signature.tsv       342 MB
│   ├── overexpression_signature.tsv  199 MB
│   ├── disease_specific_signature.tsv 7.2 MB
│   ├── inhibitory_target_disease.tsv 130 KB
│   ├── activatory_target_disease.tsv 8.2 KB
│   └── build_manifest.json
└── prepared/                 693 MB — 生成物。pathwaygnn が読む汎用形式
```

パイプラインは3段階で、`data_cancer` / `data_cdr` と同じ分割です。

```text
公開ソース ──①──▶ data_tr/raw ──②──▶ data_tr/processed ──③──▶ data_tr/prepared
      download_raw_data     tr-build-processed        tr-prepare
      （標準ライブラリのみ）  （要 h5py）               （numpy/torch のみ）
```

---

## 2. 生データの入手元（`raw/`）

すべて `python -m scripts.tr.upstream.download_raw_data` が自動取得します（認証不要）。

| ファイル | 内容 | 入手元 |
| --- | --- | --- |
| `GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx` | LINCS L1000 Level 5（MODZ 集約済み z スコア）。473,647 シグネチャ × 12,328 遺伝子の GCTX（HDF5） | GEO `GSE92742`（§2.1） |
| `GSE92742_Broad_LINCS_sig_info.txt` | 各シグネチャの実験条件（`sig_id`, `pert_iname`, `pert_type`, `cell_id`, 時間, 用量） | 同上 |
| `GSE92742_Broad_LINCS_gene_info.txt` | 遺伝子表。`pr_is_lm == 1` が **978 ランドマーク遺伝子** | 同上 |
| `disease_signatures-v1.0.json` | CREEDS の手動キュレーション疾患シグネチャ（up/down 遺伝子リスト、`do_id`, `organism`） | CREEDS（§2.2） |
| `hgnc_complete_set.txt` | 承認シンボル・旧シンボル・エイリアスの対応表 | HGNC（§2.3） |
| `PathwayCommons12.All.hgnc.sif` | パスウェイグラフ（3列 SIF、関係13種） | Pathway Commons v12（§2.4） |
| `kegg_disease_omim.list` | KEGG DISEASE → OMIM のリンク | KEGG REST（§2.5） |
| `HumanDO.obo` | Disease Ontology 本体。OMIM 相互参照を読む | Disease Ontology（§2.5） |
| `data_tr__target_disease.zip` → `{inhibitory,activatory}_target_disease.tsv` | 正解ラベル（標的遺伝子 × 疾患）。`gene`, `doid`, `label` | 九州工業大学（§2.6） |

### 2.1 LINCS L1000（GSE92742）

Subramanian et al., "A Next Generation Connectivity Map: L1000 Platform and the
First 1,000,000 Profiles", *Cell* 171(6):1437–1452 (2017).
<https://doi.org/10.1016/j.cell.2017.10.049>

- GEO: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742>
- 実際の取得先は **FTP ミラー**です。

```bash
BASE=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl
curl -O $BASE/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz   # 21.3 GB
curl -O $BASE/GSE92742_Broad_LINCS_sig_info.txt.gz
curl -O $BASE/GSE92742_Broad_LINCS_gene_info.txt.gz
gunzip GSE92742_Broad_LINCS_*.gz
```

> **GEO のダウンロード CGI（`/geo/download/?acc=...`）は使いません。** CGI は Range
> リクエストを無視するため、21 GB の転送が途中で切れると最初からやり直しになります。
> FTP ミラーは 206 を返すので、取得スクリプトはレジューム付きで再試行します。

使うのは `pert_type` が `trt_sh.cgs`（ノックダウン、36,720 プロファイル）と
`trt_oe`（過剰発現、22,205 プロファイル）、遺伝子は 978 ランドマークのみです。

### 2.2 CREEDS（疾患シグネチャ）

Wang et al., "Extraction and analysis of signatures from the Gene Expression
Omnibus by the crowd", *Nature Communications* 7:12846 (2016).
<https://doi.org/10.1038/ncomms12846>

```bash
curl -O https://maayanlab.cloud/CREEDS/download/disease_signatures-v1.0.json
```

- 入口: <https://maayanlab.cloud/CREEDS/>
- 使うのは **`do_id` が付いていて `organism == "human"`** のレコードだけです。

### 2.3 HGNC

```bash
curl -O https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt
```

- <https://www.genenames.org/download/>
- 参考実装は 2023-10-29 時点のカスタムエクスポートを使っていましたが、こちらは
  **現行の complete set** なので、シンボル変換は今日の命名法に従います。

### 2.4 Pathway Commons v12

```bash
curl -O https://download.baderlab.org/PathwayCommons/PC2/v12/PathwayCommons12.All.hgnc.sif.gz
gunzip PathwayCommons12.All.hgnc.sif.gz
```

`data_cancer/PathwayCommons12.All.hgnc.sif` と同一内容です（SHA-256 一致）。

### 2.5 KEGG DISEASE → OMIM → DOID

ラベルの疾患 ID を DOID に揃えるための2つの対応表です。参考実装が使っていた
ファイルはどちらも消滅しているため、**現行で入手できる等価物**に置き換えています。

| 参考実装 | 現状 | 置き換え先 |
| --- | --- | --- |
| LinkDB の `omim_disease.list` | 配布終了 | KEGG REST: <https://rest.genome.jp/link/omim/ds> |
| `src/DOreports/xrefs_in_DO.tsv` | リポジトリから削除 | `HumanDO.obo` の `xref: OMIM:*` を読む |

```bash
curl -o kegg_disease_omim.list https://rest.genome.jp/link/omim/ds
curl -O https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/HumanDO.obo
```

- KEGG: <https://www.genome.jp/kegg/disease/>（利用条件は KEGG のライセンスに従う）
- Disease Ontology: <https://disease-ontology.org/>（CC0）

### 2.6 正解ラベル（九州工業大学）

**元データは九州工業大学（山西研）が公開していたもの**です。

```text
http://labo.bio.kyutech.ac.jp/~yamani/target_repositioning/target_disease_data.zip
```

このURLは現在 **403 で配布が停止しており、Wayback Machine にもスナップショットが
ありません**。そこで本リポジトリのリリースに **DOID 変換済みのテーブルをミラー**して
あり、取得スクリプトはそちらを使います。

```bash
curl -LO https://github.com/kojima-r/PathwayGNN/releases/download/v1/data_tr__target_disease.zip
unzip data_tr__target_disease.zip -d data_tr/raw   # 2ファイル
```

| ファイル | 対応タスク | 行数（ヘッダ除く） |
| --- | --- | --- |
| `inhibitory_target_disease.tsv` | `kd_inh`（ノックダウン × 阻害性標的） | 7,168（陽性 568） |
| `activatory_target_disease.tsv` | `oe_act`（過剰発現 × 活性化性標的） | 450（陽性 37） |

ミラーしているのは **疾患 ID が DOID に変換済み**の版なので、②では遺伝子名を現行の
HGNC 変換表で付け直すだけです。オリジナルの **KEGG DISEASE ID 版**
（`*_target_disease.txt`）をお持ちの場合は `configs/tr/build_processed.yaml` の
`kegg_labels` を指定してください。§2.5 の KEGG→OMIM→DOID 変換経路が実際に走ります。
ただし参考実装が変換の穴を手作業で埋めた `gs_kegg_to_do_Iadd.csv` は公開されていないため、
自動変換だけでは対応が付かない疾患が残ります（`build_manifest.json` の
`kegg_ids_unmapped` に記録されます）。

---

## 3. 中間バンドル（`processed/`）

### 3.1 ②が行うこと

`pathwaygnn-data tr-build-processed`（実装は `src/pathwaygnn_datasets/tr/build.py`）は、
参考実装 `target-repositioning-share` の前処理ノートブック 01〜05 に対応します。

1. **シンボル変換表** — HGNC の承認シンボル・旧シンボル・エイリアスから
   「同義語 → 承認シンボル」の辞書を作る。承認シンボル自身は変換対象から外し、
   複数の遺伝子が同じ同義語を主張する場合はエイリアス側を優先、それでも一意に
   決まらないものは `SHARED:BBB/CCC` の形で曖昧さを残す（参考実装と同じ規則）。
2. **グラフ** — SIF の両端をこの辞書で変換して大文字化し、`graph.tsv` に書く。
3. **摂動シグネチャ** — GCTX からランドマーク 978 行だけを読み、`trt_sh.cgs` /
   `trt_oe` を `(pert_iname, cell_id)` ごとに平均する（＝複製・測定時間・用量を平均）。
4. **疾患シグネチャ** — CREEDS を (疾患, 遺伝子) の行に展開し、同一疾患の複数研究は
   平均する。
5. **ラベル** — §2.6 の通り。陽性ペアから遺伝子 × 疾患の全組み合わせを作り、
   陽性以外を陰性とする。

### 3.2 形式と実測値

②の所要時間は **52 秒**です（`raw/` が揃っている状態から）。

| ファイル | 形式 | 規模 |
| --- | --- | --- |
| `graph.tsv` | 3列（起点 / 関係 / 終点）、ヘッダなし | 1,884,849 行 / 30,895 ノード / 13 関係 |
| `knockdown_signature.tsv` | `pert_iname`, `cell_id`, 978 遺伝子列 | 33,817 行（36,720 プロファイルを集約） |
| `overexpression_signature.tsv` | 同上 | 20,131 行（22,205 プロファイル） |
| `disease_specific_signature.tsv` | `do_id`, `gene_name`, `expression` | 186,884 行 / 178 疾患（493 シグネチャ） |
| `inhibitory_target_disease.tsv` / `activatory_target_disease.tsv` | `gene`, `doid`, `label` | 7,168 行 / 450 行 |
| `build_manifest.json` | 件数と、下記2つの仕様フラグ | — |

`tr-prepare` を通した結果（③、所要 66 秒）は
[`../README_data_tr.md` §4](../README_data_tr.md) にまとめてあります
（`kd_inh` 61,101 サンプル、`oe_act` 3,465 サンプル）。

### 3.3 データの中身を決める2つのフラグ

`build_manifest.json` に記録されます。どちらもエンコードではなくデータの中身を変えます。

| フラグ | 既定 | 意味 |
| --- | --- | --- |
| `per_cell_line` | `true` | 摂動プロファイルの単位を `(pert_iname, cell_id)` にする。`false` なら細胞株を平均で潰し、行 = `pert_iname` になる |
| `human_only` | `true` | CREEDS の `organism == "human"` だけを使う |

**摂動プロファイルが細胞株ごとなので、ラベル1行は「その遺伝子が摂動された細胞株の数」
だけサンプルに展開されます**（`task.json` の `label_rows_used` と `num_samples` を
比較してください）。

なお参考実装は、先行研究が使っていた TT-WOPT による欠測補完を**採用していません**
（効果が確認できなかったため）。

---

## 4. 構築手順

```bash
conda activate gnn

# ① 公開ソース取得（標準ライブラリのみ、約21.5 GB。回線次第で1時間程度）
python -m scripts.tr.upstream.download_raw_data
#   途中で切れても同じコマンドで再開します（.part を Range で継続）

# ② 中間バンドル構築（GCTX の読み取りに h5py が必要）
pip install -e '.[tr-upstream]'
bash scripts/tr/build_processed.sh

# ③ 汎用形式へ変換
bash scripts/tr/prepare.sh
```

整合性の検証:

```bash
(cd data_tr/raw && sha256sum -c SHA256SUMS)
```

---

## 5. 学習の実行

```bash
pathwaygnn pretrain  --config configs/tr/pretrain.yaml      # 単一 GPU
bash scripts/tr/pretrain_distributed.sh                     # torchrun で分散
pathwaygnn cv        --config configs/tr/cv.yaml
pathwaygnn finetune  --config configs/tr/finetune_kd_inh.yaml
pathwaygnn benchmark --config configs/tr/benchmark_kd_inh.yaml
pathwaygnn ig        --config configs/tr/ig_kd_inh.yaml
pathwaygnn-data tr-report --config configs/tr/report.yaml
```

共変量が無いため `cv` のアブレーションは2条件（`mlp` / `gnn_mlp`）です。
