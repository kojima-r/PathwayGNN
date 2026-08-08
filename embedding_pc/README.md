# PathwayCommons 参加ノードの外部アノテーション作成スクリプト

`raw/PathwayCommons13.All.hgnc.txt` (SIF 形式) の PARTICIPANT_A / PARTICIPANT_B に現れる
ノードに対して、

- ChEBI ID → SMILES / InChI / InChIKey
- gene symbol → UniProt ID / アミノ酸配列
- gene symbol → HGNC ID / Entrez ID / Ensembl ID

の対応表を作成する。

## 実行方法

```bash
# ChEBI ID -> SMILES
python scripts/build_chebi_smiles.py

# gene symbol -> UniProt ID / 配列 (1 遺伝子 1 エントリ)
python scripts/build_gene_uniprot_seq.py --one-per-gene

# gene symbol -> HGNC ID / Entrez ID / Ensembl ID
# (ノード名が gene symbol でないコーパスと突き合わせるための別名。下の「別名」節)
python scripts/build_gene_ids.py
```

外部ファイルは `raw/ext/` にキャッシュされ、2 回目以降は再ダウンロードしない
(`--force-download` で更新)。標準ライブラリのみで動作する。

## 出力

| ファイル | 内容 |
| --- | --- |
| `data/chebi_smiles.tsv` | `chebi_id, primary_chebi_id, name, smiles, inchi, inchikey` |
| `data/chebi_smiles.missing.txt` | 構造情報が取得できなかった ChEBI ID と名前 |
| `data/gene_uniprot_seq.tsv` | `gene_symbol, uniprot_id, entry_name, reviewed, match_type, protein_name, length, sequence` |
| `data/gene_uniprot_seq.missing.txt` | UniProt に対応が無い participant (大半は低分子名) |
| `data/gene_ids.tsv` | `gene_symbol, hgnc_id, entrez_id, ensembl_gene_id, match_type` |
| `data/gene_ids.missing.txt` | HGNC に対応が無い participant |

`--out` で出力先を変更できる (missing ファイルは同じ basename に `.missing.txt`)。

## 主なオプション

`build_chebi_smiles.py`

- `--all` : SIF に出現するものだけでなく ChEBI 全化合物 (約 21.8 万件) を出力
- `--sif PATH` / `--out PATH` / `--cache-dir PATH` / `--force-download`

`build_gene_uniprot_seq.py`

- `--one-per-gene` : gene symbol ごとに 1 件へ絞る (優先度: primary 一致 > reviewed > 配列長)。
  既定は候補をすべて出力する
- `--include-unreviewed` : TrEMBL も含める (取得量が大きく増える)
- `--organism 9606` : 対象生物の taxonomy ID
- `--fasta PATH` : 配列を FASTA でも出力
- `--sif PATH` / `--out PATH` / `--cache-dir PATH` / `--force-download`

`build_gene_ids.py`

- 承認シンボル一致 (`primary`) を優先し、無ければ旧シンボル (`prev`) / 別名 (`alias`)。
  同じ旧・別名が複数の遺伝子を指す場合はどちらにも寄せずに捨てる
- `--all` : SIF に出現するものだけでなく HGNC 全遺伝子を出力
- `--sif PATH` / `--out PATH` / `--cache-dir PATH` / `--force-download`

## データソース

- ChEBI flat files: <https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files/>
  (`compounds.tsv.gz`, `secondary_ids.tsv.gz`, `structures.tsv.gz`)
- UniProtKB REST API: <https://rest.uniprot.org/uniprotkb/search>
- HGNC complete set: <https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt>

## 実行結果 (2026-08-06 時点)

SIF のユニーク participant は 60,439 件 (ChEBI ID 11,348 / それ以外 49,091)。

- ChEBI: 11,326 件を出力、うち SMILES 有り **10,249 件 (90.5%)**。
  残り 1,099 件は ChEBI 側に構造が登録されていないエントリ
  (化合物クラス、ラセミ体、タンパク質など。ChEBI REST API で確認しても
  `default_structure` が null)。
- HGNC: **19,568 gene symbol** に ID を対応付け (承認シンボル 18,865 / 旧シンボル 697 / 別名 6)。
- UniProt: **19,360 gene symbol** に対応付け (primary 一致 18,713 / synonym 一致 647)。
  未対応 29,731 件のうち gene symbol 様の文字列は 322 件で、その大半は
  低分子の商品名・化合物コード (AZD4547 等)、偽遺伝子 (`*P`)、lncRNA (`*-AS1`) であり、
  UniProt に対応するタンパク質が存在しない。

---

# ノード埋め込みの作成

`data/*.tsv` のアミノ酸配列 / SMILES を事前学習モデルで埋め込み、
`{ノード名: 埋め込みベクトル}` の `processed/node_embeddings.npz` を作る。
ノード名は SIF の participant 表記に一致する (タンパク質 = gene symbol、化合物 = `CHEBI:xxxx`)。

conda 環境 `esm` で実行する。

## 実行方法

```bash
PY=~/miniconda3/envs/esm/bin/python

# 1. タンパク質 (ESMC-6B, 2560 次元) — GPU 3 枚に分散
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$i nohup $PY scripts/embed_proteins.py \
      --shard $i --num-shards 3 --resume > logs/embed_proteins.shard$i.log 2>&1 &
done; wait

# 2. 化合物 (Uni-Mol v1, 512 次元)
CUDA_VISIBLE_DEVICES=0 $PY scripts/embed_chemicals.py --batch-size 256 --workers 16

# 3. 統合 (data/gene_ids.tsv があれば別名も入る)
$PY scripts/build_node_embeddings.py
```

## 出力

| ファイル | 内容 |
| --- | --- |
| `processed/node_embeddings.npz` | 種別ごとの `(names, embeddings)` と別名 (203 MB)。**実運用はこちら** |
| `processed/node_embeddings.json` | `{ノード名: 埋め込みベクトル}` (460 MB)。別名は含まない |
| `processed/node_embeddings.meta.json` | 種別 (protein / chemical)・モデル・次元、および別名表 `alias_to_name` |
| `processed/protein_emb.shard*.npz` | 中間ファイル (gene symbol -> [2560]) |
| `processed/chemical_emb.npz` | 中間ファイル (ChEBI ID -> [512]) |
| `processed/chemical_emb.failed.txt` | 埋め込めなかった ChEBI ID と理由 |

タンパク質と化合物で次元が異なる (2560 / 512)。統一はしていないので、
下流の GNN ではノード種別ごとに入力射影を持たせる (heterogeneous GNN) 前提。
種別は `node_embeddings.meta.json` の `node_type` で引ける。

## 別名 (ノード名が gene symbol でないコーパス向け)

`data/gene_ids.tsv` があると、タンパク質の各行に **HGNC ID / Entrez ID / Ensembl ID** の
別名を付ける。ベクトルは複製せず「別名 -> 行番号」で持つので、ファイルはほとんど太らない
(npz は 203.0 -> 203.3 MB)。

npz では ID 体系ごとに配列を分ける (`protein_alias_hgnc_id_names` + `..._rows` など)。
**混ぜないのが要点**で、HGNC ID も Entrez ID も裸の数値なので、1つの名前空間にまとめると
`5` が HGNC:5 = A1BG と Entrez 5 = 別の遺伝子の両方を指してしまう
(実測で約 30% が曖昧になり捨てられた)。どの体系で突き合わせるかは、コーパスのノード名を
知っている下流が選ぶ。

| 体系 | 件数 | 使うコーパス |
| --- | --- | --- |
| `hgnc_id` | 19,305 | `data_cancer` (被覆 34.4% → 95.6%)、`data_cdr` (0% → 98.7%) |
| `entrez_id` | 19,304 | Entrez ID をノード名にするコーパス |
| `ensembl_gene_id` | 19,263 | Ensembl ID をノード名にするコーパス |

`--no-gene-ids` を付けると別名を作らない。

## pathwaygnn の encoder に入れる

`model.node_embeddings:` に `processed/node_embeddings.npz` を指定すると、
種別ごとのアダプタ (`nn.Linear(2560 or 512 → hidden_dim)`) を通して encoder の入力になる。
表に無いノードは従来どおり `nn.Embedding` のまま。

```yaml
# configs/tr/pretrain_pc_embedding.yaml — ノード名が gene symbol
model:
  node_embeddings:
    path: embedding_pc/processed/node_embeddings.npz

# configs/cancer/pretrain_pc_embedding.yaml — ノード名が数値 HGNC ID
model:
  node_embeddings:
    path: embedding_pc/processed/node_embeddings.npz
    aliases: [hgnc_id]
```

突き合わせは **ノード名** (`prepared/nodes.json`) なので、`data_tr` のように gene symbol を
ノード名に使うコーパスはそのまま当たり (30,895 中 28,735)、それ以外は上の `aliases` を指定する。
仕組みと実測値は [../README.md §5.1](../README.md)、設定項目は
[../README_config.md §3](../README_config.md) を参照。

## 埋め込みの作り方

`embed_proteins.py` — `esm_test.py` と同じ手順。
最終 Transformer layer の hidden state から special token を除き、**残基方向に mean pooling**
して 1 配列 1 本の `[2560]` ベクトルにする。bfloat16 で推論し、保存は float32。

4096 残基を超える配列 (78 件) は attention が O(L^2) で膨らむため 4096 残基の窓に分割し、
各窓の平均を**窓内残基数で重み付き平均**する。全残基の平均という意味は保たれ、
TTN (34,350 残基) まで打ち切らずに扱える。

`embed_chemicals.py` — `unimol_test.py` と同じ設定 (`unimolv1` / `remove_hs=False`) で
`cls_repr` を取り、1 分子 1 本の `[512]` ベクトルにする。
ただし SMILES をそのまま `UniMolRepr.get_repr()` に渡すと 2 点で詰まるので前処理を挟む。

1. **不正 SMILES** — 推論時は `ValueError` を投げてバッチごと落ちる。
   ChEBI の SMILES には kekulize できないものが 781 件あるため、同じ行の InChI から
   RDKit 正準 SMILES を再生成して救済する。
2. **コンフォマー生成のハング** — `unimol_tools` 内部の `AllChem.EmbedMolecule` には
   時間制限が無く、一部の分子で事実上停止する (実測で 1 分子に 20 分以上)。
   `ETKDGv3.timeout` を設定して 3D 座標を自前で作り、`atoms` / `coordinates` を
   直接 `get_repr()` に渡す。手順は `unimol_tools` の `inner_smi2coords` の再現で、
   埋め込みは SMILES を直接渡した場合と bit 単位で一致することを確認済み。

## 実行結果 (2026-08-07 時点)

**29,596 ノード** (protein 19,360 / chemical 10,236) と、タンパク質の別名 3 体系。

- protein: `gene_uniprot_seq.tsv` の 19,360 件すべて。RTX PRO 6000 x3 で約 7 分。
- chemical: SMILES を持つ 10,249 件中 10,236 件 (99.9%)。約 3 分。
  3D コンフォマー成功 9,966 / 2D フォールバック 270。
  失敗 13 件はいずれも `*` (結合手・R 基) を含む一般化構造で、RDKit で扱えない。

健全性チェック (cosine 類似度):

| ペア | cosine |
| --- | --- |
| KRAS vs HRAS (パラログ) | 0.981 |
| KRAS vs NRAS (パラログ) | 0.965 |
| KRAS vs TTN (無関係) | 0.729 |
| protein ランダムペア平均 | 0.685 |
| ATP vs ADP | 0.888 |
| ATP vs H2O | 0.574 |
| chemical ランダムペア平均 | 0.713 |

JSON は小数点以下 5 桁に丸めてある (npz との最大差 5e-6)。

別名 (`hgnc_id` 19,305 / `entrez_id` 19,304 / `ensembl_gene_id` 19,263) は
同じ体系の中で 2 つの symbol が同じ ID を指す 17〜18 件だけを曖昧として捨てている。
