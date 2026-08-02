# README_data_tr — target repositioning（`data_tr`）

創薬標的の再配置（target repositioning）データセットです。「ある遺伝子に摂動を与えた際の
発現変化」と「ある疾患の発現シグネチャ」を突き合わせ、その遺伝子がその疾患の
**阻害性標的（inhibitory target）** あるいは **活性化性標的（activatory target）** かを
二値分類します。摂動シグネチャの実体は **LINCS L1000（GSE92742）Level 5**、
疾患シグネチャは **CREEDS**、ラベルは**九州工業大学**が公開していた標的－疾患対応表です。

- 取得元 URL の一覧: [`data_tr/README.md`](data_tr/README.md)
- 設定項目の説明: [README_config.md](README_config.md)
- 実行結果: [`docs/tr_report.md`](docs/tr_report.md) / [`docs/tr_report.html`](docs/tr_report.html)

---

## 0. 最初に読む: リポジトリをクローンした直後の状態

`data_tr/` は **`README.md` 以外すべて Git 管理外**です（`raw/` が 22 GB あるため）。
クローン直後は次の3段階で作ります。

```bash
conda activate gnn
pip install -e '.[tr-upstream]'                    # ② が使う h5py

python -m scripts.tr.upstream.download_raw_data    # ① 公開ソース -> data_tr/raw（約21.5 GB）
bash scripts/tr/build_processed.sh                 # ② -> data_tr/processed（約1分）
bash scripts/tr/prepare.sh                         # ③ -> data_tr/prepared（約1分）

pathwaygnn pretrain --config configs/tr/pretrain.yaml   # 以降は §5 参照
```

①だけが時間を要します（回線次第、実測で約50分）。②③は合わせて2分程度です。

---

## 1. パイプラインの3段階

`data_cancer` / `data_cdr` と同じ分割です。

```text
公開ソース ──①──▶ data_tr/raw ──②──▶ data_tr/processed ──③──▶ data_tr/prepared
      download_raw_data     tr-build-processed        tr-prepare
      （標準ライブラリのみ）  （要 h5py）               （numpy/torch のみ）
```

| 段階 | 実行コマンド | 実装 |
| --- | --- | --- |
| ① 取得 | `python -m scripts.tr.upstream.download_raw_data` | `scripts/tr/upstream/download_raw_data.py` |
| ② バンドル構築 | `pathwaygnn-data tr-build-processed --config configs/tr/build_processed.yaml` | `src/pathwaygnn_datasets/tr/build.py` |
| ③ 汎用形式へ変換 | `pathwaygnn-data tr-prepare --config configs/tr/prepare.yaml` | `src/pathwaygnn_datasets/tr/prepare.py` |

```text
data_tr/
├── raw/          22 GB   公開ソース（§2）
├── processed/    547 MB  中間バンドル（§3）
└── prepared/     693 MB  汎用形式（§4）
```

②は参考実装 `target-repositioning-share` の前処理ノートブック 01〜05 に対応します。

---

## 2. 元データ（`raw/`）

取得元 URL・引用すべき論文・入手手順は [`data_tr/README.md` §2](data_tr/README.md) に
まとめてあります。ここでは②が実際に読む中身だけを述べます。

### 2.1 LINCS L1000（GSE92742）

| ファイル | 内容 |
| --- | --- |
| `GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx` | Level 5 行列。473,647 シグネチャ × 12,328 遺伝子、GCTX（HDF5）。展開後 23.4 GB |
| `GSE92742_Broad_LINCS_sig_info.txt` | シグネチャごとの `pert_iname` / `pert_type` / `cell_id` / 時間 / 用量 |
| `GSE92742_Broad_LINCS_gene_info.txt` | 遺伝子表。`pr_is_lm == 1` が **978 ランドマーク遺伝子** |

使うのは次の2種類です（件数は先行研究の記載と一致します）。

| `pert_type` | 摂動 | プロファイル数 |
| --- | --- | --- |
| `trt_sh.cgs` | ノックダウン（shRNA consensus gene signature） | 36,720 |
| `trt_oe` | 過剰発現 | 22,205 |

GCTX は行列を**転置して**保持しています（`matrix[シグネチャ, 遺伝子]`）。②はランドマーク
978 列だけを、必要なシグネチャ行のブロック単位で読み出します（h5py の fancy index は1軸のみ）。
ノックダウンと過剰発現で行列を2回走査しないよう、和集合を一度に読んでから振り分けます。

### 2.2 CREEDS（`disease_signatures-v1.0.json`）

手動キュレーションされた疾患シグネチャ。各レコードが `do_id`（Disease Ontology）、
`organism`、`up_genes` / `down_genes`（`[遺伝子名, 値]` のリスト）を持ちます。
DOID が付いたレコードは 695 件、うち `organism == "human"` は **493 件**です。

### 2.3 ラベル（`{inhibitory,activatory}_target_disease.tsv`）

`gene` / `doid` / `label` の3列。`label` は 1 が「関係あり」、0 が「関係なし」。
九州工業大学の配布物（`target_disease_data.zip`）が元データですが、現在その URL は
403 のため、本リポジトリのリリースにミラーしたものを取得します。詳細は
[`data_tr/README.md` §2.6](data_tr/README.md)。

### 2.4 その他

| ファイル | 用途 |
| --- | --- |
| `hgnc_complete_set.txt` | 遺伝子シンボル変換表の元（承認・旧・エイリアス） |
| `PathwayCommons12.All.hgnc.sif` | パスウェイグラフ。3列 SIF、1,884,849 行、関係13種 |
| `kegg_disease_omim.list` / `HumanDO.obo` | KEGG DISEASE → OMIM → DOID の変換表（ラベルが KEGG ID 版のときだけ使う） |

---

## 3. ②`tr-build-processed` の処理内容

1. **シンボル変換表** — HGNC から「同義語 → 承認シンボル」の辞書を作ります（**58,074 件**）。
   承認シンボル自身は変換対象から外し、複数の遺伝子が同じ同義語を主張する場合は
   エイリアス側を優先。それでも一意に決まらない **1,135 件**は `SHARED:BBB/CCC` の形で
   曖昧さを残します（参考実装と同じ規則）。
2. **グラフ** — SIF の両端を変換して大文字化し、`graph.tsv` に書きます。
3. **摂動シグネチャ** — ランドマーク 978 遺伝子だけを読み、`(pert_iname, cell_id)` ごとに
   平均します（＝複製・測定時間・用量の平均）。
4. **疾患シグネチャ** — CREEDS を (疾患, 遺伝子) の行に展開し、同一疾患の複数研究は平均します。
5. **ラベル** — 陽性ペアから遺伝子 × 疾患の全組み合わせを作り、陽性以外を陰性とします。
   KEGG ID 版を渡した場合はここで KEGG→OMIM→DOID 変換（対応表 1,805 件）が走ります。

実データでの出力（`build_manifest.json`。所要 **52 秒**）:

| ファイル | 内容 | 規模 |
| --- | --- | --- |
| `graph.tsv` | 3列（起点 / 関係 / 終点）、ヘッダなし | 1,884,849 行 / 30,895 ノード / 13 関係 |
| `knockdown_signature.tsv` | `pert_iname`, `cell_id`, 978 遺伝子列 | **33,817 行**（36,720 プロファイルを集約） |
| `overexpression_signature.tsv` | 同上 | **20,131 行**（22,205 プロファイル） |
| `disease_specific_signature.tsv` | `do_id`, `gene_name`, `expression` | 186,884 行 / **178 疾患**（493 シグネチャ） |
| `inhibitory_target_disease.tsv` | `gene`, `doid`, `label` | 7,168 行（陽性 568） |
| `activatory_target_disease.tsv` | 同上 | 450 行（陽性 37） |

### 3.1 データの中身を決める2つのフラグ

`build_manifest.json` に記録されます。どちらもエンコードではなくデータの中身を変えます。

| フラグ | 既定 | 意味 |
| --- | --- | --- |
| `per_cell_line` | `true` | 摂動プロファイルの単位を `(pert_iname, cell_id)` にする。`false` なら細胞株を平均で潰す |
| `human_only` | `true` | CREEDS の `organism == "human"` だけを使う（DOID 付き 695 件のうち 493 件） |

なお参考実装は、先行研究が使っていた TT-WOPT による欠測補完を**採用していません**
（効果が確認できなかったため）。

---

## 4. ③`tr-prepare` の処理内容

```bash
bash scripts/tr/prepare.sh
# = pathwaygnn-data tr-prepare --config configs/tr/prepare.yaml
```

1. **グラフ構築** — 各エッジを**両方向に追加して対称化**し、`(src, relation, dst)` の集合で
   重複を除去、`sorted()` してから整数 ID を振ります。ノード名・関係名も `sorted()` するため、
   `PYTHONHASHSEED` に依存せず決定的です。
2. **疾患シグネチャ** — グラフに存在しない遺伝子の行は捨て（`disease_rows_skipped`）、
   絶対値が `cutoff`（既定 `1e-7`）未満の値も落として CSR 疎行列にします。
3. **摂動シグネチャ** — ヘッダの 978 列のうちグラフに存在するものだけを使い
   （`signature_genes_skipped`）、同じく `cutoff` で疎化します。
   `cell_id` 列があれば行名は `"<遺伝子>|<細胞株>"` になります。
4. **ラベル** — `gene` が摂動テーブルに、`doid` が疾患テーブルに存在しない行は捨てます
   （`label_rows_skipped`）。**残った1行は、その遺伝子が摂動された細胞株の数だけ
   サンプルに展開されます**（`label_rows_used` と `num_samples` の差）。
5. すべての件数と除外行数を `dataset.json` / `task.json` に記録します。

### 生成される汎用形式（実測、所要 **66 秒**）

| 項目 | 値 |
| --- | --- |
| ノード数 | 30,895 |
| 関係タイプ数 | 13 |
| 有向エッジ数 | 3,671,958（対称化・重複除去後） |
| 疾患数 | 177 |
| 除外された疾患シグネチャ行 | 10,431 |
| 除外されたランドマーク遺伝子 | 2（グラフに無い） |

**channel（データセット単位の遺伝子-値テーブル、すべて sparse）**

| channel | 行数 | 非ゼロ値数 | 内容 |
| --- | --- | --- | --- |
| `disease` | 177 | 175,975 | 疾患シグネチャ。**2タスクで共有** |
| `perturbation_kd` | 33,817 | 32,999,370 | ノックダウン（遺伝子 × 細胞株） |
| `perturbation_oe` | 20,131 | 19,638,912 | 過剰発現（同上） |

**task**

| task | サンプル数 | 陽性数 | ラベル行 | 除外ラベル行 | alias → channel |
| --- | --- | --- | --- | --- | --- |
| `kd_inh` | 61,101 | 5,013 | 6,913 | 255 | `perturbation`→`perturbation_kd`, `disease`→`disease` |
| `oe_act` | 3,465 | 294 | 450 | 0 | `perturbation`→`perturbation_oe`, `disease`→`disease` |

両タスクとも **alias は `perturbation` と `disease` で共通**なので、モデル設定がそのまま
流用できます。`groups` はサンプルが対象とする疾患（177種）で、疾患別 AUC と疾患別帰属の
集計に使われます。共変量はありません（`use_covariates: true` の variant は使えません）。

---

## 5. 実行

```bash
conda activate gnn
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

## 6. 注意点

- **サンプルは「遺伝子 × 疾患 × 細胞株」です。** 同じ (遺伝子, 疾患) ペアが細胞株の数だけ
  現れ、ラベルは同じです。サンプル単位のランダム分割では同一ペアが学習側と held-out 側に
  分かれるため、**細胞株をまたぐ汎化ではなく「別の細胞株での同じ関係の再現」**を
  測っていることになります。ペア単位で分割したい場合は fold 分割の変更が必要です
  （参考実装は `do_id` で層化した StratifiedKFold を使っています）。
- **`oe_act` は小さい**（450 ラベル行・陽性 37）。細胞株展開後も陽性は 294 件しかなく、
  5-fold の分散は大きいままです。
- **`cv` は `finetune` と同じ `pos_weight` を使います**（`configs/tr/cv.yaml` の
  `training.pos_weight: auto`＝fold の学習側の 陰性数/陽性数。実測 10.75〜11.19）。
  これを入れない素の BCE では、陽性 8.2% × 61,101 サンプルの `kd_inh` が事前確率
  0.0807 の定数出力に潰れ、F1 が 0 になります。採用値は各 fold の `metrics.json` の
  `pos_weight` に記録されます。
- **それでも `kd_inh` の順位付け性能はチャンス水準です。** `pos_weight` で操作点は
  改善しますが（F1 0.00 → 0.12）、AUC は `mlp` 0.518 / `gnn_mlp` 0.500 のままです。
  **データが無情報なわけではありません**: 同じ特徴量で木系ベースラインは AUC 0.81、
  `finetune` は test AUC 0.587 に達します。
- **木系ベースラインの方が強い**（下表）。グラフ encoder の寄与は両タスクとも負です。

5-fold 平均 ROC-AUC:

| task | `mlp` | `gnn_mlp` | LR | RF | XGBoost | finetune |
| --- | --- | --- | --- | --- | --- | --- |
| `kd_inh` | 0.518 | 0.500 | 0.755 | **0.809** | 0.803 | 0.587 |
| `oe_act` | 0.705 | 0.689 | 0.434 | **0.803** | 0.720 | 0.645 |

  詳細は [`docs/tr_report.md`](docs/tr_report.md)（`pathwaygnn-data tr-report` の生成物）。
- **IG のランキングは次数に強く相関します。** 上位は `CHEBI:*` の化学実体が占めがちで、
  疾患特異的な機序の証拠としては読めません。
- グラフノードは遺伝子シンボルだけでなく化学実体も含むため、ノード数はヒト遺伝子数より
  多くなります。
