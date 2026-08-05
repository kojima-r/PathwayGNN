"""Regenerate the toy corpus under ``data_sample/raw/``.

The files this writes are **committed to the repository** — you do not need to run
this script to follow the tutorial in ``README.md``. It exists so that the toy
corpus is auditable (every number has a documented origin) and adjustable: change
``NUM_SAMPLES`` or ``SEED`` below, re-run, and ``pathwaygnn-data sample-prepare``
still works unchanged.

Everything is synthetic. The gene names say what each module does:

* ``GROWTH1..GROWTH7`` and ``IMMUNE1..IMMUNE7`` drive the labels.
* ``NOISE1..NOISE6`` drive nothing — they are there so that attribution
  (``pathwaygnn ig``) has something it *should not* rank highly.

Every sample draws one hidden activity per module, and the genes of that module
are noisy readouts of it::

    expression[gene] = max(0, 4.0 + activity[module(gene)] + tissue_shift[tissue, gene]
                              + N(0, 0.5^2))

The 4.0 offset keeps the values non-negative, the way a real ``log2(TPM + 1)``
matrix is. That is not cosmetic: Integrated Gradients aggregates *signed*
value x gradient over samples, so a centred (positive and negative) matrix would
make the contributions of a well-fit model cancel between samples.

Labels are a documented function of those hidden activities, so the tutorial can
check that the model recovers a rule we already know::

    responder = 1  if  2.0*growth - 2.0*immune + 0.8*z(stage)  + N(0, 0.3^2)
                       is above its median over all 60 samples
    relapse   = 1  if  2.0*immune - 0.04*(age - 62)            + N(0, 0.3^2)
                       is above its median over the 48 samples that have follow-up

The two tasks are therefore driven by *different* modules — ``responder`` by both,
``relapse`` by IMMUNE and age alone — which is what makes their attributions worth
comparing, while ``NOISE*`` enters neither rule.

Usage (numpy only, ~0.1 s):

    python scripts/sample/make_raw_data.py                 # -> data_sample/raw
    python scripts/sample/make_raw_data.py --output-dir /tmp/raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 0
NUM_SAMPLES = 60
NUM_WITHOUT_FOLLOWUP = 12  # samples whose `relapse` column is NA
MODULES = {
    "GROWTH": 7,  # drives `responder` positively
    "IMMUNE": 7,  # drives `responder` negatively and `relapse` positively
    "NOISE": 6,  # drives nothing
}
GENE_NOISE = 0.5  # per-gene measurement noise on top of the module activity
BASELINE = 4.0  # keeps expression non-negative, like a log2(TPM + 1) matrix
LABEL_NOISE = 0.3  # noise in the label rules
TISSUES = ("TISSUE_A", "TISSUE_B", "TISSUE_C")
SIGNATURE_GENES_PER_TISSUE = 8
RELATIONS = ("controls-expression-of", "in-complex-with", "interacts-with")

# One undirected edge per line; `sample-prepare` adds the reverse direction, the
# same way the PathwayCommons SIF files are symmetrized for the real corpora.
EDGES: tuple[tuple[str, str, str], ...] = (
    # The GROWTH module: a linear cascade plus a hub at GROWTH1.
    ("GROWTH1", "controls-expression-of", "GROWTH2"),
    ("GROWTH2", "controls-expression-of", "GROWTH3"),
    ("GROWTH3", "controls-expression-of", "GROWTH4"),
    ("GROWTH4", "controls-expression-of", "GROWTH5"),
    ("GROWTH5", "controls-expression-of", "GROWTH6"),
    ("GROWTH6", "controls-expression-of", "GROWTH7"),
    ("GROWTH1", "in-complex-with", "GROWTH3"),
    ("GROWTH1", "in-complex-with", "GROWTH5"),
    ("GROWTH1", "in-complex-with", "GROWTH7"),
    ("GROWTH2", "interacts-with", "GROWTH6"),
    # The IMMUNE module.
    ("IMMUNE1", "controls-expression-of", "IMMUNE2"),
    ("IMMUNE2", "controls-expression-of", "IMMUNE3"),
    ("IMMUNE3", "controls-expression-of", "IMMUNE4"),
    ("IMMUNE4", "controls-expression-of", "IMMUNE5"),
    ("IMMUNE5", "controls-expression-of", "IMMUNE6"),
    ("IMMUNE6", "controls-expression-of", "IMMUNE7"),
    ("IMMUNE1", "in-complex-with", "IMMUNE4"),
    ("IMMUNE2", "in-complex-with", "IMMUNE5"),
    # The NOISE module: connected, but unrelated to both labels.
    ("NOISE1", "interacts-with", "NOISE2"),
    ("NOISE2", "interacts-with", "NOISE3"),
    ("NOISE3", "interacts-with", "NOISE4"),
    ("NOISE4", "interacts-with", "NOISE5"),
    ("NOISE5", "interacts-with", "NOISE6"),
    ("NOISE1", "in-complex-with", "NOISE4"),
    # Three edges between modules, so the graph is a single component.
    ("GROWTH4", "interacts-with", "IMMUNE1"),
    ("GROWTH7", "interacts-with", "IMMUNE7"),
    ("IMMUNE3", "interacts-with", "NOISE1"),
)


def gene_names() -> list[str]:
    return [f"{module}{index + 1}" for module, size in MODULES.items() for index in range(size)]


def _write_tsv(path: Path, header: list[str] | None, rows: list[list[object]]) -> None:
    lines = [] if header is None else ["\t".join(header)]
    lines += ["\t".join(str(value) for value in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(output_dir: Path, seed: int = SEED, num_samples: int = NUM_SAMPLES) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    genes = gene_names()
    rng = np.random.default_rng(seed)
    module_of = np.concatenate(
        [np.full(size, position) for position, size in enumerate(MODULES.values())]
    )
    # A per-tissue baseline shift, so that the tissue is visible in the data and
    # the per-group metrics mean something.
    tissue_shift = rng.normal(scale=0.4, size=(len(TISSUES), len(genes)))
    tissue_code = np.arange(num_samples) % len(TISSUES)

    # One hidden activity per module per sample. The genes of a module are noisy
    # readouts of it, so they move together — the only structure the graph could
    # possibly help the model exploit — and the labels below are a function of
    # these activities, never of the individual genes.
    activity = rng.normal(size=(num_samples, len(MODULES)))
    expression = np.round(
        np.clip(
            BASELINE
            + activity[:, module_of]
            + tissue_shift[tissue_code]
            + GENE_NOISE * rng.normal(size=(num_samples, len(genes))),
            0.0,
            None,
        ),
        3,
    )

    age = np.clip(np.round(rng.normal(62, 9, size=num_samples)), 40, 85).astype(int)
    sex_female = rng.integers(0, 2, size=num_samples)
    stage = rng.integers(1, 5, size=num_samples)
    smoker = rng.integers(0, 2, size=num_samples)

    growth_activity, immune_activity = activity[:, 0], activity[:, 1]
    responder_score = (
        2.0 * growth_activity
        - 2.0 * immune_activity
        + 0.8 * (stage - 2.5) / 1.12
        + rng.normal(scale=LABEL_NOISE, size=num_samples)
    )
    responder = (responder_score > np.median(responder_score)).astype(int)

    # `relapse` is only observed for samples that have follow-up, which is what
    # makes it a task over a *subset* of the samples (see `rows/` in the prepared
    # format).
    followup = np.ones(num_samples, dtype=bool)
    followup[rng.choice(num_samples, size=NUM_WITHOUT_FOLLOWUP, replace=False)] = False
    relapse_score = (
        2.0 * immune_activity
        - 0.04 * (age - 62)
        + rng.normal(scale=LABEL_NOISE, size=num_samples)
    )
    relapse = np.where(
        followup, (relapse_score > np.median(relapse_score[followup])).astype(int), -1
    )

    sample_ids = [f"S{index + 1:02d}" for index in range(num_samples)]
    _write_tsv(output_dir / "graph.tsv", None, [list(edge) for edge in EDGES])
    _write_tsv(
        output_dir / "expression.tsv",
        ["sample_id", *genes],
        [[sample_ids[row], *expression[row]] for row in range(num_samples)],
    )
    _write_tsv(
        output_dir / "samples.tsv",
        ["sample_id", "tissue", "age", "sex_female", "stage", "smoker", "responder", "relapse"],
        [
            [
                sample_ids[row],
                TISSUES[tissue_code[row]],
                age[row],
                sex_female[row],
                stage[row],
                smoker[row],
                responder[row],
                "NA" if relapse[row] < 0 else relapse[row],
            ]
            for row in range(num_samples)
        ],
    )
    # The tissue signature is a *long* table: one row per (tissue, gene) pair,
    # listing only the genes that mark that tissue. It becomes a sparse
    # node-level feature with one row per tissue, shared by every sample of it.
    signature_rows = []
    for code, tissue in enumerate(TISSUES):
        order = np.argsort(-np.abs(tissue_shift[code]))[:SIGNATURE_GENES_PER_TISSUE]
        for gene in sorted(order):
            signature_rows.append([tissue, genes[gene], round(float(tissue_shift[code, gene]), 3)])
    _write_tsv(output_dir / "tissue_signature.tsv", ["tissue", "gene", "value"], signature_rows)

    manifest = {
        "generator": "scripts/sample/make_raw_data.py",
        "seed": seed,
        "synthetic": True,
        "num_samples": num_samples,
        "num_genes": len(genes),
        "modules": dict(MODULES),
        "relations": list(RELATIONS),
        "num_undirected_edges": len(EDGES),
        "tissues": list(TISSUES),
        "samples_per_tissue": {
            tissue: int((tissue_code == code).sum()) for code, tissue in enumerate(TISSUES)
        },
        "expression_rule": (
            f"expression[gene] = max(0, {BASELINE} + activity[module(gene)] "
            f"+ tissue_shift[tissue, gene] + N(0,{GENE_NOISE}^2))"
        ),
        "labels": {
            "responder": {
                "rule": f"2.0*growth_activity - 2.0*immune_activity + 0.8*z(stage) "
                f"+ N(0,{LABEL_NOISE}^2), thresholded at its median",
                "num_samples": num_samples,
                "num_positive": int(responder.sum()),
            },
            "relapse": {
                "rule": f"2.0*immune_activity - 0.04*(age-62) + N(0,{LABEL_NOISE}^2), "
                "thresholded at its median over the samples with follow-up",
                "num_samples": int(followup.sum()),
                "num_positive": int((relapse == 1).sum()),
            },
        },
        "sample_features": ["age", "sex_female", "stage", "smoker"],
        "informative_sample_features": ["stage (responder)", "age (relapse)"],
        "signature_genes_per_tissue": SIGNATURE_GENES_PER_TISSUE,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data_sample/raw")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    args = parser.parse_args()
    manifest = build(Path(args.output_dir), args.seed, args.num_samples)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
