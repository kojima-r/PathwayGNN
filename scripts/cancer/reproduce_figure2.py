#!/usr/bin/env python3
"""Reproduce Figure 2: 5-year AUC against graph pre-training epoch, frozen vs end-to-end."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import yaml

EPOCHS = (0, 10, 20, 30, 40, 50)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cancer/cv.yaml")
    parser.add_argument("--report-config", default="configs/cancer/report.yaml")
    parser.add_argument("--gpus", default="0,1,2")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "src"))
    from pathwaygnn.config import load_config

    base = load_config(project / args.config)
    root = project / "outputs/cancer/pretraining_sweep"
    config_dir, logs = root / "configs", root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for epoch in EPOCHS:
        checkpoint = project / f"outputs/cancer/pretrain_sweep/epoch_{epoch}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"{checkpoint} is missing; run bash scripts/cancer/reproduce_paper.sh figure2-pretrain"
            )
        for end_to_end in (True, False):
            mode = "end_to_end" if end_to_end else "frozen"
            cfg = deepcopy(base)
            cfg["dataset"] = {**base["dataset"], "tasks": ["5year"]}
            cfg["variants"] = [{
                "name": "gnn_dnn", "use_graph": True, "use_sample_features": False,
                "seed_index": 2, "end_to_end": end_to_end,
            }]
            cfg["pretrained_checkpoint"] = str(checkpoint)
            cfg["output_dir"] = str(root / f"epoch_{epoch}" / mode)
            cfg["write_root_manifest"] = True
            path = config_dir / f"epoch_{epoch}_{mode}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            jobs.append((epoch, mode, path))
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]

    def worker(gpu: str, assigned: list[tuple[int, str, Path]]) -> list[dict]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        results = []
        for epoch, mode, path in assigned:
            log = logs / f"epoch_{epoch}_{mode}.log"
            with log.open("w") as stream:
                code = subprocess.run(
                    [sys.executable, "-m", "pathwaygnn.cli", "cv", "--config", str(path)],
                    cwd=project, env=env, stdout=stream, stderr=subprocess.STDOUT,
                ).returncode
            result = {"epoch": epoch, "mode": mode, "gpu": gpu, "returncode": code, "log": str(log)}
            print(json.dumps(result), flush=True)
            results.append(result)
        return results

    queues = [jobs[index::len(gpus)] for index in range(len(gpus))]
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, queue) for gpu, queue in zip(gpus, queues)]
        results = [item for future in futures for item in future.result()]
    (root / "manifest.json").write_text(json.dumps(results, indent=2))
    if any(item["returncode"] for item in results):
        return 1
    subprocess.run(
        [sys.executable, "-m", "pathwaygnn_datasets.cli", "cancer-report",
         "--config", str(project / args.report_config)],
        cwd=project, check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
