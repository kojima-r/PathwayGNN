#!/usr/bin/env python3
"""Run all Table 1 conditions with resumable GPU scheduling.

Splits ``configs/cancer/cv.yaml`` into one config per (task, variant), runs each
as ``pathwaygnn cv`` on its own GPU slot, then renders the report. Variants come
from the config, so the grid follows whatever that file declares.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cancer/cv.yaml")
    parser.add_argument("--report-config", default="configs/cancer/report.yaml")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU IDs; auto-detected by default")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def available_gpus(value: str | None) -> list[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        )
        found = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return found or [""]
    except (OSError, subprocess.CalledProcessError):
        return [""]


def load_grid_config(project: Path, config: str) -> dict:
    """Resolve `defaults:` the same way pathwaygnn.config does."""
    sys.path.insert(0, str(project / "src"))
    from pathwaygnn.config import load_config

    return load_config(project / config)


def main() -> int:
    args = parse_args()
    project = Path(__file__).resolve().parents[2]
    base = load_grid_config(project, args.config)
    root = project / base["output_dir"]
    config_dir, log_dir = root / "grid_configs", root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks = base["dataset"].get("tasks") or [base["dataset"]["task"]]
    jobs = []
    for task in tasks:
        for variant in base["variants"]:
            cfg = deepcopy(base)
            cfg["dataset"] = {**base["dataset"], "tasks": [task]}
            cfg["variants"] = [variant]
            cfg["write_root_manifest"] = False
            path = config_dir / f"{task}_{variant['name']}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            jobs.append((task, variant["name"], path))
    gpus = available_gpus(args.gpus)
    slots = [gpu for gpu in gpus for _ in range(max(args.jobs_per_gpu, 1))]
    expected = len(jobs)
    print(json.dumps({"jobs": expected, "gpus": gpus, "slots": len(slots), "root": str(root)}))
    if args.dry_run:
        return 0

    def run_queue(gpu: str, assigned: list[tuple[str, str, Path]]) -> list[dict]:
        queue_results = []
        env = os.environ.copy()
        if gpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        for task, variant, config in assigned:
            log_path = log_dir / f"{task}_{variant}.log"
            with log_path.open("w") as log:
                process = subprocess.run(
                    [sys.executable, "-m", "pathwaygnn.cli", "cv", "--config", str(config)],
                    cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT,
                )
            result = {"task": task, "variant": variant, "gpu": gpu or "cpu",
                      "returncode": process.returncode, "log": str(log_path)}
            queue_results.append(result)
            print(json.dumps(result), flush=True)
        return queue_results

    completed, failed = [], []
    queues = [jobs[index::len(slots)] for index in range(len(slots))]
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(run_queue, gpu, assigned) for gpu, assigned in zip(slots, queues)]
        for future in as_completed(futures):
            for result in future.result():
                completed.append(result)
                if result["returncode"]:
                    failed.append(result)

    summaries = {}
    for summary_path in sorted(root.glob("*/*/summary.json")):
        summary = json.loads(summary_path.read_text())
        summaries[f"{summary_path.parts[-3]}/{summary_path.parts[-2]}"] = summary
    (root / "cv_results.json").write_text(json.dumps(summaries, indent=2, allow_nan=True))
    manifest = {
        "complete": not failed and len(summaries) == expected,
        "conditions_expected": expected,
        "conditions_found": len(summaries),
        "jobs": completed,
        "failed": failed,
    }
    (root / "grid_manifest.json").write_text(json.dumps(manifest, indent=2))
    if failed:
        print(json.dumps(manifest, indent=2), file=sys.stderr)
        return 1
    subprocess.run(
        [sys.executable, "-m", "pathwaygnn_datasets.cli", "cancer-report",
         "--config", str(project / args.report_config)],
        cwd=project, check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
