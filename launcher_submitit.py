"""Submit one `train.py` run through Slurm/submitit and append its results to a JSON summary file.

Usage
-----
On login node run: `micromamba activate cmap`
Then:
    python launcher_submitit.py <experiment_name> <num_trials> [--debug] [--config CONFIG] [--log_file PATH]

`--config` specifies the Python import path to the config module (default: `configs.config`).
`--log_file` specifies the cumulative log path (default: `iou_summary.json` next to this script).
"""

import argparse
import json
import re
from typing import Tuple, List
from datetime import datetime
from pathlib import Path

import submitit
from submitit.helpers import CommandFunction

from configs import config

LAUNCHER_DIR = Path(__file__).resolve().parent
DEFAULT_LOG = LAUNCHER_DIR / "iou_summary.json"


def _build_cmd(args):
    """Build command to be submitted to the Slurm cluster"""
    cmd = [
        "python",
        "train.py",
        args.config,
        f"--experiment_name={args.experiment}",
        f"--num_trials={args.num_trial}",
    ]
    if args.debug:
        cmd.append("--debug")
    return cmd


def _parse_stdout(text: str) -> Tuple[float, float, List[str], List[str]]:
    """
    Extract:
      • average train IoU
      • average test  IoU
      • list of individual train IoUs
      • list of individual test  IoUs
    from the stdout produced by ``train.py``.
    """
    # Strip ANSI colour codes, just in case                      
    text = re.sub(r"\x1b\\[[0-9;]*m", "", text)

    # Regexes for the two summary blocks                       
    block   = r"{label} result:\s*\[([^\]]*)\][^\n]*?average:\s*([0-9.]+)"
    train_r = re.search(block.format(label="Training"), text, re.I | re.S)
    test_r  = re.search(block.format(label="Test"),     text, re.I | re.S)

    if not (train_r and test_r):
        raise RuntimeError(
            "Could not locate ‘Training result’ or ‘Test result’ blocks in job "
            "output. Check that train.py prints them unchanged."
        )

    # Helper: turn the comma-separated string into a list of numbers as strings
    def _split(valuestr: str) -> List[str]:
        return [v.strip() for v in valuestr.split(",") if v.strip()]

    train_list = _split(train_r.group(1))
    test_list  = _split(test_r.group(1))
    avg_train  = float(train_r.group(2))
    avg_test   = float(test_r.group(2))

    return avg_train, avg_test, train_list, test_list


def main():
    """Submit one `train.py` run through Slurm/submitit and append its results to a JSON summary file"""
    p = argparse.ArgumentParser()
    p.add_argument(
        "experiment", nargs="?", default=datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    p.add_argument("num_trial", type=int, default=5)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--config", default="configs.config")
    p.add_argument("--log_file", default=str(DEFAULT_LOG))
    # Slurm knobs
    p.add_argument("--partition", default="general")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", type=int, default=128)
    p.add_argument("--time", type=int, default=360)
    args = p.parse_args()

    cumulative_log = Path(args.log_file)

    log_root = Path("slurm_logs") / args.experiment
    log_root.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_root)
    executor.update_parameters(
        timeout_min=args.time,
        slurm_partition=args.partition,
        gpus_per_node=args.gpus,
        tasks_per_node=1,
        mem_gb=args.mem,
        cpus_per_task=args.cpus,
        name="train",
    )

    job = executor.submit(CommandFunction(_build_cmd(args), verbose=True))
    print(f"Submitted job {job.job_id}. Waiting…")

    stdout = job.result()
    avg_train, avg_test, train_list, test_list = _parse_stdout(stdout)

    out_dir = Path(config.OUTPUT_ROOT) / f"{args.experiment}_trial_0"

    summary_entry = {
        "Experiment": args.experiment,
        "AvgTrainIoU": round(avg_train, 3),
        "AvgTestIoU": round(avg_test, 3),
        "TrainIOUs": train_list,
        "TestIOUs": test_list,
        "OutputDir": str(out_dir),
    }

    global_data = json.load(cumulative_log.open()) if cumulative_log.exists() else {}
    global_data[args.experiment] = summary_entry
    json.dump(global_data, cumulative_log.open("w"), indent=2)
    print("Global log updated at", cumulative_log)


if __name__ == "__main__":
    main()
