"""Submit one `train.py` run through Slurm/submitit and append its average Train/Test IoU to a cumulative `all_iou_summaries.json`.

Usage
-----
1. On the login node: `micromamba activate cmap`
2. Launch: `python launcher_submitit.py <experiment_name> <num_trials> [--debug]`
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import submitit
from submitit.helpers import CommandFunction

LAUNCHER_DIR = Path(__file__).resolve().parent
GLOBAL_LOG = LAUNCHER_DIR / "all_iou_summaries.json"

## Helper Functions: (1) Build `train.py` command, (2) Parse standard output for Train/Test IOUs


def _build_cmd(args):
    """Return the shell command that executes *one* train.py call."""
    cmd = [
        "python",
        "train.py",
        "configs.config",
        f"--experiment_name={args.experiment}",
        f"--num_trials={args.num_trial}",
    ]
    if args.debug:
        cmd.append("--debug")
    return cmd


def _parse_stdout(text, expected_num_vals=2):
    """Extract Train and Test IoU averages from train.py stdout."""
    avg_re = re.compile(r"average:\s*([0-9.]+)")
    vals = avg_re.findall(text)
    if len(vals) < expected_num_vals:
        raise RuntimeError("Missing 'average:' lines in stdout")
    return float(vals[0]), float(vals[1])


## Main Function:


def main():
    """Parse CLI args, submit the Slurm job, and update the global IoU log."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "experiment", nargs="?", default=datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    p.add_argument("num_trial", type=int, default=5)
    p.add_argument("--debug", action="store_true")
    # Slurm resources with explicit types
    p.add_argument("--partition", default="general")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", type=int, default=128)
    p.add_argument("--time", type=int, default=360)
    args = p.parse_args()

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
    train_iou, test_iou = _parse_stdout(stdout)

    if GLOBAL_LOG.exists():
        with GLOBAL_LOG.open() as fp:
            global_data = json.load(fp)
    else:
        global_data = {}

    global_data[args.experiment] = [train_iou, test_iou]
    with GLOBAL_LOG.open("w") as fp:
        json.dump(global_data, fp, indent=2)

    print("Global log updated at", GLOBAL_LOG)


if __name__ == "__main__":
    main()
