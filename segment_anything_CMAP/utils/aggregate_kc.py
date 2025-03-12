#!/usr/bin/env python

"""Aggregator script that merges IoU results from all available runs in kc_sam_outputs.

Computes final mean/std per class, adds an "overall" row, and visualizes results.
"""

import math
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use a non-interactive backend for matplotlib (headless mode)
matplotlib.use("Agg")


def find_all_runs(base_out_dir):
    """Finds all subdirectories inside kc_sam_outputs/ to aggregate results from."""
    return sorted(
        Path(base_out_dir).glob("kc_sam_run_*"), key=lambda p: p.stat().st_mtime
    )


def main():
    """Aggregates IoU results from multiple kc_sam_runs and generates summary statistics."""
    home_dir = Path.home()
    base_out_dir = home_dir / "CMAP" / "segment_anything_CMAP" / "kc_sam_outputs"
    run_folders = find_all_runs(base_out_dir)

    if not run_folders:
        print("[ERROR] No previous kc_sam_run_* directories found in kc_sam_outputs/")
        return

    print(f"[INFO] Found {len(run_folders)} completed runs.")

    # Output directory for final CSV + plot
    stats_dir = home_dir / "CMAP" / "segment_anything_CMAP" / "kc_sam_statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Gather all per_class_ious_subset_*.csv files from all runs
    csv_files = [
        csv_file
        for run_folder in run_folders
        for csv_file in run_folder.glob("per_class_ious_subset_*.csv")
    ]

    if not csv_files:
        print("[ERROR] No per_class_ious_subset_*.csv files found in any run folder.")
        return

    print(f"[INFO] Found {len(csv_files)} total CSVs from all runs.")

    # Store per-class statistics {class_id: [(n_i, mean_iou, std_iou), ...]}
    partial_data = {}

    for csvf in csv_files:
        iou_data = pd.read_csv(csvf)  # Fixed: Renamed from `df` to `iou_data`
        for _, row in iou_data.iterrows():
            cls_id = int(row["class_id"])
            mu_i = float(row["mean_iou"])
            sigma_i = float(row["std_iou"])
            n_i = int(row["count"])

            partial_data.setdefault(cls_id, []).append((n_i, mu_i, sigma_i))

    # Compute weighted mean & pooled std for each class
    results = {}
    for cls_id, vals in partial_data.items():
        N, sum_mu, sum_squares = 0, 0.0, 0.0

        for n_i, mu_i, sigma_i in vals:
            N += n_i
            sum_mu += n_i * mu_i
            sum_squares += (n_i - 1) * (sigma_i**2) + n_i * (mu_i**2)

        if N > 1:
            global_mean = sum_mu / N
            var = (sum_squares - N * (global_mean**2)) / (N - 1)
            global_std = math.sqrt(max(var, 0.0))
        elif N == 1:
            global_mean, global_std = sum_mu, 0.0
        else:
            global_mean, global_std = 0.0, 0.0

        results[cls_id] = {
            "class_id": cls_id,
            "mean_iou": global_mean,
            "std_iou": global_std,
            "count": N,
        }

    # Compute "overall" row across all classes
    total_N = sum(info["count"] for info in results.values())
    total_sum_mu = sum(info["count"] * info["mean_iou"] for info in results.values())
    total_sum_squares = sum(
        (info["count"] - 1) * (info["std_iou"] ** 2)
        + info["count"] * (info["mean_iou"] ** 2)
        for info in results.values()
    )

    if total_N > 1:
        overall_mean = total_sum_mu / total_N
        var = (total_sum_squares - total_N * (overall_mean**2)) / (total_N - 1)
        overall_std = math.sqrt(max(var, 0.0))
    elif total_N == 1:
        overall_mean, overall_std = total_sum_mu, 0.0
    else:
        overall_mean, overall_std = 0.0, 0.0

    results["overall"] = {
        "class_id": -1,
        "mean_iou": overall_mean,
        "std_iou": overall_std,
        "count": total_N,
    }

    # Convert to DataFrame & save
    final_df = pd.DataFrame(results.values()).sort_values(by=["class_id"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_csv_path = stats_dir / f"kc_sam_aggregated_allruns_{ts}.csv"
    final_df.to_csv(final_csv_path, index=False)

    print(f"[INFO] Wrote aggregated CSV to {final_csv_path}")

    # Plot results
    plot_df = final_df[final_df["count"] > 0].copy()
    x_labels = [
        "Overall" if row["class_id"] == -1 else str(int(row["class_id"]))
        for _, row in plot_df.iterrows()
    ]
    means, stds = plot_df["mean_iou"].tolist(), plot_df["std_iou"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = np.arange(len(x_labels))

    ax.bar(
        x_positions,
        means,
        yerr=stds,
        align="center",
        alpha=0.7,
        ecolor="black",
        capsize=5,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylim([0, 1.0])
    ax.set_ylabel("Mean IoU")
    ax.set_xlabel("Class Label")
    ax.set_title("Segment Anything Class-wise Mean IoU with Std. Dev.")
    ax.yaxis.grid(True)

    # Save plot
    plot_path = stats_dir / f"kc_sam_aggregated_allruns_{ts}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Saved bar chart with error bars to {plot_path}")


if __name__ == "__main__":
    main()
