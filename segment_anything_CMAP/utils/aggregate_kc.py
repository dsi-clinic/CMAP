#!/usr/bin/env python

"""
Aggregator script that merges IoU results from all available runs in kc_sam_outputs/,
computes final mean/std per class, adds an "overall" row, and visualizes results.

Steps:
  1) Finds **all** subdirectories in 'kc_sam_outputs/' (ignoring SLURM_JOB_ID).
  2) Gathers and merges 'per_class_ious_subset_*.csv' files from all runs.
  3) Computes weighted mean + pooled std for each class across subsets.
  4) Creates an "overall" bar that combines all classes.
  5) Saves a final CSV and produces a bar chart (class on x-axis, mean IoU on y-axis,
     error bars for std).
"""

import glob
import os
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from datetime import datetime

def find_all_runs(base_out_dir):
    """Finds all subdirectories inside kc_sam-outputs/ to aggregate results from."""
    return sorted(glob.glob(os.path.join(base_out_dir, "kc_sam_run_*")), key=os.path.getmtime)

def main():
    home_dir = os.path.expanduser("~")

    # Use all available kc_sam_run_* directories instead of SLURM_JOB_ID
    base_out_dir = os.path.join(home_dir, "CMAP", "segment_anything_CMAP", "kc_sam_outputs")
    run_folders = find_all_runs(base_out_dir)

    if not run_folders:
        print("[ERROR] No previous kc_sam_run_* directories found in kc_sam_outputs/")
        return

    print(f"[INFO] Found {len(run_folders)} completed runs.")

    # Output directory for final CSV + plot
    stats_dir = os.path.join(home_dir, "CMAP", "segment_anything_CMAP", "kc_sam_statistics")
    os.makedirs(stats_dir, exist_ok=True)

    # Gather all per_class_ious_subset_*.csv files from all runs
    csv_files = []
    for run_folder in run_folders:
        csv_files.extend(glob.glob(os.path.join(run_folder, "per_class_ious_subset_*.csv")))

    if not csv_files:
        print("[ERROR] No per_class_ious_subset_*.csv files found in any run folder.")
        return

    print(f"[INFO] Found {len(csv_files)} total CSVs from all runs.")

    # Store per-class statistics {class_id: [(n_i, mean_iou, std_iou), ...]}
    partial_data = {}

    for csvf in csv_files:
        df = pd.read_csv(csvf)
        for _, row in df.iterrows():
            cls_id = int(row["class_id"])
            mu_i = float(row["mean_iou"])
            sigma_i = float(row["std_iou"])
            n_i = int(row["count"])

            if cls_id not in partial_data:
                partial_data[cls_id] = []
            partial_data[cls_id].append((n_i, mu_i, sigma_i))

    # Weighted mean & pooled std aggregator
    results = {}
    for cls_id, vals in partial_data.items():
        N = 0
        sum_mu = 0.0
        sum_squares = 0.0

        for (n_i, mu_i, sigma_i) in vals:
            N += n_i
            sum_mu += n_i * mu_i
            sum_squares += (n_i - 1)*(sigma_i**2) + n_i*(mu_i**2)

        if N > 1:
            global_mean = sum_mu / N
            var = (sum_squares - N*(global_mean**2)) / (N - 1)
            var = max(var, 0.0)
            global_std = math.sqrt(var)
        elif N == 1:
            global_mean = sum_mu
            global_std = 0.0
        else:
            global_mean = 0.0
            global_std = 0.0

        results[cls_id] = {
            "class_id": cls_id,
            "mean_iou": global_mean,
            "std_iou": global_std,
            "count": N
        }

    # Compute "overall" row across all classes
    total_N = sum(info["count"] for info in results.values())
    total_sum_mu = sum(info["count"] * info["mean_iou"] for info in results.values())
    total_sum_squares = sum((info["count"] - 1) * (info["std_iou"]**2) +
                            info["count"] * (info["mean_iou"]**2) for info in results.values())

    if total_N > 1:
        overall_mean = total_sum_mu / total_N
        var = (total_sum_squares - total_N * (overall_mean**2)) / (total_N - 1)
        var = max(var, 0.0)
        overall_std = math.sqrt(var)
    elif total_N == 1:
        overall_mean = total_sum_mu
        overall_std = 0.0
    else:
        overall_mean = 0.0
        overall_std = 0.0

    results["overall"] = {
        "class_id": -1,
        "mean_iou": overall_mean,
        "std_iou": overall_std,
        "count": total_N
    }

    # Convert to DataFrame
    final_df = pd.DataFrame(list(results.values()))
    final_df.sort_values(by=["class_id"], inplace=True)

    # Save final aggregator CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_csv_name = f"kc_sam_aggregated_allruns_{ts}.csv"
    final_path = os.path.join(stats_dir, final_csv_name)
    final_df.to_csv(final_path, index=False)

    print(f"[INFO] Wrote aggregated CSV to {final_path}")

    # Plot classes on x-axis, mean iou on y-axis, error bars for std
    plot_df = final_df[final_df["count"] > 0].copy()  # Skip empty

    x_labels = []
    means = []
    stds = []

    for _, row in plot_df.iterrows():
        cid = row["class_id"]
        mu = row["mean_iou"]
        sd = row["std_iou"]
        x_labels.append("Overall" if cid == -1 else str(int(cid)))
        means.append(mu)
        stds.append(sd)

    import numpy as np
    x_positions = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_positions, means, yerr=stds, align='center', alpha=0.7,
           ecolor='black', capsize=5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylim([0, 1.0])
    ax.set_ylabel("Mean IoU")
    ax.set_xlabel("Class Label")
    ax.set_title("Segment Anything Class-wise Mean IoU with Std. Dev.")
    ax.yaxis.grid(True)

    # Save bar chart
    plot_name = f"kc_sam_aggregated_allruns_{ts}.png"
    plot_path = os.path.join(stats_dir, plot_name)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Saved bar chart with error bars to {plot_path}")

if __name__ == "__main__":
    main()
