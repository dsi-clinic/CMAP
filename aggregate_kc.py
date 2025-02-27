#!/usr/bin/env python

import os
import glob
import pandas as pd
from datetime import datetime

def main():
    home_dir = os.path.expanduser("~")

    # 1) The partial outputs directory
    # job id is read from environment or you can manually put it if no Slurm
    job_id = os.getenv("SLURM_JOB_ID", "local")

    base_out_dir = os.path.join(
        home_dir, "CMAP", "segment-anything", "kc-sam-outputs"
    )
    run_folder_name = f"kc_sam_run_{job_id}"

    partial_dir = os.path.join(base_out_dir, run_folder_name)
    if not os.path.isdir(partial_dir):
        print(f"[ERROR] partial_dir not found: {partial_dir}")
        return

    # 2) Where we save the final CSV
    stats_dir = os.path.join(home_dir, "CMAP", "segment-anything", "kc-sam-statistics")
    os.makedirs(stats_dir, exist_ok=True)

    # 3) Gather partial CSVs 
    # NOTE: THIS INCLUDES ALL CSVs IN THE OUTPUT DIRECTORY, INCLUDING THOSE FROM PREVIOUS RUNS!
    csv_files = glob.glob(os.path.join(partial_dir, "per_class_ious_subset_*.csv"))
    if not csv_files:
        print(f"No partial CSVs found in {partial_dir}")
        return

    print(f"[INFO] Found {len(csv_files)} partial CSVs in {partial_dir}. Merging...")

    # 4) Merge them
    all_dfs = []
    for csvf in csv_files:
        df = pd.read_csv(csvf)
        # optionally note the subset index from filename
        all_dfs.append(df)

    merged = pd.concat(all_dfs, ignore_index=True)

    # 5) Summarize or just save raw
    # e.g. if we want overall stats, do a groupby('class_id') ...
    # For now, we just write the combined data.

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_csv_name = f"kc_sam_aggregated_{job_id}_{timestamp}.csv"
    final_path = os.path.join(stats_dir, final_csv_name)

    merged.to_csv(final_path, index=False)
    print(f"[INFO] Wrote final aggregated CSV to {final_path}")
    print("[INFO] No images copied; we only wrote a single final CSV in statistics directory.")

if __name__ == "__main__":
    main()
