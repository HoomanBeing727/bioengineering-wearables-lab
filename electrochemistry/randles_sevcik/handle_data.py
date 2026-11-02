"""Convert raw cyclic-voltammetry text exports into tidy per-electrode CSVs.

Expected raw file name format:

    10 mVs Bare_1.txt

Run from the repository root:

    python electrochemistry/randles_sevcik/handle_data.py
"""

import os
import re
from io import StringIO

import pandas as pd


FILE_PATTERN = re.compile(
    r"^(?P<scanrate>\d+)\s+mVs\s+(?P<electrode_id>.+)\.txt$"
)


def _read_cv_file(path: str, electrode_id: str, scanrate: int):
    """Read one CHI-style CV export, starting at the Potential/Current header."""
    with open(path, "r") as f:
        for line in f:
            clean_line = line.strip()
            if "Potential/V" in clean_line and "Current/A" in clean_line:
                remaining_data = line + f.read()
                df_temp = pd.read_csv(
                    StringIO(remaining_data), sep=",", skipinitialspace=True
                )
                break
        else:
            print(
                f"Skipping {os.path.basename(path)}: could not find "
                "Potential/V and Current/A header"
            )
            return None

    df_temp["Electrode_ID"] = electrode_id
    df_temp["Scan Rate"] = scanrate
    return df_temp


def main(
    raw_dir: str = "Data/randles/raw",
    tidy_dir: str = "Data/randles/tidy",
) -> None:
    if not os.path.isdir(raw_dir):
        print(f"Error: raw data folder '{raw_dir}' does not exist.")
        print("Run examples/generate_synthetic_data.py from the repository root first.")
        return

    all_dataframes = []
    for filename in sorted(os.listdir(raw_dir)):
        if not filename.endswith(".txt"):
            continue

        match = FILE_PATTERN.match(filename)
        if not match:
            print(f"Skipping {filename}: did not match '10 mVs Electrode.txt' pattern")
            continue

        electrode_id = match.group("electrode_id").strip()
        scanrate = int(match.group("scanrate"))
        df_temp = _read_cv_file(
            os.path.join(raw_dir, filename), electrode_id, scanrate
        )
        if df_temp is not None:
            all_dataframes.append(df_temp)

    if not all_dataframes:
        print("No data found - check file naming and folder path")
        return

    master_df = pd.concat(all_dataframes, ignore_index=True)
    master_df = master_df[["Potential/V", "Current/A", "Electrode_ID", "Scan Rate"]]

    os.makedirs(tidy_dir, exist_ok=True)
    for electrode, group_df in master_df.groupby("Electrode_ID"):
        final_file = f"Electrode_{str(electrode).strip()}.csv"
        output_path = os.path.join(tidy_dir, final_file)
        group_df.to_csv(output_path, index=False)
        print(f"Saved data for electrode '{electrode}' to {output_path}")


if __name__ == "__main__":
    import sys

    raw = sys.argv[1] if len(sys.argv) > 1 else "Data/randles/raw"
    tidy = sys.argv[2] if len(sys.argv) > 2 else "Data/randles/tidy"
    main(raw, tidy)
