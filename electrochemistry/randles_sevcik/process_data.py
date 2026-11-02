"""Extract peak currents and estimate electrode area from tidy CV data.

Run ``handle_data.py`` first for each batch, then run this script on each
batch's tidy folder. Example:

    python electrochemistry/randles_sevcik/handle_data.py
    python electrochemistry/randles_sevcik/process_data.py

The default paths point at the synthetic demo dataset.
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


CONFIG = {
    "n": 1,              # number of electrons
    "dr": 7.6e-6,        # diffusion coefficient, cm^2/s
    "constant": 2.69e5,  # Randles-Sevcik constant
    "conc": 5e-6,        # concentration, mol/cm^3
}


def main(tidy_dir: str = "Data/randles/tidy") -> None:
    if not os.path.isdir(tidy_dir):
        print(f"Error: tidy data folder '{tidy_dir}' does not exist.")
        print("Run handle_data.py first.")
        return

    peak_rows = []

    for filename in sorted(os.listdir(tidy_dir)):
        if not filename.endswith(".csv"):
            continue
        if "summary" in filename or "slopes" in filename:
            continue
        if "Electrode" not in filename:
            continue

        temp_df = pd.read_csv(os.path.join(tidy_dir, filename))
        electrode_id = temp_df["Electrode_ID"].iloc[0]

        for scan in sorted(temp_df["Scan Rate"].unique()):
            scan_df = temp_df[temp_df["Scan Rate"] == scan]
            potential = scan_df["Potential/V"].values
            current = scan_df["Current/A"].values

            potential_window = (0.1, 0.5)
            mask = (potential >= potential_window[0]) & (
                potential <= potential_window[1]
            )
            potential = potential[mask]
            current = current[mask]

            min_prominence = 0.1 * np.max(np.abs(current))
            peaks, properties = find_peaks(
                current, prominence=min_prominence, height=0, distance=10
            )

            if len(peaks) > 0:
                best_idx = np.argmax(properties["prominences"])
                peak_idx = peaks[best_idx]
                peak_current = current[peak_idx]
            else:
                peak_current = np.max(current)

            peak_rows.append(
                {
                    "Electrode_ID": electrode_id,
                    "Scan Rate (V/s)": scan / 1000,
                    "Ip (A)": peak_current,
                }
            )

    summary_df = (
        pd.DataFrame(peak_rows)
        .sort_values(by=["Electrode_ID", "Scan Rate (V/s)"])
        .reset_index(drop=True)
    )
    if summary_df.empty:
        print("No peak data found.")
        return

    slope_rows = []
    for electrode_id in summary_df["Electrode_ID"].unique():
        electrode_df = summary_df[
            summary_df["Electrode_ID"] == electrode_id
        ].copy()
        electrode_df["sqrt_scan_rate"] = np.sqrt(electrode_df["Scan Rate (V/s)"])
        slope, intercept = np.polyfit(
            electrode_df["sqrt_scan_rate"], electrode_df["Ip (A)"], 1
        )
        r_value = np.corrcoef(
            electrode_df["sqrt_scan_rate"], electrode_df["Ip (A)"]
        )[0, 1]
        slope_rows.append(
            {
                "Electrode_ID": electrode_id,
                "Slope (A/(V/s)^1/2)": slope,
                "R value": r_value,
            }
        )
        plt.plot(
            electrode_df["sqrt_scan_rate"],
            electrode_df["Ip (A)"],
            marker="o",
            label=f"Electrode {electrode_id}",
        )

    plt.xlabel("sqrt(Scan Rate) (V/s)^1/2")
    plt.ylabel("Ip (A)")
    plt.title("Ip vs sqrt(Scan Rate)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(tidy_dir, "Ip_vs_ScanRate.png"), dpi=300)
    plt.close()

    slopes_df = (
        pd.DataFrame(slope_rows)
        .sort_values(by=["Electrode_ID"])
        .reset_index(drop=True)
    )
    slopes_df["Area (cm^2)"] = slopes_df["Slope (A/(V/s)^1/2)"] / (
        CONFIG["constant"]
        * (CONFIG["n"] ** 1.5)
        * (CONFIG["dr"] ** 0.5)
        * CONFIG["conc"]
    )

    slopes_df.to_csv(os.path.join(tidy_dir, "slopes.csv"), index=False)
    summary_df.to_csv(os.path.join(tidy_dir, "summary.csv"), index=False)
    print(slopes_df)


if __name__ == "__main__":
    import sys

    tidy = sys.argv[1] if len(sys.argv) > 1 else "Data/randles/tidy"
    main(tidy)
