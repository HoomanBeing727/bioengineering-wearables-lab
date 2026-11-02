import os

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.stats import linregress


def integrate_data(
    subfolder,
    input_folder,
    output_folder,
    time_col="Time/sec",
    current_col="Current/A",
    conc_col="Concentration",
):
    input_folder = os.path.join("Data", subfolder, input_folder)
    output_folder = os.path.join("Data", subfolder, output_folder)
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' does not exist.")
        return None

    processed_any = False

    for filename in os.listdir(input_folder):
        if not filename.endswith(".csv"):
            continue
        if "Tidy" in filename or "optimisation" in filename:
            continue

        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path).sort_values(by=[conc_col, time_col])

        all_dataframes = []
        for _, group in df.groupby(conc_col):
            group = group.copy()
            group["Charge/C"] = cumulative_trapezoid(
                group[current_col], group[time_col], initial=0
            )
            all_dataframes.append(group)

        if all_dataframes:
            master_df = pd.concat(all_dataframes, ignore_index=True)
            os.makedirs(output_folder, exist_ok=True)
            master_df.to_csv(os.path.join(output_folder, f"Tidy_{filename}"), index=False)
            print(f"Saved integrated data for {filename}")
            processed_any = True

    if not processed_any:
        print("Warning: No valid CSV data files were found or processed.")

    return None


def find_intercept(
    subfolder,
    final_folder,
    time_col="Time/sec",
    charge_col="Charge/C",
    conc_col="Concentration",
):
    final_path = os.path.join("Data", subfolder, final_folder)
    os.makedirs(final_path, exist_ok=True)

    all_dfs = []
    for filename in os.listdir(final_path):
        if filename.endswith(".csv") and "Tidy" in filename:
            all_dfs.append(pd.read_csv(os.path.join(final_path, filename)))

    if not all_dfs:
        print("No tidy data found for intercept analysis.")
        return None

    df_all = pd.concat(all_dfs, ignore_index=True)

    time_steps = range(40, 121)
    rows = []
    for target_time in time_steps:
        df_at_time = df_all[np.isclose(df_all[time_col], target_time, atol=0.05)]
        if df_at_time.empty:
            continue

        x_concentration = df_at_time[conc_col].values
        y_charge = df_at_time[charge_col].values
        if len(x_concentration) < 2:
            continue

        slope, intercept, r_value, _, _ = linregress(x_concentration, y_charge)
        rows.append(
            {
                "Time": target_time,
                "Intercept": intercept,
                "Abs_Intercept": abs(intercept),
                "Slope": slope,
                "R_squared": r_value ** 2,
            }
        )

    df_results = pd.DataFrame(rows)
    if df_results.empty:
        print("No valid regression points found for intercept analysis.")
        return None

    best_intercept = df_results.loc[df_results["Abs_Intercept"].idxmin()]
    print("\n" + "=" * 40)
    print("ABSOLUTE INTERCEPT NEAREST ZERO RESULTS")
    print("=" * 40)
    print(f"Optimal Time Found:   {best_intercept['Time']:.2f} seconds")
    print(f"Minimum Intercept:    {best_intercept['Intercept']:.6e}")
    print(f"Sensitivity (Slope):  {best_intercept['Slope']:.6e}")
    print(f"Linearity (R²):       {best_intercept['R_squared']:.5f}")
    print("=" * 40)

    best_r2 = df_results.loc[df_results["R_squared"].idxmax()]
    print("        MAXIMUM R² RESULTS        ")
    print("=" * 40)
    print(f"Optimal Time Found:   {best_r2['Time']:.2f} seconds")
    print(f"Intercept:            {best_r2['Intercept']:.6e}")
    print(f"Sensitivity (Slope):  {best_r2['Slope']:.6e}")
    print(f"Linearity (R²):       {best_r2['R_squared']:.5f}")
    print("=" * 40)

    df_results.to_csv(os.path.join(final_path, "intercept_optimisation.csv"), index=False)
    print(f"Results saved to {final_path}/intercept_optimisation.csv")
    return None
