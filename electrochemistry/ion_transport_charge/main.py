"""Run the ion-transport charge analysis pipeline for one experiment folder.

The input layout is:

    Data/<experiment>/
        <experiment>_Psess/   # optional PalmSens .pssession files
        <experiment>_Raw/     # converted CSVs or direct .txt exports
        <experiment>_Tidy/    # generated tidy CSVs
        <experiment>_Final/   # generated charge/intercept results

Run from the repository root:

    python electrochemistry/ion_transport_charge/main.py
"""

import re
import sys
from pathlib import Path

import handle_data
import process_data


# Matches names such as:
#   0.1mM KCl 1.0.csv
#   0.1 mM KCl 1.0.txt
FILENAME_PATTERN = re.compile(
    r"^(?P<concentration>\d+(?:\.\d+)?)"
    r"\s*"                              # optional space (PalmSens has no space; CHI has space)
    r"(?P<units>[a-zA-Zμ]+)"
    r"(?:\s+(?P<name>[a-zA-Z]\w*))?"    # optional analyte name, e.g. KCl
    r"\s+"
    r"(?P<id>\d+(?:\.\d+)?)"
    r"\.(?:csv|txt)$"                   # PalmSens -> .csv; CHI -> .txt
)


def main(subfolder: str = "demo_ion") -> None:
    base = Path("Data", subfolder)
    if not base.is_dir():
        print(f"Error: input folder '{base}' does not exist.")
        print("Run examples/generate_synthetic_data.py from the repository root first.")
        return

    pssess_data = f"{subfolder}_Psess"
    raw_data = f"{subfolder}_Raw"
    tidy_data = f"{subfolder}_Tidy"
    final_data = f"{subfolder}_Final"

    pssess_path = base / pssess_data
    if pssess_path.is_dir():
        import convert

        filetype = ".csv"
        print(f"Found {pssess_path}; converting .pssession files...")
        convert.convert_pssession_to_csv(subfolder, pssess_data, raw_data)
        print(f"Converted .pssession files to .csv in {raw_data}")
    else:
        filetype = ".txt"
        print(f"No {pssess_data} folder found; using direct .txt exports.")

    handle_data.format_data(subfolder, raw_data, tidy_data, FILENAME_PATTERN, filetype)
    process_data.integrate_data(
        subfolder,
        tidy_data,
        final_data,
        time_col="Time/sec",
        current_col="Current/A",
        conc_col="Concentration",
    )
    process_data.find_intercept(
        subfolder,
        final_data,
        time_col="Time/sec",
        charge_col="Charge/C",
        conc_col="Concentration",
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_ion")
