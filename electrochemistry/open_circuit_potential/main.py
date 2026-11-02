"""Run the open-circuit-potential (OCP) analysis pipeline for one experiment.

The input layout is:

    Data/<experiment>/
        <experiment>_Psess/   # optional PalmSens .pssession files
        <experiment>_Raw/     # converted CSVs or direct .txt exports
        <experiment>_Tidy/    # generated tidy CSVs
        <experiment>_Final/   # generated summaries and slopes

Run from the repository root:

    python electrochemistry/open_circuit_potential/main.py
"""

import re
import sys
from pathlib import Path

import handle_data
import process_data


# Matches names such as:
#   0.1mM KCl 1.0_2.csv
#   0.1mM KCl 1.0 2.txt
FILENAME_PATTERN = re.compile(
    r"^(?P<concentration>\d+(?:\.\d+)?)"
    r"\s*"
    r"(?P<units>[a-zA-Zμ]+)"
    r"(?:\s+(?P<name>[a-zA-Z]\w*))?"
    r"\s+"
    r"(?P<id>\d+(?:\.\d+)?)"
    r"[_\s]+"
    r"(?P<step>\d+)"
    r"\.(?:csv|txt)$"
)


def main(
    subfolder: str = "demo_ocp",
    channels: int = 1,
    concentrations=(150, 100, 50),
    sep_direction: bool = True,
    window_sec: float = 30,
    sample_int: float = 0.1,
) -> None:
    base = Path("Data", subfolder)
    if not base.is_dir():
        print(f"Error: input folder '{base}' does not exist.")
        print("Run examples/generate_synthetic_data.py from the repository root first.")
        return

    pssess_data = f"{subfolder}_Psess"
    raw_data = f"{subfolder}_Raw"
    tidy_data = f"{subfolder}_Tidy"
    summary_data = f"{subfolder}_Final"

    if channels > 1:
        process_data.process_4channel_ocp(
            subfolder, raw_data, summary_data, channels, list(concentrations)
        )
        return

    pssess_path = base / pssess_data
    if pssess_path.is_dir():
        import convert

        filetype = ".csv"
        print(f"Found {pssess_path} - converting .pssession files...")
        convert.convert_pssession_to_csv(subfolder, pssess_data, raw_data)
    else:
        filetype = ".txt"
        print(f"No {pssess_data} folder - using direct .txt files")

    window = int(round(window_sec / sample_int))
    handle_data.format_data(subfolder, raw_data, tidy_data, FILENAME_PATTERN, filetype)
    process_data.compute_ocp_summary(
        subfolder, tidy_data, summary_data, sep_direction, window
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_ocp")
