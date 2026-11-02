#!/usr/bin/env python3
"""Run the electrochemistry demo pipelines against the synthetic dataset."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def run(*args: str) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, check=True)


run(sys.executable, "electrochemistry/ion_transport_charge/main.py", "demo_ion")
run(sys.executable, "electrochemistry/open_circuit_potential/main.py", "demo_ocp")

batches = {
    "Bare_Electrode": "Bare_Electrode_Tidy",
    "AuND": "AuND_Tidy",
}
for raw_batch, tidy_batch in batches.items():
    run(
        sys.executable,
        "electrochemistry/randles_sevcik/handle_data.py",
        f"Data/demo_randles/raw/{raw_batch}",
        f"Data/demo_randles/tidy/{tidy_batch}",
    )
    run(
        sys.executable,
        "electrochemistry/randles_sevcik/process_data.py",
        f"Data/demo_randles/tidy/{tidy_batch}",
    )

print("\nElectrochemistry demo complete.")
print("Ion transport:  Data/demo_ion/demo_ion_Final/")
print("OCP:            Data/demo_ocp/demo_ocp_Final/")
print("Randles-Sevcik: Data/demo_randles/tidy/")
