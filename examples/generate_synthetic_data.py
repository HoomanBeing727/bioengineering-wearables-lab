#!/usr/bin/env python3
"""Generate small, deterministic synthetic datasets for the demo pipelines.

All files are written under ``Data/`` (gitignored). No real subject or
laboratory data is used.

Run from the repository root:

    python examples/generate_synthetic_data.py
    python examples/generate_synthetic_data.py --force
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"

ECG_FS = 250
BREATH_FS = 50
SEGMENT_SEC = 300
N_SEGMENTS = 4
DEMO_SUBJECT = "S01"
DEMO_FOLDER = "20260101-VR-S01"


def _write_csv_with_metadata(path: Path, df: pd.DataFrame, metadata: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in metadata:
            f.write(f"# {line}\n")
        df.to_csv(f, index=False, header=False)


def make_ecg(seed: int) -> pd.DataFrame:
    """Create a synthetic ECG trace with clear, regular R peaks."""
    rng = np.random.default_rng(seed)
    n = SEGMENT_SEC * ECG_FS
    t = np.arange(n) / ECG_FS

    ecg = (
        0.035 * np.sin(2 * np.pi * 0.33 * t)
        + 0.02 * np.sin(2 * np.pi * 0.08 * t)
        + rng.normal(0, 0.012, n)
    )

    # Build a slightly variable RR series around 1 Hz.
    beat_times = []
    current = 0.5
    while current < SEGMENT_SEC - 0.5:
        beat_times.append(current)
        current += float(np.clip(rng.normal(1.0, 0.05), 0.75, 1.25))

    for beat in beat_times:
        # P wave, QRS complex, T wave.
        for offset, amplitude, width in (
            (-0.16, 0.08, 0.04),
            (0.0, 1.0, 0.016),
            (0.25, 0.18, 0.07),
        ):
            ecg += amplitude * np.exp(-0.5 * ((t - (beat + offset)) / width) ** 2)

    return pd.DataFrame({"Time(s)": t, "ECG": ecg})


def make_breath(seed: int) -> pd.Series:
    """Create a smooth 0.25 Hz respiration-like signal on the 0-255 scale."""
    rng = np.random.default_rng(seed)
    n = SEGMENT_SEC * BREATH_FS
    t = np.arange(n) / BREATH_FS

    # 15 breaths/min, with slow amplitude modulation and sensor noise.
    amp = 34 * (1 + 0.08 * np.sin(2 * np.pi * 0.015 * t))
    signal = (
        128
        + amp * np.sin(2 * np.pi * 0.25 * t)
        + 0.8 * np.sin(2 * np.pi * 0.05 * t)
        + rng.normal(0, 1.2, n)
    )
    return pd.Series(np.clip(np.round(signal), 0, 255).astype(int))


def generate_mental_stress() -> None:
    folder = DATA / "demo" / DEMO_FOLDER
    folder.mkdir(parents=True, exist_ok=True)

    for segment in range(1, N_SEGMENTS + 1):
        ecg = make_ecg(seed=1000 + segment)
        ecg_path = folder / f"{DEMO_SUBJECT}_{segment:02d}_ECG.csv"
        _write_csv_with_metadata(
            ecg_path,
            ecg,
            [
                "Synthetic ECG for pipeline demonstration only",
                f"fs={ECG_FS}, duration={SEGMENT_SEC}s",
            ],
        )

        breath = make_breath(seed=2000 + segment)
        breath_path = folder / f"{DEMO_SUBJECT}_{segment:02d}_Breath.txt"
        breath.to_csv(breath_path, index=False, header=False)

    print(f"Mental-stress demo data: {folder}")


def generate_ion_transport() -> None:
    raw_dir = DATA / "demo_ion" / "demo_ion_Raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    t = np.arange(0, 130.0 + 1e-9, 0.1)
    for conc in (0.1, 0.5, 1.0):
        for replicate in (1.0, 2.0):
            rng = np.random.default_rng(int(conc * 1000) + int(replicate))
            current = (
                2.0e-6 * conc * (1 + 0.03 * np.sin(2 * np.pi * 0.02 * t))
                + 1.0e-8 * np.sin(2 * np.pi * 0.5 * t)
                + rng.normal(0, 2.0e-8, t.size)
            )
            df = pd.DataFrame({"Time/sec": t, "Current/A": current})
            path = raw_dir / f"{conc}mM KCl {replicate:.1f}.txt"
            df.to_csv(path, index=False, float_format="%.10g")

    print(f"Ion-transport demo data: {raw_dir}")


def generate_ocp() -> None:
    raw_dir = DATA / "demo_ocp" / "demo_ocp_Raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # A V-shaped concentration sequence: down to 1 mM, then back up.
    concentrations = [50, 10, 1, 10, 50]
    t = np.arange(0, 60.0 + 1e-9, 0.1)
    for step, conc in enumerate(concentrations, start=1):
        rng = np.random.default_rng(3000 + step)
        potential = (
            0.105
            + 0.045 * np.log10(conc)
            + 0.002 * np.sin(2 * np.pi * 0.05 * t)
            + rng.normal(0, 2.0e-4, t.size)
        )
        df = pd.DataFrame({"Time/sec": t, "Potential/V": potential})
        path = raw_dir / f"{conc}mM KCl 1_{step}.txt"
        df.to_csv(path, index=False, float_format="%.10g")

    print(f"OCP demo data: {raw_dir}")


def generate_randles_sevcik() -> None:
    batches = {
        "Bare_Electrode": 1.0,
        "AuND": 1.4,
    }
    for batch, scale in batches.items():
        raw_dir = DATA / "demo_randles" / "raw" / batch
        raw_dir.mkdir(parents=True, exist_ok=True)

        for electrode in (1, 2):
            for scan_rate in (10, 20, 40):
                rng = np.random.default_rng(4000 + scan_rate + electrode)
                potential = np.linspace(0.1, 0.5, 1000)
                peak_current = (
                    scale * 2.0e-5 * np.sqrt(scan_rate / 10.0)
                )
                current = (
                    peak_current
                    * np.exp(-0.5 * ((potential - 0.30) / 0.045) ** 2)
                    + 1.0e-7
                    + rng.normal(0, 2.0e-8, potential.size)
                )
                df = pd.DataFrame(
                    {"Potential/V": potential, "Current/A": current}
                )
                path = raw_dir / f"{scan_rate} mVs {batch}_{electrode}.txt"
                df.to_csv(path, index=False, float_format="%.10g")

    print(f"Randles-Sevcik demo data: {DATA / 'demo_randles' / 'raw'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing synthetic demo folders",
    )
    args = parser.parse_args()

    demo_paths = [
        DATA / "demo",
        DATA / "demo_ion",
        DATA / "demo_ocp",
        DATA / "demo_randles",
    ]
    existing = [p for p in demo_paths if p.exists()]
    if existing and not args.force:
        print("Refusing to overwrite existing demo data:")
        for path in existing:
            print(f"  {path}")
        print("Re-run with --force to replace it.")
        raise SystemExit(1)

    if args.force:
        for path in demo_paths:
            if path.exists():
                shutil.rmtree(path)

    generate_mental_stress()
    generate_ion_transport()
    generate_ocp()
    generate_randles_sevcik()

    print("\nDone. Next commands:")
    print("  python examples/run_ecg_demo.py")
    print("  python examples/run_respiration_demo.py")
    print("  python examples/run_electrochem_demo.py")


if __name__ == "__main__":
    main()
