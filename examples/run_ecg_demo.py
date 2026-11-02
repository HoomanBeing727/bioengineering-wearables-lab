#!/usr/bin/env python3
"""Run the ECG demo pipeline against the synthetic dataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "mental-stress" / "ecg"))

import ecg_rri_exporter  # noqa: E402
import export_features  # noqa: E402


ecg_rri_exporter.CONFIG.update(
    {
        "input_folder": "Data/demo/20260101-VR-S01",
        "output_root": "mental-stress/ecg/output",
        "start_sec": 0.0,
        "duration_sec": 300.0,
        "invert_signal": False,
        "show_plots": False,
    }
)
ecg_rri_exporter.main()

export_features.CONFIG.update(
    {
        "experiment_output_dir": "mental-stress/ecg/output/20260101-VR-S01",
        "rmssd_win_sec": 60.0,
        "freq_win_sec": 120.0,
        "time_win_sec": 120.0,
        "feature_win_sec": 120.0,
        "seg_len_sec": 300.0,
        "auto_make_report": True,
    }
)
export_features.main()

print("\nECG demo complete.")
print("RRI files:      mental-stress/ecg/output/20260101-VR-S01/rri/")
print("Feature files:  mental-stress/ecg/output/20260101-VR-S01/features/")
print("Report figure:  mental-stress/ecg/output/20260101-VR-S01/report/")
