#!/usr/bin/env python3
"""Run the respiration demo pipeline against the synthetic dataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "mental-stress" / "respiration"))

import breath_exporter  # noqa: E402
import export_features  # noqa: E402


breath_exporter.CONFIG.update(
    {
        "input_folder": "Data/demo/20260101-VR-S01",
        "output_root": "mental-stress/respiration/output",
        "start_sec": 0.0,
        "duration_sec": 300.0,
        "FS": 50,
        "show_peak_plots": False,
        "show_trough_plots": False,
    }
)
breath_exporter.main()

export_features.CONFIG.update(
    {
        "experiment_output_dir": "mental-stress/respiration/output/20260101-VR-S01",
        "feature_win_sec": 120.0,
        "feature_step_sec": 20.0,
        "seg_len_sec": 300.0,
        "auto_make_report": True,
    }
)
export_features.main()

print("\nRespiration demo complete.")
print("Interval files: mental-stress/respiration/output/20260101-VR-S01/intervals/")
print("Signal files:   mental-stress/respiration/output/20260101-VR-S01/signals/")
print("Feature files:  mental-stress/respiration/output/20260101-VR-S01/features/")
print("Report figure:  mental-stress/respiration/output/20260101-VR-S01/report/")
