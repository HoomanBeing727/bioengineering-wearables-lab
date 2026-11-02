#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from matplotlib.widgets import Slider
from scipy.signal import butter, filtfilt, find_peaks


# ======================================================
# User configuration
# Edit this block to choose which experiment folder to process.
# The script will batch-handle all matching ECG csv files in that folder.
# ======================================================

CONFIG = {
    # Process one experiment folder at a time. Paths are relative to the
    # repository root. The default points at the synthetic demo dataset.
    # Real recordings use the same layout, e.g.
    # Data/20260101-Presentation-S02/
    "input_folder": "Data/demo/20260101-VRDemo-S01",
    "output_root": "mental-stress/ecg/output",
    # Only files ending with this suffix are treated as ECG inputs.
    "input_glob_suffix": "_ECG.csv",
    # Set to True only when the ECG polarity is reversed, for example due to swapped electrodes.
    "invert_signal": False,

    # Keep the original sampling and csv-reading assumptions unchanged.
    "fs": 250,
    "skiprows": 2,  # Skip the same header rows used in the notebook/script before.
    "usecol": 1,  # ECG is read from column B (0-based index 1).

    # Segment extraction remains fixed to the original 30-minute logic.
    "start_sec": 60.0,
    "end_sec": None,
    "duration_sec": 1800.0,

    

    # Optional extra text appended to exported file names.
    "extra_tag": "",

    # Initial x-range shown in the interactive inspection window.
    "inspect_view_sec": 60.0,

    # Export RRI png files in parallel with csv export. Interactive plots are
    # off by default so the script also runs on headless machines.
    "export_rri_png": True,
    "show_plots": False,
}


FOLDER_PATTERN = re.compile(
    r"^(?P<date>\d{8})-(?P<experiment_type>.+)-(?P<subject_code>S\d+)$"
)
FILE_PATTERN = re.compile(r"^(?P<name>.+)_(?P<segment_index>\d+)_(?P<signal_type>[^.]+)\.csv$")


# =========================
# Metadata containers
# Store parsed naming information and processing results.
# =========================

@dataclass
class MetaInfo:
    experiment_type: str
    segment_index: int
    subject_id: str
    raw_file_name: str
    fs: int
    segment_start_sec: float
    segment_end_sec: float
    created_at: str


@dataclass
class InputSpec:
    file_path: str
    raw_folder_name: str
    experiment_type: str
    segment_index: int
    subject_id: str


@dataclass
class ProcessResult:
    out_csv: str
    out_png: str
    title: str
    cleaned_ecg: np.ndarray
    peaks: np.ndarray
    fs: int
    time_clean: np.ndarray
    rr_clean: np.ndarray


# =========================
# Raw data loading and fixed-length segment selection
# =========================

def load_ecg_csv(file_path: str, skiprows: int = 2, usecols=(1,), header=None) -> np.ndarray:
    df = pd.read_csv(file_path, skiprows=skiprows, header=header, usecols=list(usecols))
    ecg = df.iloc[:, 0].astype(float).to_numpy()
    return ecg


def select_segment(
    ecg: np.ndarray,
    fs: int,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    duration_sec: float = 1800.0,
) -> tuple[np.ndarray, float, float]:
    n = len(ecg)
    total_sec = n / fs

    if end_sec is None:
        end_sec = start_sec + duration_sec

    start_sec = max(0.0, float(start_sec))
    end_sec = min(float(end_sec), total_sec)

    start_i = int(round(start_sec * fs))
    end_i = int(round(end_sec * fs))

    if end_i <= start_i:
        raise ValueError(f"Invalid segment: start_sec={start_sec}, end_sec={end_sec}")

    seg = ecg[start_i:end_i]
    seg_len_sec = len(seg) / fs

    tol = 1.0 / fs
    if abs(seg_len_sec - duration_sec) > tol:
        raise ValueError(
            f"Segment length is {seg_len_sec:.6f}s, not {duration_sec}s. "
            f"Check start/end. (fs={fs}, tol={tol:.6f}s)"
        )

    return seg, start_sec, end_sec


def maybe_invert_signal(ecg: np.ndarray, invert_signal: bool = False) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=float)
    return -ecg if invert_signal else ecg


# =========================
# Signal preprocessing
# The algorithms and parameters in this section are intentionally unchanged.
# =========================

def remove_baseline_wander_wavelet(ecg_signal: np.ndarray, wavelet: str = "db4", level: int = 5) -> np.ndarray:
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])
    filtered = pywt.waverec(coeffs, wavelet)
    return filtered[: len(ecg_signal)]


def bandpass_filter(ecg_signal: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 1) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, ecg_signal)


# =========================
# R-peak detection and RR cleaning
# The logic here is kept identical to the previous notebook/script behavior.
# =========================

def detect_r_peaks(cleaned_ecg: np.ndarray, fs: int) -> np.ndarray:
    peaks, _ = find_peaks(
        cleaned_ecg,
        prominence=0.3 * np.std(cleaned_ecg),
        distance=int(0.5 * fs),
    )
    return peaks


def compute_rr(peaks: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    if len(peaks) < 3:
        raise ValueError("Too few peaks to compute RR.")
    rr_intervals = np.diff(peaks) / fs
    rr_time = peaks[1:] / fs
    return rr_intervals, rr_time


def clean_rr(rr_val: np.ndarray, rr_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rr_val = np.asarray(rr_val, dtype=float)
    rr_t = np.asarray(rr_t, dtype=float)

    mask1 = (rr_val > 0.4) & (rr_val < 2.0)
    if mask1.sum() < 10:
        raise ValueError("Too few RR left after mask1 (0.4~2.0s).")

    med = np.median(rr_val[mask1])
    mask2 = (rr_val[mask1] > 0.7 * med) & (rr_val[mask1] < 1.3 * med)

    rr_c = rr_val[mask1][mask2]
    t_c = rr_t[mask1][mask2]
    if rr_c.size < 10:
        raise ValueError("Too few RR left after mask2 (median range).")

    rr_diff = np.abs(np.diff(rr_c))
    thr = 0.3 * np.median(rr_c)
    mask3 = np.concatenate(([True], rr_diff < thr))

    rr_clean = rr_c[mask3]
    time_clean = t_c[mask3]
    return rr_clean, time_clean


# =========================
# Export helpers
# Write cleaned RRI series and optional preview plots to disk.
# =========================

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def export_rri_csv(out_csv: str, meta: MetaInfo, time_s: np.ndarray, rri_s: np.ndarray) -> None:
    ensure_dir(os.path.dirname(out_csv))

    header_lines = [
        f"# experiment_type: {meta.experiment_type}",
        f"# segment_index: {meta.segment_index}",
        f"# subject_id: {meta.subject_id}",
        f"# raw_file_name: {meta.raw_file_name}",
        f"# fs_hz: {meta.fs}",
        f"# segment_start_sec: {meta.segment_start_sec}",
        f"# segment_end_sec: {meta.segment_end_sec}",
        f"# segment_duration_sec: {meta.segment_end_sec - meta.segment_start_sec}",
        f"# created_at: {meta.created_at}",
        "#",
    ]

    df = pd.DataFrame(
        {
            "Time(s)": np.asarray(time_s, dtype=float),
            "RR intervals(s)": np.asarray(rri_s, dtype=float),
        }
    )

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n")
        df.to_csv(f, index=False)


def export_rri_png(out_png: str, time_s: np.ndarray, rri_s: np.ndarray, title: str = "") -> None:
    ensure_dir(os.path.dirname(out_png))
    plt.figure(figsize=(12, 3))
    plt.plot(time_s, rri_s, linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("RR intervals (s)")
    plt.xlim(0, max(float(time_s[-1]), 1.0))
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =========================
# Naming-rule parsing
# Folder names provide experiment metadata; file names provide segment numbers.
# =========================

def parse_folder_name(folder_name: str) -> tuple[str, str]:
    match = FOLDER_PATTERN.match(folder_name)
    if not match:
        raise ValueError(
            "Folder name must follow YYYYMMDD-TestType-Sxx: "
            f"{folder_name}"
        )
    experiment_type = match.group("experiment_type").strip()
    subject_id = match.group("subject_code").strip()
    return experiment_type, subject_id


def parse_file_name(file_name: str) -> tuple[int, str]:
    match = FILE_PATTERN.match(file_name)
    if not match:
        raise ValueError(f"File name must follow name_segno_type.csv: {file_name}")

    signal_type = match.group("signal_type").strip().upper()
    if signal_type != "ECG":
        raise ValueError(f"Only ECG csv files are supported: {file_name}")

    return int(match.group("segment_index")), match.group("name").strip()


def collect_input_specs(input_folder: str, suffix: str) -> list[InputSpec]:
    specs: list[InputSpec] = []
    folder_name = os.path.basename(os.path.normpath(input_folder))
    experiment_type, subject_id = parse_folder_name(folder_name)

    # Only scan the selected experiment folder; do not recurse into the whole dataset.
    for file_name in sorted(os.listdir(input_folder)):
        if not file_name.endswith(suffix):
            continue

        file_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(file_path):
            continue

        segment_index, _ = parse_file_name(file_name)

        specs.append(
            InputSpec(
                file_path=file_path,
                raw_folder_name=folder_name,
                experiment_type=experiment_type,
                segment_index=segment_index,
                subject_id=subject_id,
            )
        )

    if not specs:
        raise FileNotFoundError(
            f"No input files ending with {suffix!r} found under {input_folder!r}."
        )

    specs.sort(key=lambda item: (item.raw_folder_name, item.segment_index, item.file_path))
    return specs


# =========================
# Interactive inspection
# Plots are shown only after all exports finish, so batch processing is not blocked midway.
# =========================

def inspect_rpeaks_batch(
    results: list[ProcessResult],
    inspect_view_sec: float = 60.0,
) -> None:
    if not results:
        return

    ncols = 2 if len(results) > 1 else 1
    nrows = math.ceil(len(results) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, max(3.2 * nrows, 3.2)),
        sharex=True,
    )
    axes_arr = np.atleast_1d(axes).ravel()
    max_time = 0.0

    for ax, result in zip(axes_arr, results):
        t = np.arange(len(result.cleaned_ecg)) / result.fs
        max_time = max(max_time, float(t[-1]))
        ax.plot(t, result.cleaned_ecg, linewidth=0.8)
        ax.plot(result.peaks / result.fs, result.cleaned_ecg[result.peaks], "rx", markersize=4)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(result.title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(inspect_view_sec, t[-1]))

    for ax in axes_arr[len(results):]:
        ax.set_visible(False)

    fig.suptitle("R-peak inspection", fontsize=14)

    # One shared slider controls the visible time window for all subplots together.
    window_sec = min(inspect_view_sec, max_time)
    slider_max = max(0.0, max_time - window_sec)
    slider_ax = fig.add_axes([0.12, 0.04, 0.76, 0.035])
    time_slider = Slider(
        ax=slider_ax,
        label="Window start (s)",
        valmin=0.0,
        valmax=slider_max if slider_max > 0 else 1.0,
        valinit=0.0,
    )

    def update_window(start_sec: float) -> None:
        end_sec = min(start_sec + window_sec, max_time)
        for ax in axes_arr[: len(results)]:
            ax.set_xlim(start_sec, end_sec)
        fig.canvas.draw_idle()

    time_slider.on_changed(update_window)

    plt.tight_layout(rect=[0, 0.09, 1, 0.96])
    plt.show(block=True)


# =========================
# Processing pipeline
# One file is processed end-to-end, then main() batches over one experiment folder.
# =========================

def process_one_file(
    spec: InputSpec,
    output_root: str,
    fs: int = 250,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    duration_sec: float = 1800.0,
    skiprows: int = 2,
    usecol: int = 1,
    invert_signal: bool = False,
    extra_tag: str = "",
    export_png: bool = True,
) -> ProcessResult:
    raw = load_ecg_csv(spec.file_path, skiprows=skiprows, usecols=(usecol,))
    raw = maybe_invert_signal(raw, invert_signal=invert_signal)
    seg, seg_start, seg_end = select_segment(
        raw, fs=fs, start_sec=start_sec, end_sec=end_sec, duration_sec=duration_sec
    )

    wavelet_ecg = remove_baseline_wander_wavelet(seg, wavelet="db4", level=5)
    cleaned_ecg = bandpass_filter(wavelet_ecg, lowcut=0.5, highcut=45.0, fs=fs, order=1)
    peaks = detect_r_peaks(cleaned_ecg, fs=fs)
    rr_intervals, rr_time = compute_rr(peaks, fs=fs)
    rr_clean, time_clean = clean_rr(rr_intervals, rr_time)

    meta = MetaInfo(
        experiment_type=spec.experiment_type,
        segment_index=int(spec.segment_index),
        subject_id=str(spec.subject_id),
        raw_file_name=os.path.basename(spec.file_path),
        fs=int(fs),
        segment_start_sec=float(seg_start),
        segment_end_sec=float(seg_end),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )

    # Mirror the raw experiment folder name under output/, then store RRI exports in rri/.
    subject_output_dir = os.path.join(output_root, spec.raw_folder_name, "rri")
    fname = f"RRI_{spec.experiment_type}_{spec.subject_id}_Seg{int(spec.segment_index):02d}"
    if extra_tag:
        fname = f"{fname}_{extra_tag}"

    out_csv = os.path.join(subject_output_dir, f"{fname}.csv")
    out_png = os.path.join(subject_output_dir, f"{fname}.png")
    title = f"R-peaks - {spec.experiment_type} - {spec.subject_id} - Seg{int(spec.segment_index):02d}"

    export_rri_csv(out_csv, meta, time_clean, rr_clean)

    if export_png:
        export_rri_png(
            out_png,
            time_clean,
            rr_clean,
            title=f"RRI (NN) - {spec.experiment_type} - {spec.subject_id} - Seg{int(spec.segment_index):02d}",
        )
    else:
        out_png = ""

    return ProcessResult(
        out_csv=out_csv,
        out_png=out_png,
        title=title,
        cleaned_ecg=cleaned_ecg,
        peaks=peaks,
        fs=fs,
        time_clean=time_clean,
        rr_clean=rr_clean,
    )


def main() -> None:
    c = CONFIG

    specs = collect_input_specs(c["input_folder"], c["input_glob_suffix"])
    results: list[ProcessResult] = []

    print(f"Found {len(specs)} ECG file(s) under {c['input_folder']}.")

    # Export first, then open a single interactive window for visual review.
    for spec in specs:
        result = process_one_file(
            spec=spec,
            output_root=c["output_root"],
            fs=c["fs"],
            start_sec=c["start_sec"],
            end_sec=c["end_sec"],
            duration_sec=c["duration_sec"],
            skiprows=c["skiprows"],
            usecol=c["usecol"],
            invert_signal=c["invert_signal"],
            extra_tag=c["extra_tag"],
            export_png=c["export_rri_png"],
        )
        results.append(result)
        print(f"✅ CSV : {result.out_csv}")
        if result.out_png:
            print(f"✅ PNG : {result.out_png}")

    if c["show_plots"]:
        inspect_rpeaks_batch(results, inspect_view_sec=c["inspect_view_sec"])

    print(f"✅ Done. Exported {len(results)} RRI file(s).")


if __name__ == "__main__":
    main()
