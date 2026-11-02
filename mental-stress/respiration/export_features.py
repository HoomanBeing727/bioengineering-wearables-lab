# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import report_plotting

CONFIG = {
    # Process one exported experiment folder at a time. Paths are relative to
    # the repository root. The default points at the synthetic demo dataset.
    "experiment_output_dir": "mental-stress/respiration/output/20260101-VRDemo-S01",

    # Input/output layout inside the experiment output folder.
    "intervals_subdir": "intervals",
    "signals_subdir": "signals",
    "features_subdir": "features",

    # Window settings for feature export.
    "feature_win_sec": 120.0,
    "feature_step_sec": 20.0,

    # Expected segment duration.
    "seg_len_sec": 1800.0,

    # Automatically generate the report figure after feature export.
    "auto_make_report": True,
}

PEAK_FILE_PATTERN = re.compile(
    r"^Peak Intervals_(?P<test>.+)_(?P<subject>S\d+)_Seg(?P<segment>\d{2})\.csv$"
)
TROUGH_FILE_PATTERN = re.compile(
    r"^Trough Intervals_(?P<test>.+)_(?P<subject>S\d+)_Seg(?P<segment>\d{2})\.csv$"
)
SIGNAL_FILE_PATTERN = re.compile(
    r"^Filtered Signal_(?P<test>.+)_(?P<subject>S\d+)_Seg(?P<segment>\d{2})\.csv$"
)


def normalize_test_name(test_name: str) -> str:
    key = re.sub(r"\s+", " ", str(test_name).strip()).lower()
    aliases = {
        "gaming": "gaming",
        "gaming test": "gaming",
        "vr": "vr",
        "vr test": "vr",
    }
    return aliases.get(key, key)


def get_test_protocol(test_name: str) -> dict[str, object]:
    key = normalize_test_name(test_name)
    protocols = {
        "gaming": {
            "segment_count": 4,
            "stress_spans": [(1, 0.0, 1800.0)],
        },
        "vr": {
            "segment_count": 4,
            "stress_spans": [(1, 0.0, 900.0)], # 900 seconds of activity
        }
    }
    return protocols.get(key, {"segment_count": None, "stress_spans": []})


def classify_window_state(
    segment_index: int,
    window_start_in_segment: float,
    window_end_in_segment: float,
    stress_spans: list[tuple[int, float, float]],
) -> str:
    """
    To handle state conflict when applying sliding window average and window is in intersection between rest and stress states 
    If sliding window overlaps with stress state, it defaults to stress state to capture transitional data
    """
    for stress_segment_zero_based, stress_start, stress_end in stress_spans:
        if segment_index - 1 != stress_segment_zero_based:
            continue
        overlap_start = max(window_start_in_segment, stress_start)
        overlap_end = min(window_end_in_segment, stress_end)
        if overlap_end > overlap_start:
            return "stress"
    return "rest"


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def read_interval_csv_with_meta(path: str) -> tuple[dict[str, str], np.ndarray, np.ndarray]:
    meta: dict[str, str] = {}
    header_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                header_lines += 1
                m = re.match(r"#\s*([^:]+)\s*:\s*(.*)\s*$", line)
                if m:
                    meta[m.group(1).strip()] = m.group(2).strip()
                continue
            break

    df = pd.read_csv(path, skiprows=header_lines)
    tcol = [c for c in df.columns if "Time" in c][0]
    icol = [c for c in df.columns if "Breath" in c or "Interval" in c][0]
    return meta, df[tcol].to_numpy(dtype=float), df[icol].to_numpy(dtype=float)


def read_signal_csv_with_meta(path: str) -> tuple[dict[str, str], np.ndarray, np.ndarray]:
    meta: dict[str, str] = {}
    header_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                header_lines += 1
                m = re.match(r"#\s*([^:]+)\s*:\s*(.*)\s*$", line)
                if m:
                    meta[m.group(1).strip()] = m.group(2).strip()
                continue
            break

    df = pd.read_csv(path, skiprows=header_lines)
    tcol = [c for c in df.columns if "Time" in c][0]
    acol = [c for c in df.columns if "Amplitude" in c][0]
    return meta, df[tcol].to_numpy(dtype=float), df[acol].to_numpy(dtype=float)


# ========================
# Interval based features
# ========================

def interval_to_rate(interval_s: np.ndarray) -> np.ndarray:
    interval_s = np.asarray(interval_s, dtype=float)
    return 60.0 / interval_s


def brv_rmssd(interval_s: np.ndarray) -> float:
    interval_s = np.asarray(interval_s, dtype=float)
    interval_s = interval_s[np.isfinite(interval_s)]
    if interval_s.size < 3:
        return np.nan
    d = np.diff(interval_s)
    return float(np.sqrt(np.mean(d ** 2)))


def std_dev(interval_s: np.ndarray) -> float:
    # Use ddof=1 for sample standard deviation (consistent with HRV analysis)
    return float(np.std(interval_s, ddof=1))


def variation_coefficient(interval_s: np.ndarray) -> float:
    m = np.mean(interval_s)
    if m == 0:
        return np.nan
    return float(np.std(interval_s, ddof=1) / abs(m))


# =========================
# Signal Based Features
# =========================

def amp_mean(amplitude: np.ndarray) -> float:
    amplitude = np.asarray(amplitude, dtype=float)
    return float(np.mean(amplitude))


def amp_std(amplitude: np.ndarray) -> float:
    amplitude = np.asarray(amplitude, dtype=float)
    return float(np.std(amplitude, ddof=1))


def amp_cv(amplitude: np.ndarray) -> float:
    m = np.mean(amplitude)
    if m == 0:
        return np.nan
    return float(np.std(amplitude, ddof=1) / m)


def detect_peaks_troughs_amplitudes(
    signal: np.ndarray,
    fs: int = 50,
    prominence_factor: float = 0.3,
    distance: int = 40
):
    """
    For filtered signal, re-calculate the inspiration (peak -> trough) and expiration
    (trough -> next peak). Return the ratio for that interval

    Returns:
        peak_times: relative time of each peak (s) within the window
        trough_times: relative time of each trough (s) within the window
        peak_amplitudes: amplitude at each peak
        trough_amplitues: amplitude at each trough
    """

    std = np.std(signal)
    prominence = prominence_factor * std

    peaks, _ = find_peaks(signal, prominence=prominence, distance=distance)
    troughs, _ = find_peaks(-signal, prominence=prominence, distance=distance)
    
    peak_times = peaks / fs
    trough_times = troughs / fs
    peak_amplitudes = signal[peaks]
    trough_amplitudes = signal[troughs]

    return peak_times, trough_times, peak_amplitudes, trough_amplitudes


def ie_amp_ratio(peak_amplitudes: np.ndarray, trough_amplitudes: np.ndarray) -> np.ndarray:
    """Inspiration:Expiration Amplitude Ratio per breath cycle"""
    # Broadcasting mismatch indicates artifact data
    # Pairing done to allow computation but without labelling or data deletion
    min_len = min(len(peak_amplitudes), len(trough_amplitudes))
    return peak_amplitudes[:min_len] / np.abs(trough_amplitudes[:min_len])


# ==================
# File Parsing
# ==================

def parse_interval_file_name(file_name: str, pattern: "re.Pattern") -> tuple[str, str, int]:
    m = pattern.match(file_name)
    if not m:
        raise ValueError(f"Unexpected Interval file name: {file_name}")
    return m.group("test"), m.group("subject"), int(m.group("segment"))


def parse_signal_file_name(file_name: str) -> tuple[str, str, int]:
    m = SIGNAL_FILE_PATTERN.match(file_name)
    if not m:
        raise ValueError(f"Unexpected SIGNAL file name: {file_name}")
    return m.group("test"), m.group("subject"), int(m.group("segment"))


# ===================
# Segment Loading
# ===================

def load_experiment_segments(experiment_dir: str, cfg: dict) -> tuple[str, str, list[tuple[int, dict[str, str], np.ndarray, np.ndarray]]]:
    """
    Load all peak intervals, trough intervals and filtered signals
    Return (test_name, subject_id, segments)

    Segments is a tuple:
    (segment_index, meta, peak_times, peak_intervals,
    trough_times, trough_intervals, signal_times, signal_amplitudes)

    """
    intervals_dir = os.path.join(experiment_dir, cfg["intervals_subdir"])
    signals_dir = os.path.join(experiment_dir, cfg["signals_subdir"])

    segments = []
    test_name: str | None = None
    subject_id: str | None = None

    # Find all peak interval files
    for file_name in sorted(os.listdir(intervals_dir)):
        if not file_name.endswith(".csv"):
            continue

        # Parse Peak File
        if not PEAK_FILE_PATTERN.match(file_name):
            continue
        peak_test, peak_subject, segment_index = parse_interval_file_name(file_name, PEAK_FILE_PATTERN)

        # Find matching trough file
        trough_file = f"Trough Intervals_{peak_test}_{peak_subject}_Seg{segment_index:02d}.csv"
        trough_path = os.path.join(intervals_dir, trough_file)
        if not os.path.exists(trough_path):
            raise FileNotFoundError(f"Missing trough file: {trough_path}")

        # Find matching signal file
        signal_file = f"Filtered Signal_{peak_test}_{peak_subject}_Seg{segment_index:02d}.csv"
        signal_path = os.path.join(signals_dir, signal_file)
        if not os.path.exists(signal_path):
            raise FileNotFoundError(f"Missing signal file: {signal_path}")

        # Load data
        peak_meta, peak_times, peak_intervals = read_interval_csv_with_meta(
            os.path.join(intervals_dir, file_name)
        )

        trough_meta, trough_times, trough_intervals = read_interval_csv_with_meta(trough_path)

        signal_meta, signal_times, signal_amplitude = read_signal_csv_with_meta(signal_path)

        if test_name is None:
            test_name = peak_test
            subject_id = peak_subject
        if peak_test != test_name or peak_subject != subject_id:
            raise ValueError("One experiment folder should only contain one test and subject")

        segments.append((segment_index, peak_meta,
                         peak_times, peak_intervals,
                         trough_times, trough_intervals,
                         signal_times, signal_amplitude,))

    if not segments:
        raise FileNotFoundError(f"No interval files found in {intervals_dir}")

    segments.sort(key=lambda item: item[0])

    protocol = get_test_protocol(test_name)
    expected_segment_count = protocol.get("segment_count")

    if expected_segment_count and len(segments) != expected_segment_count:
        raise ValueError(f"Test {test_name!r} expects {expected_segment_count} segments, found {len(segments)}")

    return test_name, subject_id, segments


# ====================
# Build Breath Series
# ====================
def build_breath_series(
    test_name: str,
    subject_id: str,
    segments: list[tuple[int, dict[str, str], np.ndarray, np.ndarray]],
    seg_len_sec: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for segment_index, _, peak_times, peak_intervals, trough_times, trough_intervals, sig_t, sig_a in segments:
        seg_offset = (segment_index - 1) * seg_len_sec
        br_peak = interval_to_rate(peak_intervals)
        br_trough = interval_to_rate(trough_intervals)

        for rel_t, pi, ti, brp, brt in zip(peak_times, peak_intervals,
                                           trough_intervals, br_peak, br_trough):
            rows.append(
                {
                    "subject": subject_id,
                    "test": test_name,
                    "segment": f"Seg{segment_index:02d}",
                    "segment_index": segment_index,
                    "relative_time_sec": float(rel_t),
                    "absolute_time_sec": float(rel_t + seg_offset),
                    "peak_interval_s": float(pi),
                    "trough_interval_s": float(ti),
                    "BR_peak_bpm": float(brp),
                    "BR_trough_bpm": float(brt),
                })
    return pd.DataFrame(rows)


# ===========================
# Compute Metric Series
# ===========================

def compute_metric_series(
    test_name: str,
    subject_id: str,
    segments: list,
    seg_len_sec: float,
    cfg: dict,
) -> pd.DataFrame:
    """
    Sliding window feature extraction
    """
    rows = []
    protocol = get_test_protocol(test_name)
    stress_spans = protocol["stress_spans"]
    win_sec = float(cfg["feature_win_sec"])
    step_sec = float(cfg["feature_step_sec"])

    for segment_index, meta, peak_times, peak_intervals, trough_times, trough_intervals, signal_times, signal_amplitude in segments:
        seg_start = (segment_index - 1) * seg_len_sec
        seg_end = segment_index * seg_len_sec
        t_abs_peak = peak_times + seg_start
        t_abs_trough = trough_times + seg_start
        fs = float(meta.get("fs_hz", 50))

        starts = np.arange(seg_start, seg_end - win_sec + 1e-9, step_sec)
        for start_sec in starts:
            end_sec = start_sec + win_sec
            center_sec = start_sec + win_sec / 2.0
            state = classify_window_state(
                segment_index,
                start_sec - seg_start,
                end_sec - seg_start,
                stress_spans
            )

            # --- Interval Based Features ---
            metrics_specs = [
                ("BR_mean_peak", peak_intervals, t_abs_peak),
                ("BR_mean_trough", trough_intervals, t_abs_trough),
                ("BRV_RMSSD_peak", peak_intervals, t_abs_peak),
                ("BRV_RMSSD_trough", trough_intervals, t_abs_trough),
                ("BRV_SD_peak", peak_intervals, t_abs_peak),
                ("BRV_SD_trough", trough_intervals, t_abs_trough),
                ("BRV_CV_peak", peak_intervals, t_abs_peak),
                ("BRV_CV_trough", trough_intervals, t_abs_trough),
            ]
            
            for metric_name, intervals, t_abs in metrics_specs:
                m = (t_abs >= start_sec) & (t_abs < end_sec)
                iv = intervals[m]
                if iv.size < 3:
                    continue

                if "BR_mean" in metric_name:
                    value = np.mean(interval_to_rate(iv))
                elif "BRV_RMSSD" in metric_name:
                    value = brv_rmssd(iv)
                elif "SD" in metric_name:
                    value = std_dev(iv)
                elif "CV" in metric_name:
                    value = variation_coefficient(iv)
                else:
                    continue

                rows.append(
                    {
                        "subject": subject_id,
                        "test": test_name,
                        "segment": f"Seg{segment_index:02d}",
                        "segment_index": segment_index,
                        "metric": metric_name,
                        "window_start_sec": float(start_sec),
                        "window_end_sec": float(end_sec),
                        "window_center_sec": float(center_sec),
                        "window_size_sec": float(win_sec),
                        "state": state,
                        "value": float(value) if np.isfinite(value) else np.nan,
                    })

            # Signal Based Features (Amplitude)
            m_sig = (signal_times + seg_start >= start_sec) & (signal_times + seg_start < end_sec)
            sig = signal_amplitude[m_sig]
            
            if sig.size >= 50:
                for metric_name, func in [
                    ("AMP_mean", amp_mean),
                    ("AMP_std", amp_std),
                    ("AMP_CV", amp_cv),
                ]:
                    value = func(sig)
                    rows.append({
                        "subject": subject_id,
                        "test": test_name,
                        "segment": f"Seg{segment_index:02d}",
                        "segment_index": segment_index,
                        "metric": metric_name,
                        "window_start_sec": float(start_sec),
                        "window_end_sec": float(end_sec),
                        "window_center_sec": float(center_sec),
                        "window_size_sec": float(win_sec),
                        "state": state,
                        "value": float(value) if np.isfinite(value) else np.nan,
                    })

                # Inspiration Expiration Ratio
                peak_times_win, trough_times_win, peak_amp, trough_amp = detect_peaks_troughs_amplitudes(sig, fs)

                if peak_amp.size >= 3 and trough_amp.size >= 3:
                    ie_ratios = ie_amp_ratio(peak_amp, trough_amp)
                    ie_ratios = ie_ratios[np.isfinite(ie_ratios)]
                    if ie_ratios.size >= 3:
                        for metric_name, func in [
                            ("IE_amp_ratio_mean", np.mean),
                            ("IE_amp_ratio_std", np.std)
                        ]:
                            value = float(func(ie_ratios))
                            rows.append({
                                "subject": subject_id,
                                "test": test_name,
                                "segment": f"Seg{segment_index:02d}",
                                "segment_index": segment_index,
                                "metric": metric_name,
                                "window_start_sec": float(start_sec),
                                "window_end_sec": float(end_sec),
                                "window_center_sec": float(center_sec),
                                "window_size_sec": float(win_sec),
                                "state": state,
                                "value": float(value) if np.isfinite(value) else np.nan,
                            })

    return pd.DataFrame(rows)


# ============================
# Compute Wide Feature Table
# ============================

def compute_wide_feature_table(
    test_name: str,
    subject_id: str,
    segments: list,
    seg_len_sec: float,
    cfg: dict,
) -> pd.DataFrame:

    rows = []
    protocol = get_test_protocol(test_name)
    stress_spans = protocol["stress_spans"]
    win_sec = float(cfg["feature_win_sec"])
    step_sec = float(cfg["feature_step_sec"])

    for segment_index, meta, peak_times, peak_intervals, trough_times, trough_intervals, signal_times, signal_amplitude in segments:
        seg_start = (segment_index - 1) * seg_len_sec
        seg_end = segment_index * seg_len_sec
        t_abs_peak = peak_times + seg_start
        t_abs_trough = trough_times + seg_start
        fs = float(meta.get("fs_hz", 50))

        starts = np.arange(seg_start, seg_end - win_sec + 1e-9, step_sec)
        for start_sec in starts:
            end_sec = start_sec + win_sec
            center_sec = start_sec + win_sec / 2.0
            state = classify_window_state(
                segment_index,
                start_sec - seg_start,
                end_sec - seg_start,
                stress_spans
            )

            row = {
                "subject": subject_id,
                "test": test_name,
                "segment": f"Seg{segment_index:02d}",
                "segment_index": segment_index,
                "t_start": float(start_sec),
                "t_end": float(end_sec),
                "t_center": float(center_sec),
                "state": state,
            }

            # Interval Based Features
            for prefix, intervals, t_abs in [
                ("peak", peak_intervals, t_abs_peak),
                ("trough", trough_intervals, t_abs_trough),
            ]:
                m = (t_abs >= start_sec) & (t_abs < end_sec)
                iv = intervals[m]
                if iv.size >= 3:
                    br = interval_to_rate(iv)
                    row[f"BR_mean_{prefix}"] = np.mean(br)
                    row[f"BRV_RMSSD_{prefix}"] = brv_rmssd(iv)
                    row[f"BRV_SD_{prefix}"] = std_dev(iv)
                    row[f"BRV_CV_{prefix}"] = variation_coefficient(iv)
                else:
                    row[f"BR_mean_{prefix}"] = np.nan
                    row[f"BRV_RMSSD_{prefix}"] = np.nan
                    row[f"BRV_SD_{prefix}"] = np.nan
                    row[f"BRV_CV_{prefix}"] = np.nan

            # Signal Based Features
            m_sig = (signal_times + seg_start >= start_sec) & (signal_times + seg_start < end_sec)
            sig = signal_amplitude[m_sig]

            row["AMP_mean"] = np.nan
            row["AMP_std"] = np.nan
            row["AMP_CV"] = np.nan
            row["IE_amp_ratio_mean"] = np.nan
            row["IE_amp_ratio_std"] = np.nan

            if sig.size >= 50:
                row["AMP_mean"] = amp_mean(sig)
                row["AMP_std"] = amp_std(sig)
                row["AMP_CV"] = amp_cv(sig)

                _, _, peak_amp, trough_amp = detect_peaks_troughs_amplitudes(sig, fs)

                if peak_amp.size >= 3 and trough_amp.size >= 3:
                    ie_ratios = ie_amp_ratio(peak_amp, trough_amp)
                    ie_ratios = ie_ratios[np.isfinite(ie_ratios)]

                    if ie_ratios.size >= 3:
                        row["IE_amp_ratio_mean"] = np.mean(ie_ratios)
                        row["IE_amp_ratio_std"] = np.std(ie_ratios)

            rows.append(row)

    return pd.DataFrame(rows)


# ================
# Main
# ================

def main() -> None:
    cfg = CONFIG
    experiment_dir = cfg["experiment_output_dir"]
    features_dir = os.path.join(experiment_dir, cfg["features_subdir"])
    ensure_dir(features_dir)

    test_name, subject_id, segments = load_experiment_segments(experiment_dir, cfg)
    seg_len_sec = float(cfg["seg_len_sec"])

    breath_df = build_breath_series(test_name, subject_id, segments, seg_len_sec)
    metric_df = compute_metric_series(test_name, subject_id, segments, seg_len_sec, cfg)
    wide_df = compute_wide_feature_table(test_name, subject_id, segments, seg_len_sec, cfg)

    breath_path = os.path.join(features_dir, f"BreathSeries_{test_name}_{subject_id}.csv")
    metric_path = os.path.join(features_dir, f"MetricSeries_{test_name}_{subject_id}.csv")
    wide_path = os.path.join(features_dir, f"Features_{test_name}_{subject_id}.csv")

    # sort metric series as seen in ecg processing and needed in report_plotting.py
    metric_df = metric_df.sort_values(["metric", "segment_index", "window_center_sec"]).reset_index(drop=True)
    
    
    breath_df.to_csv(breath_path, index=False)
    metric_df.to_csv(metric_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    print("Saved:", breath_path)
    print("Saved:", metric_path)
    print("Saved:", wide_path)

    if bool(cfg.get("auto_make_report", True)):
        report_plotting.CONFIG["experiment_output_dir"] = experiment_dir
        report_plotting.CONFIG["test_name"] = test_name
        report_plotting.CONFIG["subject_id"] = subject_id
        report_plotting.CONFIG["seg_len_sec"] = seg_len_sec
        report_plotting.main()


if __name__ == "__main__":
    main()