# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from scipy.signal import detrend, welch

import report_plotting


CONFIG = {
    # Process one exported experiment folder at a time. Paths are relative to
    # the repository root. The default points at the synthetic demo dataset.
    "experiment_output_dir": "mental-stress/ecg/output/20260101-VRDemo-S01",

    # Input/output layout inside the experiment output folder.
    "rri_subdir": "rri",
    "features_subdir": "features",

    # Window settings for plot-ready metric series.
    "rmssd_win_sec": 120.0,
    "rmssd_step_sec": 20.0,
    "freq_win_sec": 300.0,
    "freq_step_sec": 20.0,
    "time_win_sec": 300.0,
    "time_step_sec": 20.0,

    # Window settings for wide feature export.
    "feature_win_sec": 300.0,
    "feature_step_sec": 20.0,

    # Frequency-domain parameters.
    "freq_interp_fs": 4.0,
    "freq_nperseg": 256,
    "freq_detrend": True,
    "band_vlf": (0.0033, 0.04),
    "band_lf": (0.04, 0.15),
    "band_hf": (0.15, 0.40),

    # pNN50 / SI parameters.
    "pnn_threshold_ms": 50.0,
    "si_bin_ms": 50.0,
    "si_range_percentiles": (5.0, 95.0),

    # Expected segment duration.
    "seg_len_sec": 1800.0,

    # Automatically generate the report figure after feature export.
    "auto_make_report": True,
}


RRI_FILE_PATTERN = re.compile(
    r"^RRI_(?P<test>.+)_(?P<subject>S\d+)_Seg(?P<segment>\d{2})\.csv$"
)


def normalize_test_name(test_name: str) -> str:
    key = re.sub(r"\s+", " ", str(test_name).strip()).lower()
    aliases = {
        "gaming": "gaming",
        "gaming test": "gaming",
        "vr": "vr",
        "vr test": "vr",
        "presentation": "presentation",
        "presentation test": "presentation",
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
            "stress_spans": [(1, 0.0, 900.0)],
        },
        "presentation": {
            "segment_count": 5,
            "stress_spans": [(2, 0.0, 1800.0)],
        },
    }
    return protocols.get(key, {"segment_count": None, "stress_spans": []})


def classify_window_state(
    segment_index: int,
    window_start_in_segment: float,
    window_end_in_segment: float,
    stress_spans: list[tuple[int, float, float]],
) -> str:
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


def read_rri_csv_with_meta(path: str) -> tuple[dict[str, str], np.ndarray, np.ndarray]:
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
    rcol = [c for c in df.columns if "RR" in c][0]
    return meta, df[tcol].to_numpy(dtype=float), df[rcol].to_numpy(dtype=float)


def rr_to_hr(rr_s: np.ndarray) -> np.ndarray:
    rr_s = np.asarray(rr_s, dtype=float)
    return 60.0 / rr_s


def rmssd(rr_s: np.ndarray) -> float:
    rr_s = np.asarray(rr_s, dtype=float)
    rr_s = rr_s[np.isfinite(rr_s)]
    if rr_s.size < 3:
        return np.nan
    d = np.diff(rr_s)
    return float(np.sqrt(np.mean(d ** 2)))


def pnn50(rr_s: np.ndarray, threshold_ms: float = 50.0) -> float:
    rr_s = np.asarray(rr_s, dtype=float)
    rr_s = rr_s[np.isfinite(rr_s)]
    if rr_s.size < 3:
        return np.nan
    d_ms = np.abs(np.diff(rr_s)) * 1000.0
    return float(100.0 * np.mean(d_ms > float(threshold_ms)))


def poincare_sd1_sd2(rr_s: np.ndarray) -> tuple[float, float]:
    rr_s = np.asarray(rr_s, dtype=float)
    rr_s = rr_s[np.isfinite(rr_s)]
    if rr_s.size < 3:
        return np.nan, np.nan
    d = np.diff(rr_s)
    sdnn_v = np.std(rr_s, ddof=1)
    sdd_v = np.std(d, ddof=1)
    sd1 = np.sqrt(0.5) * sdd_v
    sd2_sq = 2.0 * (sdnn_v ** 2) - 0.5 * (sdd_v ** 2)
    sd2 = np.sqrt(sd2_sq) if sd2_sq > 0 else np.nan
    return float(sd1), float(sd2)


def _bandpower_from_psd(f: np.ndarray, pxx: np.ndarray, band: tuple[float, float]) -> float:
    lo, hi = band
    m = (f >= lo) & (f < hi) & np.isfinite(pxx)
    if np.sum(m) < 2:
        return np.nan
    return float(np.trapz(pxx[m], f[m]))


def compute_freq_metrics_window(
    t_rr: np.ndarray,
    rr: np.ndarray,
    win_start: float,
    win_end: float,
    interp_fs: float,
    nperseg: int,
    detrend_flag: bool,
    band_vlf: tuple[float, float],
    band_lf: tuple[float, float],
    band_hf: tuple[float, float],
) -> tuple[float, float, float, float]:
    m = (t_rr >= win_start) & (t_rr < win_end) & np.isfinite(rr) & np.isfinite(t_rr)
    if np.sum(m) < 10:
        return np.nan, np.nan, np.nan, np.nan

    tw = np.asarray(t_rr[m], dtype=float)
    rrw = np.asarray(rr[m], dtype=float)
    order = np.argsort(tw)
    tw = tw[order]
    rrw = rrw[order]

    _, unique_idx = np.unique(tw, return_index=True)
    tw = tw[unique_idx]
    rrw = rrw[unique_idx]
    if tw.size < 10:
        return np.nan, np.nan, np.nan, np.nan

    dt = 1.0 / float(interp_fs)
    t_grid = np.arange(win_start, win_end + 1e-9, dt)
    if t_grid.size < 32:
        return np.nan, np.nan, np.nan, np.nan

    rr_grid = np.interp(t_grid, tw, rrw)
    x = rr_grid - np.mean(rr_grid)
    if detrend_flag:
        x = detrend(x, type="linear")

    nper = int(min(max(32, nperseg), x.size))
    f, pxx = welch(
        x,
        fs=float(interp_fs),
        nperseg=nper,
        noverlap=int(0.5 * nper),
        detrend=False,
        scaling="density",
    )

    vlf = _bandpower_from_psd(f, pxx, band_vlf)
    lf = _bandpower_from_psd(f, pxx, band_lf)
    hf = _bandpower_from_psd(f, pxx, band_hf)
    lfhf = (lf / hf) if (np.isfinite(lf) and np.isfinite(hf) and hf > 0) else np.nan
    return vlf, lf, hf, lfhf


def baevsky_stress_index(
    rr_s: np.ndarray,
    bin_ms: float = 50.0,
    range_percentiles: tuple[float, float] = (5.0, 95.0),
) -> float:
    rr_s = np.asarray(rr_s, dtype=float)
    rr_s = rr_s[np.isfinite(rr_s)]
    if rr_s.size < 10:
        return np.nan

    bin_s = float(bin_ms) / 1000.0
    if bin_s <= 0:
        return np.nan

    p_lo, p_hi = range_percentiles
    rmin, rmax = np.percentile(rr_s, [p_lo, p_hi])
    rmin, rmax = float(rmin), float(rmax)
    if not np.isfinite(rmin) or not np.isfinite(rmax) or rmax <= rmin:
        return np.nan

    nbins = int(np.ceil((rmax - rmin) / bin_s))
    nbins = max(1, nbins)
    edges = rmin + bin_s * np.arange(nbins + 1)
    hist, edges = np.histogram(rr_s, bins=edges)
    if hist.size == 0 or np.max(hist) <= 0:
        return np.nan

    idx = int(np.argmax(hist))
    mo = 0.5 * (edges[idx] + edges[idx + 1])
    amo = 100.0 * float(hist[idx]) / float(rr_s.size)
    mxdmn = rmax - rmin
    if mo <= 0 or mxdmn <= 0:
        return np.nan

    return float(amo / (2.0 * mo * mxdmn))


def parse_rri_file_name(file_name: str) -> tuple[str, str, int]:
    m = RRI_FILE_PATTERN.match(file_name)
    if not m:
        raise ValueError(f"Unexpected RRI file name: {file_name}")
    return m.group("test"), m.group("subject"), int(m.group("segment"))


def load_experiment_segments(rri_dir: str) -> tuple[str, str, list[tuple[int, dict[str, str], np.ndarray, np.ndarray]]]:
    segments: list[tuple[int, dict[str, str], np.ndarray, np.ndarray]] = []
    test_name: str | None = None
    subject_id: str | None = None

    for file_name in sorted(os.listdir(rri_dir)):
        if not file_name.endswith(".csv"):
            continue
        current_test, current_subject, segment_index = parse_rri_file_name(file_name)
        meta, t, rr = read_rri_csv_with_meta(os.path.join(rri_dir, file_name))
        test_name = current_test if test_name is None else test_name
        subject_id = current_subject if subject_id is None else subject_id
        if current_test != test_name or current_subject != subject_id:
            raise ValueError("One experiment folder should only contain one test and one subject.")
        segments.append((segment_index, meta, t, rr))

    if not segments:
        raise FileNotFoundError(f"No RRI csv files found in {rri_dir}")

    segments.sort(key=lambda item: item[0])
    protocol = get_test_protocol(test_name)
    expected_segment_count = protocol.get("segment_count")
    if expected_segment_count is not None and len(segments) != expected_segment_count:
        raise ValueError(
            f"Test {test_name!r} expects {expected_segment_count} segments, found {len(segments)} in {rri_dir}."
        )
    return test_name, subject_id, segments


def build_beat_series(
    test_name: str,
    subject_id: str,
    segments: list[tuple[int, dict[str, str], np.ndarray, np.ndarray]],
    seg_len_sec: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for segment_index, _, t, rr in segments:
        seg_offset = (segment_index - 1) * seg_len_sec
        hr = rr_to_hr(rr)
        for rel_t, rr_val, hr_val in zip(t, rr, hr):
            rows.append(
                {
                    "subject": subject_id,
                    "test": test_name,
                    "segment": f"Seg{segment_index:02d}",
                    "segment_index": segment_index,
                    "relative_time_sec": float(rel_t),
                    "absolute_time_sec": float(rel_t + seg_offset),
                    "RR_s": float(rr_val),
                    "HR_bpm": float(hr_val),
                }
            )
    return pd.DataFrame(rows)


def compute_metric_series(
    test_name: str,
    subject_id: str,
    segments: list[tuple[int, dict[str, str], np.ndarray, np.ndarray]],
    seg_len_sec: float,
    cfg: dict,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    protocol = get_test_protocol(test_name)
    stress_spans = protocol["stress_spans"]
    freq_kwargs = {
        "interp_fs": float(cfg["freq_interp_fs"]),
        "nperseg": int(cfg["freq_nperseg"]),
        "detrend_flag": bool(cfg["freq_detrend"]),
        "band_vlf": tuple(cfg["band_vlf"]),
        "band_lf": tuple(cfg["band_lf"]),
        "band_hf": tuple(cfg["band_hf"]),
    }

    for segment_index, _, t_rel, rr in segments:
        seg_start = (segment_index - 1) * seg_len_sec
        seg_end = segment_index * seg_len_sec
        t_abs = t_rel + seg_start

        metric_specs = [
            ("RMSSD_ms", float(cfg["rmssd_win_sec"]), float(cfg["rmssd_step_sec"])),
            ("pNN50_pct", float(cfg["time_win_sec"]), float(cfg["time_step_sec"])),
            ("SD1_ms", float(cfg["time_win_sec"]), float(cfg["time_step_sec"])),
            ("SD2_ms", float(cfg["time_win_sec"]), float(cfg["time_step_sec"])),
            ("Baevsky_SI", float(cfg["time_win_sec"]), float(cfg["time_step_sec"])),
            ("VLF", float(cfg["freq_win_sec"]), float(cfg["freq_step_sec"])),
            ("LF", float(cfg["freq_win_sec"]), float(cfg["freq_step_sec"])),
            ("HF", float(cfg["freq_win_sec"]), float(cfg["freq_step_sec"])),
            ("LFHF", float(cfg["freq_win_sec"]), float(cfg["freq_step_sec"])),
        ]

        for metric_name, win_sec, step_sec in metric_specs:
            starts = np.arange(seg_start, seg_end - win_sec + 1e-9, step_sec)
            for start_sec in starts:
                end_sec = start_sec + win_sec
                center_sec = start_sec + win_sec / 2.0
                m = (t_abs >= start_sec) & (t_abs < end_sec) & np.isfinite(rr)
                if np.sum(m) < 10:
                    continue

                rrw = rr[m]
                if metric_name == "RMSSD_ms":
                    value = rmssd(rrw) * 1000.0
                elif metric_name == "pNN50_pct":
                    value = pnn50(rrw, threshold_ms=float(cfg["pnn_threshold_ms"]))
                elif metric_name == "SD1_ms":
                    value = poincare_sd1_sd2(rrw)[0] * 1000.0
                elif metric_name == "SD2_ms":
                    value = poincare_sd1_sd2(rrw)[1] * 1000.0
                elif metric_name == "Baevsky_SI":
                    value = baevsky_stress_index(
                        rrw,
                        bin_ms=float(cfg["si_bin_ms"]),
                        range_percentiles=tuple(cfg["si_range_percentiles"]),
                    )
                else:
                    vlf, lf, hf, lfhf = compute_freq_metrics_window(
                        t_rr=t_abs,
                        rr=rr,
                        win_start=float(start_sec),
                        win_end=float(end_sec),
                        **freq_kwargs,
                    )
                    value_map = {"VLF": vlf, "LF": lf, "HF": hf, "LFHF": lfhf}
                    value = value_map[metric_name]

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
                        "state": classify_window_state(
                            segment_index=segment_index,
                            window_start_in_segment=float(start_sec - seg_start),
                            window_end_in_segment=float(end_sec - seg_start),
                            stress_spans=stress_spans,
                        ),
                        "value": float(value) if np.isfinite(value) else np.nan,
                    }
                )

    return pd.DataFrame(rows)


def compute_wide_feature_table(
    test_name: str,
    subject_id: str,
    segments: list[tuple[int, dict[str, str], np.ndarray, np.ndarray]],
    seg_len_sec: float,
    cfg: dict,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    win_sec = float(cfg["feature_win_sec"])
    step_sec = float(cfg["feature_step_sec"])
    protocol = get_test_protocol(test_name)
    stress_spans = protocol["stress_spans"]

    for segment_index, _, t_rel, rr in segments:
        seg_start = (segment_index - 1) * seg_len_sec
        seg_end = segment_index * seg_len_sec
        t_abs = t_rel + seg_start

        starts = np.arange(seg_start, seg_end - win_sec + 1e-9, step_sec)
        for start_sec in starts:
            end_sec = start_sec + win_sec
            center_sec = start_sec + win_sec / 2.0
            m = (t_abs >= start_sec) & (t_abs < end_sec) & np.isfinite(rr)
            if np.sum(m) < 10:
                continue

            rrw = rr[m]
            hrw = rr_to_hr(rrw)
            vlf, lf, hf, lfhf = compute_freq_metrics_window(
                t_rr=t_abs,
                rr=rr,
                win_start=float(start_sec),
                win_end=float(end_sec),
                interp_fs=float(cfg["freq_interp_fs"]),
                nperseg=int(cfg["freq_nperseg"]),
                detrend_flag=bool(cfg["freq_detrend"]),
                band_vlf=tuple(cfg["band_vlf"]),
                band_lf=tuple(cfg["band_lf"]),
                band_hf=tuple(cfg["band_hf"]),
            )
            sd1, sd2v = poincare_sd1_sd2(rrw)

            rows.append(
                {
                    "subject": subject_id,
                    "test": test_name,
                    "segment": f"Seg{segment_index:02d}",
                    "segment_index": segment_index,
                    "t_start": float(start_sec),
                    "t_end": float(end_sec),
                    "t_center": float(center_sec),
                    "state": classify_window_state(
                        segment_index=segment_index,
                        window_start_in_segment=float(start_sec - seg_start),
                        window_end_in_segment=float(end_sec - seg_start),
                        stress_spans=stress_spans,
                    ),
                    "HR_mean": float(np.mean(hrw)) if hrw.size > 0 else np.nan,
                    "RMSSD_ms": rmssd(rrw) * 1000.0,
                    "pNN50_pct": pnn50(rrw, threshold_ms=float(cfg["pnn_threshold_ms"])),
                    "SD1_ms": sd1 * 1000.0,
                    "SD2_ms": sd2v * 1000.0,
                    "VLF": vlf,
                    "LF": lf,
                    "HF": hf,
                    "LFHF": lfhf,
                    "Baevsky_SI": baevsky_stress_index(
                        rrw,
                        bin_ms=float(cfg["si_bin_ms"]),
                        range_percentiles=tuple(cfg["si_range_percentiles"]),
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    cfg = CONFIG
    experiment_output_dir = cfg["experiment_output_dir"]
    rri_dir = os.path.join(experiment_output_dir, cfg["rri_subdir"])
    features_dir = os.path.join(experiment_output_dir, cfg["features_subdir"])
    ensure_dir(features_dir)

    test_name, subject_id, segments = load_experiment_segments(rri_dir)
    seg_len_sec = float(cfg["seg_len_sec"])

    beat_df = build_beat_series(test_name, subject_id, segments, seg_len_sec)
    metric_df = compute_metric_series(test_name, subject_id, segments, seg_len_sec, cfg)
    wide_df = compute_wide_feature_table(test_name, subject_id, segments, seg_len_sec, cfg)

    beat_path = os.path.join(features_dir, f"BeatSeries_{test_name}_{subject_id}.csv")
    metric_path = os.path.join(features_dir, f"MetricSeries_{test_name}_{subject_id}.csv")
    wide_path = os.path.join(features_dir, f"Features_{test_name}_{subject_id}.csv")

    beat_df.to_csv(beat_path, index=False)
    metric_df.to_csv(metric_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    print("Saved:", beat_path)
    print("Saved:", metric_path)
    print("Saved:", wide_path)

    if bool(cfg.get("auto_make_report", True)):
        report_plotting.CONFIG["experiment_output_dir"] = experiment_output_dir
        report_plotting.CONFIG["test_name"] = test_name
        report_plotting.CONFIG["subject_id"] = subject_id
        report_plotting.CONFIG["seg_len_sec"] = seg_len_sec
        report_plotting.main()


if __name__ == "__main__":
    main()
