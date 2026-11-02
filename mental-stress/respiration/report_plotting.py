# report_plotting.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import blended_transform_factory


# =========================
# Global style
# =========================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 28
plt.rcParams["xtick.labelsize"] = 28
plt.rcParams["ytick.labelsize"] = 28


CONFIG = {
    # Paths are relative to the repository root. The defaults point at the
    # synthetic demo dataset.
    "experiment_output_dir": "mental-stress/respiration/output/20260101-VRDemo-S01",
    "features_subdir": "features",
    "report_subdir": "report",
    "dpi": 300,

    # Output file name is still configurable.
    "out_name": None,

    # A/B shared display settings.
    "test_name": "VR Test",
    "subject_id": "S01",
    "seg_len_sec": 1800.0,
    "major_tick_min": 10,
    "minor_tick_min": 1,
    "shade_seg02": True,
    "seg02_alpha": 0.10,
    "mean_window_min": 10,
    "mean_line_color": "red",
    "mean_line_alpha": 0.95,
    "mean_line_style": "--",
    "axis_linewidth": 2,
    "curve_linewidth": 2,
    "hr_alpha": 0.5,
    "draw_major_vlines": True,
    "vline_color": "0.85",
    "vline_width": 2,
    "avg_text_offset_bpm": -6.0,
    "avg_label_y_axes": 0.96,
    "avg_text_offset_rmssd": 0.005,
    "rmssd_ylim": (0, 60),
    "rmssd_y_major": 20,
    "rmssd_pad_low": 0.8,
    "rmssd_pad_high": 2.4,
    "rmssd_window_min": 5,
    "panel_pad_low": 0.8,
    "panel_pad_high": 1.2,
    "baseline_percentiles": (5, 95),
    "baseline_band_color": "0.90",
    "baseline_band_alpha": 0.3,
    "hr_smooth_win_sec": 30.0,
    "hr_smooth_dt_sec": 1.0,
    "hr_raw_color": "#8ecae6",
    "hr_raw_alpha": 0.20,
    "hr_smooth_color": "#066290",
    "hr_smooth_alpha": 0.95,
    "panel_label_fontsize": 32,

    # Frequency/time colors remain unchanged.
    "c_vlf_color": "#6c757d",
    "c_lf_color": "#198754",
    "c_hf_color": "#0d6efd",
    "c_lfhf_color": "#dc3545",
}


def normalize_test_name(test_name: str) -> str:
    key = " ".join(str(test_name).strip().lower().split())
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
            "stress_spans": [(1, 0.0, 1800.0)],
        },
        "vr": {
            "stress_spans": [(1, 0.0, 900.0)],
        },
    }
    return protocols.get(key, {"stress_spans": []})


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def interval_to_rate(interval_s: np.ndarray) -> np.ndarray:
    interval_s = np.asarray(interval_s, dtype=float)
    return 60.0 / interval_s 


def subject_short(subject_id: str) -> str:
    return subject_id.split("-")[0] if subject_id else "Subject"


def style_axes(ax, axis_lw: float) -> None:
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_linewidth(axis_lw)
    ax.tick_params(width=axis_lw, length=8)

def smooth_timeseries_by_rolling_mean(
    t_s: np.ndarray,
    y: np.ndarray,
    dt: float = 1.0,
    win_sec: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    t_s = np.asarray(t_s, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(t_s) & np.isfinite(y)
    t_s = t_s[m]
    y = y[m]
    if t_s.size < 3:
        return t_s, y

    order = np.argsort(t_s)
    t_s = t_s[order]
    y = y[order]

    t0, t1 = float(t_s[0]), float(t_s[-1])
    t_grid = np.arange(t0, t1 + 1e-9, dt)
    y_grid = np.interp(t_grid, t_s, y)

    win_n = max(1, int(round(win_sec / dt)))
    y_smooth = (
        pd.Series(y_grid)
        .rolling(win_n, center=True, min_periods=max(3, win_n // 3))
        .mean()
        .to_numpy()
    )
    return t_grid, y_smooth


def add_stress_background(ax, cfg, n_segments: int) -> None:
    if not cfg["shade_seg02"]:
        return

    seg_len = float(cfg["seg_len_sec"])
    stress_spans = get_test_protocol(cfg["test_name"])["stress_spans"]
    for segment_zero_based, start_sec, end_sec in stress_spans:
        x0 = segment_zero_based * seg_len + start_sec
        x1 = segment_zero_based * seg_len + end_sec
        if x0 >= n_segments * seg_len:
            continue
        ax.axvspan(x0, min(x1, n_segments * seg_len), color="mistyrose", alpha=0.3, zorder=-10)


def read_breath_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(["segment_index", "absolute_time_sec"]).reset_index(drop=True)
    return df


def read_metric_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(["metric", "segment_index", "window_center_sec"]).reset_index(drop=True)
    return df

def build_segment_arrays_from_breath_df(breath_df: pd.DataFrame) -> tuple[list[int]]:
    segment_indices = sorted(breath_df["segment_index"].dropna().astype(int).unique().tolist())
    return segment_indices

def metric_lists(metric_df: pd.DataFrame, metric_name: str, segment_indices: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    centers_list: list[np.ndarray] = []
    values_list: list[np.ndarray] = []

    sub = metric_df[metric_df["metric"] == metric_name]
    for seg_idx in segment_indices:
        seg_df = sub[sub["segment_index"] == seg_idx].copy()
        centers_list.append(seg_df["window_center_sec"].to_numpy(dtype=float))
        values_list.append(seg_df["value"].to_numpy(dtype=float))
    return centers_list, values_list

def draw_time_metric_panel(
    ax,
    centers_list,
    y_list,
    cfg,
    ylabel: str,
    color: str,
    show_mean_lines: bool = False,
    show_mean_labels: bool = False,
    extra_vlines_min: float | None = None,
):
    seg_len = float(cfg["seg_len_sec"])
    n_segments = len(centers_list)
    total_sec = n_segments * seg_len
    curve_lw = cfg["curve_linewidth"]
    axis_lw = cfg["axis_linewidth"]
    major_tick_sec = seg_len
    minor_tick_sec = seg_len / 6.0

    add_stress_background(ax, cfg, n_segments)

    if cfg.get("draw_major_vlines", True):
        for x in np.arange(0, total_sec + 1e-6, major_tick_sec):
            ax.axvline(x, color=cfg["vline_color"], linewidth=cfg["vline_width"], zorder=1)
    if extra_vlines_min is not None:
        extra_sec = float(extra_vlines_min) * 60.0
        for x in np.arange(0, total_sec + 1e-6, extra_sec):
            ax.axvline(x, color=cfg["vline_color"], linewidth=cfg["vline_width"], zorder=1)

    # Calculate actual total from data (works for both shifted and unshifted)                                                                                                                
    if centers_list:                                                                                                                                                                         
        max_times = [np.max(c) for c in centers_list if len(c) > 0]                                                                                                                          
        total_sec = max(max_times) + 120 if max_times else n_segments * seg_len                                                                                                              
    else:                                                                                                                                                                                    
        total_sec = n_segments * seg_len
         
    ax.set_xlim(0, total_sec)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_sec))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_sec))
    xticks = np.arange(0, total_sec + 1e-6, major_tick_sec)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x/60)}" for x in xticks])
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(ylabel)

    yy = np.concatenate([np.asarray(y, dtype=float) for y in y_list]) if y_list else np.array([])
    yy = yy[np.isfinite(yy)]
    if yy.size > 0:
        y0, y1 = float(np.min(yy)), float(np.max(yy))
        if np.isclose(y0, y1):
            eps = 0.1 * (abs(y0) if y0 != 0 else 1.0)
            ax.set_ylim(y0 - eps, y1 + eps)
        else:
            pad = 0.08 * (y1 - y0)
            low_mul = float(cfg.get("panel_pad_low", 0.8))
            high_mul = float(cfg.get("panel_pad_high", 1.2))
            ax.set_ylim(y0 - low_mul * pad, y1 + high_mul * pad)

    for centers, y in zip(centers_list, y_list):
        ax.plot(centers, y, linewidth=curve_lw, alpha=0.9, color=color, zorder=3)
    style_axes(ax, axis_lw)

    if show_mean_lines:
        mean_win_sec = float(cfg.get("mean_window_min", 10)) * 60.0
        nwin = int(np.floor(total_sec / mean_win_sec))
        trans = blended_transform_factory(ax.transData, ax.transAxes)

        for k in range(nwin):
            x0 = k * mean_win_sec
            x1 = (k + 1) * mean_win_sec
            win_vals = []
            for centers, vals in zip(centers_list, y_list):
                m = (centers >= x0) & (centers < x1) & np.isfinite(vals)
                if np.any(m):
                    win_vals.append(vals[m])
            if not win_vals:
                continue

            mean_v = float(np.mean(np.concatenate(win_vals)))
            ax.hlines(
                mean_v, x0, x1,
                colors=cfg["mean_line_color"],
                linestyles=cfg["mean_line_style"],
                linewidth=curve_lw,
                alpha=cfg["mean_line_alpha"],
                zorder=4
            )

            if show_mean_labels:
                x_mid = (x0 + x1) / 2.0
                ax.text(
                    x_mid,
                    float(cfg.get("avg_label_y_axes", 0.96)),
                    f"Avg. = {mean_v:.1f}",
                    transform=trans,
                    ha="center", va="top",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.2),
                    zorder=20
                )


def build_3x4_canvas():
    mosaic = [
        ["A", "A", "B", "B"],
        ["C", "C", "D", "D"],
        ["E", "E", "F", "F"],
    ]
    fig, axd = plt.subplot_mosaic(mosaic, figsize=(36, 24), constrained_layout=False)
    return fig, axd


def main() -> None:
    c = CONFIG
    
    experiment_output_dir = c["experiment_output_dir"]
    features_dir = os.path.join(experiment_output_dir, c["features_subdir"])
    report_dir = os.path.join(experiment_output_dir, c["report_subdir"])
    ensure_dir(report_dir)

    test_name = c["test_name"]
    subject_id = c["subject_id"]


    breath_path = os.path.join(features_dir, f"BreathSeries_{test_name}_{subject_id}.csv")
    metric_path = os.path.join(features_dir, f"MetricSeries_{test_name}_{subject_id}.csv")

    if not os.path.exists(breath_path):
        raise FileNotFoundError(f"Missing breath series file: {breath_path}")
    if not os.path.exists(metric_path):
        raise FileNotFoundError(f"Missing metric series file: {metric_path}")

    breath_df = read_breath_series(breath_path)
    metric_df = read_metric_series(metric_path)
    
    segment_indices = build_segment_arrays_from_breath_df(breath_df)

    br_mean_peak_centers, br_mean_peak_vals = metric_lists(metric_df, "BR_mean_peak", segment_indices)
    br_mean_trough_centers, br_mean_trough_vals = metric_lists(metric_df, "BR_mean_trough", segment_indices)
    brv_rmssd_centers, brv_rmssd_vals = metric_lists(metric_df, "BRV_RMSSD_peak", segment_indices)
    amp_std_centers, amp_std_vals = metric_lists(metric_df, "AMP_std", segment_indices)
    ie_ratio_centers, ie_ratio_vals = metric_lists(metric_df, "IE_amp_ratio_mean", segment_indices)
    brv_cv_centers, brv_cv_vals = metric_lists(metric_df, "BRV_CV_peak", segment_indices)

    fig, axd = build_3x4_canvas()

    fig.text(
        0.01, 0.99,
        f"{test_name} {subject_short(subject_id)}",
        ha="left", va="top"
    )
    
    # After all metric_lists calls (after line 341)
                 
    for key, ax in axd.items():
        if key not in ("A", "B", "C", "D", "E", "F"):
            ax.axis("off")

    # plot everything as a time series 
    # Panel A: Mean Breath Rate from Peaks
    draw_time_metric_panel(axd["A"], br_mean_peak_centers, br_mean_peak_vals, c, ylabel="Breathing Rate \n (Peak Average) (br/min)", 
                           color="#0d6efd", show_mean_lines=True, show_mean_labels=False) 

    # Panel B: Mean Breath Rate from Troughs 
    draw_time_metric_panel(axd["B"], br_mean_trough_centers, br_mean_trough_vals, c, ylabel="Breathing Rate \n (Trough Average) (br/min)",
                           color="#6c757d", show_mean_lines=True, show_mean_labels=False)
    
    # Panel C: Breath Rate Variability RMSSD (HRV RMSSD Analog)
    draw_time_metric_panel(axd["C"], brv_rmssd_centers, brv_rmssd_vals, c, ylabel="BRV RMSSD (s)", 
                           color="#198754", show_mean_lines=True)
   
    # Panel D: Standard Deviation of Inspiration Volume (Using Amplitude as Analog )
    draw_time_metric_panel(axd["D"], amp_std_centers, amp_std_vals, c, ylabel="Amplitude Standard Deviation (s)", 
                           color="#dc3545")
    
    # Panel E: Inspiration:Expiration Ratio 
    draw_time_metric_panel(axd["E"], ie_ratio_centers, ie_ratio_vals, c, ylabel="I/E Ratio", color="#f4a261", show_mean_lines=True)
    
    # Breath Rate Variability Coefficient of Variation 
    draw_time_metric_panel(axd["F"], brv_cv_centers, brv_cv_vals, c, ylabel="BRV CV (%)", color="#9a9d1f")

    out_name = c["out_name"] or f"Report_{test_name}_{subject_id}.png"
    out_path = os.path.join(report_dir, out_name)

    fig.subplots_adjust(hspace=0.6, wspace=0.35, top=0.95, bottom=0.06)
    fig.savefig(out_path, dpi=c["dpi"])
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
