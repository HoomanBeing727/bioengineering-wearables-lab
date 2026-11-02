# report_plotting.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
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
    "experiment_output_dir": "mental-stress/ecg/output/20260101-VRDemo-S01",
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
        "presentation": "presentation",
        "presentation test": "presentation",
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
        "presentation": {
            "stress_spans": [(2, 0.0, 1800.0)],
        },
    }
    return protocols.get(key, {"stress_spans": []})


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def rr_to_hr(rr_s: np.ndarray) -> np.ndarray:
    rr_s = np.asarray(rr_s, dtype=float)
    return 60.0 / rr_s


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


def read_beat_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(["segment_index", "absolute_time_sec"]).reset_index(drop=True)
    return df


def read_metric_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(["metric", "segment_index", "window_center_sec"]).reset_index(drop=True)
    return df


def build_segment_arrays_from_beat_df(beat_df: pd.DataFrame) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    segment_indices = sorted(beat_df["segment_index"].dropna().astype(int).unique().tolist())
    Ts: list[np.ndarray] = []
    RRs: list[np.ndarray] = []
    for seg_idx in segment_indices:
        seg_df = beat_df[beat_df["segment_index"] == seg_idx].copy()
        Ts.append(seg_df["relative_time_sec"].to_numpy(dtype=float))
        RRs.append(seg_df["RR_s"].to_numpy(dtype=float))
    return segment_indices, Ts, RRs


def metric_lists(metric_df: pd.DataFrame, metric_name: str, segment_indices: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    centers_list: list[np.ndarray] = []
    values_list: list[np.ndarray] = []

    sub = metric_df[metric_df["metric"] == metric_name]
    for seg_idx in segment_indices:
        seg_df = sub[sub["segment_index"] == seg_idx].copy()
        centers_list.append(seg_df["window_center_sec"].to_numpy(dtype=float))
        values_list.append(seg_df["value"].to_numpy(dtype=float))
    return centers_list, values_list


def draw_A_hr_concat(ax, Ts, RRs, cfg):
    seg_len = float(cfg["seg_len_sec"])
    major_tick_sec = cfg["major_tick_min"] * 60.0
    minor_tick_sec = cfg["minor_tick_min"] * 60.0
    win_sec = cfg["mean_window_min"] * 60.0

    curve_lw = cfg["curve_linewidth"]
    axis_lw = cfg["axis_linewidth"]

    all_t, all_hr = [], []
    n_segments = len(Ts)
    for i in range(n_segments):
        t = Ts[i]
        rr = RRs[i]
        hr = rr_to_hr(rr)
        offset = i * seg_len
        t_seg = t + offset
        all_t.append(t_seg)
        all_hr.append(hr)

        ax.plot(
            t_seg, hr,
            linewidth=curve_lw,
            color=cfg.get("hr_raw_color", "#8ecae6"),
            alpha=cfg.get("hr_raw_alpha", 0.20),
            zorder=3
        )

        t_hr_s, hr_s = smooth_timeseries_by_rolling_mean(
            t_seg, hr,
            dt=float(cfg.get("hr_smooth_dt_sec", 1.0)),
            win_sec=float(cfg.get("hr_smooth_win_sec", 30.0))
        )
        ax.plot(
            t_hr_s, hr_s,
            linewidth=curve_lw,
            color=cfg.get("hr_smooth_color", "#023047"),
            alpha=cfg.get("hr_smooth_alpha", 0.95),
            zorder=4
        )

    all_t = np.concatenate(all_t)
    all_hr = np.concatenate(all_hr)

    add_stress_background(ax, cfg, n_segments)

    total_sec = n_segments * seg_len
    ax.set_xlim(0, total_sec)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_sec))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_sec))

    xticks = np.arange(0, total_sec + 1e-6, major_tick_sec)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x/60)}" for x in xticks])

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("HR (bpm)")

    finite_hr = all_hr[np.isfinite(all_hr)]
    if finite_hr.size > 0:
        hr_min = float(np.min(finite_hr))
        hr_max = float(np.max(finite_hr))
        hr_span = max(8.0, hr_max - hr_min)

        # Leave extra room at the top for Avg labels while keeping the lower bound tight.
        y_low = max(0.0, hr_min - 0.08 * hr_span)
        y_high = hr_max + 0.28 * hr_span

        # Keep a reasonable minimum display span so quiet traces do not look cramped.
        if (y_high - y_low) < 25.0:
            mid = 0.5 * (y_high + y_low)
            y_low = max(0.0, mid - 12.5)
            y_high = mid + 12.5
        ax.set_ylim(y_low, y_high)
    else:
        ax.set_ylim(50, 100)

    hr_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    if hr_range <= 40:
        y_major = 5
    elif hr_range <= 80:
        y_major = 10
    else:
        y_major = 15
    ax.yaxis.set_major_locator(MultipleLocator(y_major))

    nwin = int(np.floor(total_sec / win_sec))
    for k in range(nwin):
        x0 = k * win_sec
        x1 = (k + 1) * win_sec
        m = (all_t >= x0) & (all_t < x1) & np.isfinite(all_hr)
        if not np.any(m):
            continue
        mean_hr = float(np.mean(all_hr[m]))

        ax.hlines(
            mean_hr, x0, x1,
            colors=cfg["mean_line_color"],
            linestyles=cfg["mean_line_style"],
            linewidth=curve_lw,
            alpha=cfg["mean_line_alpha"],
            zorder=2
        )

        x_mid = (x0 + x1) / 2.0
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            x_mid,
            float(cfg.get("avg_label_y_axes", 0.96)),
            f"Avg. = {mean_hr:.1f}",
            transform=trans,
            ha="center",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.2),
            zorder=20
        )

        if cfg["draw_major_vlines"]:
            ax.axvline(x0, color=cfg["vline_color"], linewidth=cfg["vline_width"], zorder=1)

    if cfg["draw_major_vlines"]:
        ax.axvline(total_sec, color=cfg["vline_color"], linewidth=cfg["vline_width"], zorder=1)

    style_axes(ax, axis_lw)


def draw_B_rmssd_concat(ax, centers_list, vals_list, cfg):
    seg_len = float(cfg["seg_len_sec"])
    major_tick_sec = cfg["major_tick_min"] * 60.0
    minor_tick_sec = cfg["minor_tick_min"] * 60.0

    curve_lw = cfg["curve_linewidth"]
    axis_lw = cfg["axis_linewidth"]
    n_segments = len(centers_list)
    total_sec = n_segments * seg_len
    ax.set_xlim(0, total_sec)

    add_stress_background(ax, cfg, n_segments)

    ax.xaxis.set_major_locator(MultipleLocator(major_tick_sec))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_sec))
    xticks = np.arange(0, total_sec + 1e-6, major_tick_sec)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x/60)}" for x in xticks])

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("RMSSD (ms)")

    if cfg["draw_major_vlines"]:
        for x in np.arange(0, total_sec + 1e-6, major_tick_sec):
            ax.axvline(x, color=cfg["vline_color"], linewidth=cfg["vline_width"], zorder=1)

    finite_vals = np.concatenate([v[np.isfinite(v)] for v in vals_list]) if vals_list else np.array([])
    base_vals = vals_list[0][np.isfinite(vals_list[0])] if vals_list else np.array([])

    y_limits = cfg.get("rmssd_ylim", None)
    if y_limits is None:
        if finite_vals.size > 0:
            y0, y1 = float(np.min(finite_vals)), float(np.max(finite_vals))
            pad = 0.08 * (y1 - y0) if not np.isclose(y0, y1) else max(1.0, 0.1 * abs(y1))
            low_mul = float(cfg.get("rmssd_pad_low", 0.8))
            high_mul = float(cfg.get("rmssd_pad_high", 2.4))
            y_low = y0 - low_mul * pad
            y_high = y1 + high_mul * pad
        else:
            y_low, y_high = 0.0, 80.0
    else:
        y_low, y_high = y_limits

    # Reserve extra headroom so the curve and the top Avg labels do not collide.
    if finite_vals.size > 0:
        data_top = float(np.max(finite_vals))
        data_bottom = float(np.min(finite_vals))
        span = max(1.0, y_high - y_low, data_top - data_bottom)
        y_low = min(y_low, data_bottom - 0.05 * span)
        y_high = max(y_high, data_top + 0.18 * span)

    ax.set_ylim(y_low, y_high)
    y0, y1 = ax.get_ylim()
    ax.yaxis.set_major_locator(MultipleLocator(cfg.get("rmssd_y_major", 20)))

    if base_vals.size >= 5:
        p_lo, p_hi = cfg.get("baseline_percentiles", (10, 90))
        y_lo, y_hi = np.percentile(base_vals, [p_lo, p_hi])
        ax.axhspan(
            y_lo, y_hi,
            color=cfg.get("baseline_band_color", "0.90"),
            alpha=cfg.get("baseline_band_alpha", 0.6),
            zorder=-20
        )

        label = f"Personal reference range: {y_lo:.0f} - {y_hi:.0f} ms"
        y_text = min(y_hi + 0.08 * (y1 - y0), y1 - 0.06 * (y1 - y0))
        x_text = 0.08 * seg_len
        ax.text(
            x_text, y_text, label,
            ha="left", va="top",
            color="black",
            fontsize=20,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.25),
            zorder=6
        )

    for centers, vals in zip(centers_list, vals_list):
        ax.plot(
            centers, vals,
            linewidth=curve_lw,
            color=cfg.get("hr_smooth_color", "#066290"),
            alpha=0.85,
            zorder=3
        )

    mean_win_sec = float(cfg.get("mean_window_min", 10)) * 60.0
    nwin = int(np.floor(total_sec / mean_win_sec))
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for k in range(nwin):
        x0 = k * mean_win_sec
        x1 = (k + 1) * mean_win_sec
        win_vals = []
        for centers, vals in zip(centers_list, vals_list):
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

    style_axes(ax, axis_lw)


def draw_freq_panel(ax, centers_list, y_list, cfg, ylabel: str, color: str):
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

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_size(20)
    ax.yaxis.get_offset_text().set_color("black")
    style_axes(ax, axis_lw)


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


def build_4x4_canvas():
    mosaic = [
        ["A", "A", "A", "A"],
        ["B", "B", "B", "K"],
        ["C", "D", "E", "F"],
        ["L", "M", "N", "N"],
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

    beat_path = os.path.join(features_dir, f"BeatSeries_{test_name}_{subject_id}.csv")
    metric_path = os.path.join(features_dir, f"MetricSeries_{test_name}_{subject_id}.csv")

    if not os.path.exists(beat_path):
        raise FileNotFoundError(f"Missing beat series file: {beat_path}")
    if not os.path.exists(metric_path):
        raise FileNotFoundError(f"Missing metric series file: {metric_path}")

    beat_df = read_beat_series(beat_path)
    metric_df = read_metric_series(metric_path)
    segment_indices, Ts, RRs = build_segment_arrays_from_beat_df(beat_df)

    rmssd_centers, rmssd_vals = metric_lists(metric_df, "RMSSD_ms", segment_indices)
    vlf_centers, vlf_vals = metric_lists(metric_df, "VLF", segment_indices)
    lf_centers, lf_vals = metric_lists(metric_df, "LF", segment_indices)
    hf_centers, hf_vals = metric_lists(metric_df, "HF", segment_indices)
    lfhf_centers, lfhf_vals = metric_lists(metric_df, "LFHF", segment_indices)
    pnn_centers, pnn_vals = metric_lists(metric_df, "pNN50_pct", segment_indices)
    sd1_centers, sd1_vals = metric_lists(metric_df, "SD1_ms", segment_indices)
    sd2_centers, sd2_vals = metric_lists(metric_df, "SD2_ms", segment_indices)
    si_centers, si_vals = metric_lists(metric_df, "Baevsky_SI", segment_indices)

    fig, axd = build_4x4_canvas()

    fig.text(
        0.01, 0.99,
        f"{test_name} {subject_short(subject_id)}",
        ha="left", va="top"
    )

    for key, ax in axd.items():
        if key not in ("A", "B", "C", "D", "E", "F", "K", "L", "M", "N"):
            ax.axis("off")

    draw_A_hr_concat(axd["A"], Ts, RRs, c)
    draw_B_rmssd_concat(axd["B"], rmssd_centers, rmssd_vals, c)
    draw_freq_panel(axd["C"], vlf_centers, vlf_vals, c, ylabel="VLF power (a.u.)", color=c.get("c_vlf_color", "#6c757d"))
    draw_freq_panel(axd["D"], lf_centers, lf_vals, c, ylabel="LF power (a.u.)", color=c.get("c_lf_color", "#198754"))
    draw_freq_panel(axd["E"], hf_centers, hf_vals, c, ylabel="HF power (a.u.)", color=c.get("c_hf_color", "#0d6efd"))
    draw_freq_panel(axd["F"], lfhf_centers, lfhf_vals, c, ylabel="LF/HF", color=c.get("c_lfhf_color", "#dc3545"))
    draw_time_metric_panel(axd["K"], pnn_centers, pnn_vals, c, ylabel="pNN50 (%)", color="#f4a261")
    draw_time_metric_panel(axd["L"], sd1_centers, sd1_vals, c, ylabel="SD1 (ms)", color="#2a9d8f")
    draw_time_metric_panel(axd["M"], sd2_centers, sd2_vals, c, ylabel="SD2 (ms)", color="#e76f51")
    draw_time_metric_panel(
        axd["N"],
        si_centers,
        si_vals,
        c,
        ylabel="Baevsky SI (a.u.)",
        color="#6f42c1",
        show_mean_lines=True,
        show_mean_labels=False,
        extra_vlines_min=float(c.get("mean_window_min", 10)),
    )

    out_name = c["out_name"] or f"Report_{test_name}_{subject_id}.png"
    out_path = os.path.join(report_dir, out_name)

    fig.subplots_adjust(hspace=0.6, wspace=0.35, top=0.95, bottom=0.06)
    fig.savefig(out_path, dpi=c["dpi"])
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
