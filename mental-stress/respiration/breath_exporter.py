import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from matplotlib.widgets import Slider

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime

# ======================================================
# User configuration
# Edit this block to choose which experiment folder to process.
# The script will batch-handle all matching breath .txt files in that folder.
# ======================================================

CONFIG = {
    # Process one experiment folder at a time. Paths are relative to the
    # repository root. The default points at the synthetic demo dataset.
    # Real recordings use the same layout, e.g.
    # Data/20260101-Presentation-S02/
    "input_folder": "Data/demo/20260101-VRDemo-S01",
    "output_root": "mental-stress/respiration/output",
    # Only files ending with this suffix are treated as breath inputs.
    "input_glob_suffix": "_Breath.txt",

    # Keep the original sampling and csv-reading assumptions unchanged.
    "FS": 50,

    # Segment extraction defaults to the original 30-minute protocol.
    "start_sec": 0.0,
    "end_sec": None,
    "duration_sec": 1800.0,

    # Optional extra text appended to exported file names.
    "extra_tag": "",

    # Initial x-range shown in the interactive inspection window.
    "inspect_view_sec": 60.0,

    # Export preview PNGs in parallel with CSV export. Interactive windows are
    # off by default so the script also runs on headless machines.
    "export_peak_png": True,
    "export_trough_png": True,
    "show_peak_plots": False,
    "show_trough_plots": False,
}

FOLDER_PATTERN = re.compile(
    r"^(?P<date>\d{8})-(?P<experiment_type>.+)-(?P<subject_code>S\d+)$"
)
FILE_PATTERN = re.compile(r"^(?P<name>.+)_(?P<segment_index>\d+)_(?P<signal_type>[^.]+)\.txt$")


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
    FS: int
    segment_start_sec: float
    segment_end_sec: float
    created_at: str
    global_clipping: str
    peak_clipping: str
    trough_clipping: str


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
    title_peak: str
    title_trough: str
    cleaned_breath: np.ndarray
    peaks: np.ndarray
    troughs: np.ndarray
    FS: int
    peak_time_clean: np.ndarray
    peak_interval_clean: np.ndarray
    trough_time_clean: np.ndarray
    trough_interval_clean: np.ndarray

# =========================
# Raw data loading and fixed-length segment selection
# =========================

def load_raw_data(file_path: str, header=None) -> np.ndarray:
    try: 
        raw_breath = pd.read_csv(file_path,  
                        header=header)
        breath_signal = raw_breath.values.flatten()
        return breath_signal
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit()

def select_segment(
    breath_signal: np.ndarray,
    FS: int,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    duration_sec: float = 1800.0,
) -> tuple[np.ndarray, float, float]:
    n = len(breath_signal)
    total_sec = n / FS

    if end_sec is None:
        end_sec = start_sec + duration_sec
        
    # Clamp to available data
    start_sec = max(0.0, float(start_sec))
    end_sec = min(float(end_sec), total_sec)

    start_i = int(round(start_sec * FS))
    end_i = int(round(end_sec * FS))

    if end_i <= start_i:
        raise ValueError(f"Invalid segment: start_sec={start_sec}, end_sec={end_sec}")

    seg = breath_signal[start_i:end_i]
    
    seg_len_sec = len(seg) / FS
    
    if seg_len_sec < duration_sec * 0.5: 
        print(f"Warning: The data segment is only {seg_len_sec} is significantly shorter than expected." 
              f"Expected File Length is {duration_sec}(s). File may be truncated or corrupted.")
        
    elif total_sec > duration_sec * 1.2: 
        print(f"Warning: The raw file is {total_sec}(s) long and is significantly longer than expected." 
              f"Segment for analysis should be {duration_sec}(s). Inspect end of data for excessive noise.")
    
    return seg, start_sec, end_sec

# ===============================
# Data Quality Assurance Pt. 1
# ===============================
'''
For checking quality of raw data 
'''
def check_clipping(breath_signal: np.ndarray, tol=1, low=0, high=255):
    '''
    Raw Data is collected using 8 bit optical encoder. Data values range from 0 - 255
    with the baseline being 120. If data approaches 0 or 255, it is considered corrupted data
    during the ADC conversion process. 
    
    Returns 
    clipped_frac: [float] --> global saturation as a fraction
    clipped_peak_frac: [int] --> local maxima saturation as a fraction
    clipped_trough_frac: [int] --> local minima saturation as a fraction
    '''
    
    x = np.asarray(breath_signal, dtype=float) # x = Raw Signal as timeseries
    
    clipped_mask: np.ndarray[bool]  = (x <= tol + low) | (x >= high - tol) # mask 1 for all values persisting at 255 or 0 
    clipped_frac: float = float(np.mean(clipped_mask)) # Percentage of clipping
    
    peaks, _ = find_peaks(
    breath_signal, 
    prominence=0.3 * np.std(breath_signal), 
    distance=40)     
    
    troughs, _ = find_peaks(
    -breath_signal, 
    prominence=0.3 * np.std(breath_signal), 
    distance=40)   
    
    # mask 2 for identifying specific 'peaks' that are too close to max and min
    clipped_peak_count: int = int(np.sum(x[peaks] >= high - tol)) # np.sum returns total count
    clipped_trough_count: int = int(np.sum(x[troughs] <= tol + low)) # np.sum returns total count
    
    # as a fraction
    if len(peaks) > 0: 
        clipped_peak_frac: float = float(clipped_peak_count / len(peaks))
    else: 
        clipped_peak_frac: float = np.nan
        
    
    if len(troughs) > 0:
        clipped_trough_frac: float = float(clipped_trough_count / len(troughs)) 
    else: 
        clipped_trough_frac: float = np.nan
        
    return clipped_frac, clipped_peak_frac, clipped_trough_frac

# =========================
# Signal preprocessing
# =========================
'''
Assuming average breath per minute = 12-20 
0.2 - 0.33 Hz 
0.1 Hz set for low cut filter
0.5 Hz set for high cut filter
Previous high cut was 1.0Hz but too lenient due to being ECG baseline
'''
def bandpass_filter(breath_signal: np.ndarray, lowcut: float=0.1, highcut: float = 0.5, 
                    FS: int = 50, order: int = 4) -> np.ndarray:
    nyq = 0.5 * FS
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, breath_signal)

# =========================
# Peak and Trough detection and NN Interval cleaning
# =========================
def detect_peaks(preprocessed_signal: np.ndarray): 
# Finding peaks --> found troughs because of previous peaking in raw data
    peaks, _ = find_peaks(
    preprocessed_signal, 
    prominence=0.3 * np.std(preprocessed_signal), 
    distance=40)     
    
    return peaks   

def detect_troughs(preprocessed_signal: np.ndarray): 
# Finding peaks --> found troughs because of previous peaking in raw data
    troughs, _ = find_peaks(
    -preprocessed_signal, 
    prominence=0.3 * np.std(preprocessed_signal), 
    distance=40)     
    
    return troughs   

def compute_breath_interval(peaks: np.ndarray, FS:int) -> np.ndarray:
    if len(peaks) < 3: 
        raise ValueError("Not enough peaks to calculate rate")
    
    intervals = np.diff(peaks) / FS
    interval_times = peaks[1:] / FS 
    return intervals, interval_times # in seconds

def clean_breath_interval(intervals, times):
    # Artifact Handling
    mask1 = (intervals > 1.2) & (intervals < 10.0)
    intervals_clean = intervals[mask1]
    times_clean = times[mask1]
    
    # If no data returned / pass check 
    if len(intervals_clean) < 1:
        raise ValueError("Not enough peaks to calculate rate")
    
    # Median Check
    med = np.median(intervals_clean)
    mask2 = (intervals_clean > 0.6 * med) & (intervals_clean < 1.2 * med)
    final_intervals = intervals_clean[mask2]
    final_times = times_clean[mask2]
    
    if len(final_intervals) < 1:
        raise ValueError("Not enough peaks to calculate rate")
    
    return final_intervals, final_times    
# ===============================
# Data Quality Assurance Pt. 2
# ===============================

# =========================
# Export helpers
# Write cleaned series and optional preview plots to disk.
# =========================

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def export_breath_csv(out_csv: str, meta: MetaInfo, time_s: np.ndarray, breath_s: np.ndarray) -> None:
    ensure_dir(os.path.dirname(out_csv))

    header_lines = [
        f"# experiment_type: {meta.experiment_type}",
        f"# segment_index: {meta.segment_index}",
        f"# subject_id: {meta.subject_id}",
        f"# raw_file_name: {meta.raw_file_name}",
        f"# fs_hz: {meta.FS}",
        f"# segment_start_sec: {meta.segment_start_sec}",
        f"# segment_end_sec: {meta.segment_end_sec}",
        f"# segment_duration_sec: {meta.segment_end_sec - meta.segment_start_sec}",
        f"# created_at: {meta.created_at}",
        f"# Global Clipping (%): {meta.global_clipping}",
        f"# Peak Clipping (%): {meta.peak_clipping}", 
        f"# Trough Clipping (%): {meta.trough_clipping}",
        "#",
    ]

    df = pd.DataFrame(
        {
            "Time(s)": np.asarray(time_s, dtype=float),
            "Breath Intervals(s)": np.asarray(breath_s, dtype=float),
        }
    )

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n")
        df.to_csv(f, index=False)

def export_filtered_signal_csv(out_csv: str, meta: MetaInfo, signal: np.ndarray, FS: int) -> None:
    """
    Saves the filtered waveform to a CSV file.
    Columns: Time(s), Filtered Signal Amplitude
    """
    ensure_dir(os.path.dirname(out_csv))
    
    # Generate time array based on sampling frequency
    time_s = np.arange(len(signal)) / FS
    
    header_lines = [
        f"# experiment_type: {meta.experiment_type}",
        f"# segment_index: {meta.segment_index}",
        f"# subject_id: {meta.subject_id}",
        f"# raw_file_name: {meta.raw_file_name}",
        f"# fs_hz: {meta.FS}",
        f"# segment_start_sec: {meta.segment_start_sec}",
        f"# segment_end_sec: {meta.segment_end_sec}",
        f"# created_at: {meta.created_at}",
        f"# global_clipping: {meta.global_clipping}", 
        f"# peak_clipping: {meta.peak_clipping}", 
        f"# trough_clipping: {meta.trough_clipping}",
        "#",
    ]
    
    df = pd.DataFrame({
        "Time(s)": time_s,
        "Amplitude": signal
    })
    
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n")
        df.to_csv(f, index=False)

def export_breath_interval_png(out_png: str, time_s: np.ndarray, interval_s: np.ndarray, title_peak: str = "", title_trough: str = "") -> None:
    ensure_dir(os.path.dirname(out_png))
    plt.figure(figsize=(12, 3))
    plt.plot(time_s, interval_s, linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Breath intervals (s)")
    plt.xlim(0, max(time_s[-1], 1))
    if title_peak:
        plt.title(title_peak)
    elif title_trough: 
        plt.title(title_trough)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def export_filtered_signal_png(out_png: str, time_s: np.ndarray, signal: np.ndarray, title: str = "") -> None:
    ensure_dir(os.path.dirname(out_png))
    plt.figure(figsize=(12, 3))
    plt.plot(time_s, signal, linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.xlim(0, max(float(time_s[-1]), 1.0))
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
        raise ValueError(f"File name must follow name_segno_type.txt: {file_name}")

    signal_type = match.group("signal_type").strip().upper()
    if signal_type != "BREATH":
        raise ValueError(f"Only breath .txt files are supported: {file_name}")

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

def inspect_peaks_batch(
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
        t = np.arange(len(result.cleaned_breath)) / result.FS
        max_time = max(max_time, float(t[-1]))
        ax.plot(t, result.cleaned_breath, linewidth=0.8)
        ax.plot(result.peaks / result.FS, result.cleaned_breath[result.peaks], "rx", markersize=4)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(result.title_peak)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(inspect_view_sec, t[-1]))

    for ax in axes_arr[len(results):]:
        ax.set_visible(False)

    fig.suptitle("Peak inspection (Inhale)", fontsize=14)

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


def inspect_troughs_batch(
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
        t = np.arange(len(result.cleaned_breath)) / result.FS
        max_time = max(max_time, float(t[-1]))
        ax.plot(t, result.cleaned_breath, linewidth=0.8)
        ax.plot(result.troughs / result.FS, result.cleaned_breath[result.troughs], "bx", markersize=4)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(result.title_trough)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(inspect_view_sec, t[-1]))

    for ax in axes_arr[len(results):]:
        ax.set_visible(False)

    fig.suptitle("Trough inspection (Exhale)", fontsize=14)

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
    FS: int = 50,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    duration_sec: float = 1800.0,
    extra_tag: str = "",
    export_peak_png: bool = True,
    export_trough_png: bool = True,
) -> ProcessResult:
    
    # Load and split into 1 minute segments of breath signal
    raw = load_raw_data(spec.file_path)
    seg, seg_start, seg_end = select_segment(
        raw, FS=FS, start_sec=start_sec, end_sec=end_sec, duration_sec=duration_sec
    )
    
    clipped_frac, clipped_peak_frac, clipped_trough_frac = check_clipping(seg)
    
    issues = []
    if clipped_frac > 0.1: 
        issues.append(f"Global Clipping: {clipped_frac*100:.1f}%")
    if clipped_peak_frac > 0.1 and np.isfinite(clipped_peak_frac): 
        issues.append(f"Peak Clipping: {clipped_peak_frac*100:.1f}%")
    if clipped_trough_frac > 0.1 and np.isfinite(clipped_trough_frac): 
        issues.append(f"Trough Clipping: {clipped_trough_frac*100:.1f}%")     
            
    if issues:
        print(f"[QA Warning!] {os.path.basename(spec.file_path)}: " + ", ".join(issues))
        
    # Run a butterworth filter on breath signal 
    cleaned_breath = bandpass_filter(seg, lowcut=0.1, highcut=0.5, FS=FS, order=4)
    
    # Detect peaks and troughs (to validate data integrity)
    peaks = detect_peaks(cleaned_breath)
    troughs = detect_troughs(cleaned_breath)
    # Computer 
    peak_intervals, peak_intervals_time = compute_breath_interval(peaks, FS=FS)
    trough_intervals, trough_intervals_time = compute_breath_interval(troughs, FS=FS)
    peak_interval_clean, peak_time_clean = clean_breath_interval(peak_intervals, peak_intervals_time)
    trough_interval_clean, trough_time_clean = clean_breath_interval(trough_intervals, trough_intervals_time)

    meta = MetaInfo(
        experiment_type=spec.experiment_type,
        segment_index=int(spec.segment_index),
        subject_id=str(spec.subject_id),
        raw_file_name=os.path.basename(spec.file_path),
        FS=int(FS),
        segment_start_sec=float(seg_start),
        segment_end_sec=float(seg_end),
        created_at=datetime.now().isoformat(timespec="seconds"),
        global_clipping = f"{float(clipped_frac * 100):.1f}%", 
        peak_clipping = f"{float(clipped_peak_frac * 100):.1f}%", 
        trough_clipping = f"{float(clipped_trough_frac * 100):.1f}%"
    )

    subject_output_dir = os.path.join(output_root, spec.raw_folder_name, "intervals")
    
    # ====================
    # Export Peak Intervals
    # ====================
    
    fname = f"Peak Intervals_{spec.experiment_type}_{spec.subject_id}_Seg{int(spec.segment_index):02d}"
    if extra_tag:
        fname = f"{fname}_{extra_tag}"

    out_peak_csv = os.path.join(subject_output_dir, f"{fname}.csv")
    out_peak_png = os.path.join(subject_output_dir, f"{fname}.png")
    title_peak = f"Peak Intervals - {spec.experiment_type} - {spec.subject_id} - Seg{int(spec.segment_index):02d}"

    export_breath_csv(out_peak_csv, meta, peak_time_clean, peak_interval_clean)
    
    if export_peak_png:
        export_breath_interval_png(
            out_peak_png,
            peak_time_clean,
            peak_interval_clean,
            title_peak=f"Peak Intervals (NN) - {spec.experiment_type} - {spec.subject_id} - Seg{int(spec.segment_index):02d}",
        )
    else:
        out_peak_png = ""
        
    # ========================    
    # Export Trough Intervals
    # ========================
    subject_output_dir = os.path.join(output_root, spec.raw_folder_name, "intervals")
    fname = f"Trough Intervals_{spec.experiment_type}_{spec.subject_id}_Seg{int(spec.segment_index):02d}"
    if extra_tag:
        fname = f"{fname}_{extra_tag}"

    out_trough_csv = os.path.join(subject_output_dir, f"{fname}.csv")
    out_trough_png = os.path.join(subject_output_dir, f"{fname}.png")
    title_trough = f"Trough Intervals - {spec.experiment_type} - {spec.subject_id} - Seg{int(spec.segment_index):02d}"

    export_breath_csv(out_trough_csv, meta, trough_time_clean, trough_interval_clean)

    if export_trough_png:
        export_breath_interval_png(
            out_trough_png,
            trough_time_clean,
            trough_interval_clean,
            title_trough=f"Trough Intervals (NN) - {spec.experiment_type} - {spec.subject_id} - Seg{int(spec.segment_index):02d}",
        )
    else:
        out_trough_png = ""
        
    # ========================
    # Export Filtered Signal
    # ========================
    
    signal_output_dir = os.path.join(output_root, spec.raw_folder_name, "signals")
    fsignalname = f"Filtered Signal_{spec.experiment_type}_{spec.subject_id}_Seg{int(spec.segment_index):02d}"
    if extra_tag: 
        fsignalname = f"{fsignalname}_{extra_tag}"
        
    out_signal_csv = os.path.join(signal_output_dir, f"{fsignalname}.csv")

    export_filtered_signal_csv(out_signal_csv, meta, cleaned_breath, FS=FS)
    
    if export_peak_png:
        out_signal_png = os.path.join(signal_output_dir, f"{fsignalname}.png")
        export_filtered_signal_png(
            out_signal_png, 
            np.arange(len(cleaned_breath)) / FS, 
            cleaned_breath, 
            title=f"Filtered Signal - {spec.experiment_type} - {spec.subject_id}"
        )
        
    return ProcessResult(
        out_csv=out_peak_csv, # Peak CSV as default metadata
        out_png=out_peak_png, # Peak PNG as default metadata
        title_peak=title_peak,
        title_trough=title_trough,
        cleaned_breath=cleaned_breath,
        peaks=peaks,
        troughs=troughs,
        FS=FS,
        peak_time_clean=peak_time_clean,
        peak_interval_clean=peak_interval_clean,
        trough_time_clean=trough_time_clean,
        trough_interval_clean=trough_interval_clean,
    )


def main() -> None:
    c = CONFIG

    specs = collect_input_specs(c["input_folder"], c["input_glob_suffix"])
    results: list[ProcessResult] = []

    print(f"Found {len(specs)} Breath file(s) under {c['input_folder']}.")

    # Export first, then open a single interactive window for visual review.
    for spec in specs:
        result = process_one_file(
            spec=spec,
            output_root=c["output_root"],
            FS=c["FS"],
            start_sec=c["start_sec"],
            end_sec=c["end_sec"],
            duration_sec=c["duration_sec"],
            extra_tag=c["extra_tag"],
            export_peak_png=c["export_peak_png"],
            export_trough_png=c["export_trough_png"],
        )
        results.append(result)
        print(f"CSV : {result.out_csv}")
        if result.out_png:
            print(f"PNG : {result.out_png}")

    if c["show_peak_plots"]:
        inspect_peaks_batch(results, inspect_view_sec=c["inspect_view_sec"])
    
    if c["show_trough_plots"]:
        inspect_troughs_batch(results, inspect_view_sec=c["inspect_view_sec"])
        
    print(f"Done. Exported {len(results) * 2} Interval file(s).")

if __name__ == "__main__":
    main()