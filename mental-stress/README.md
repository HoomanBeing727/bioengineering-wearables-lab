# Mental-stress ECG and respiration processing

Python pipelines for ECG-derived heart-rate variability and respiration
intervals from wearable sensor recordings. The scripts were used to process
30-minute mental-stress sessions and generate report-ready figures.

No participant data is included in this repository. The demo uses synthetic
signals generated locally.

## Directory structure

```text
mental-stress/
├── ecg/
│   ├── ecg_rri_exporter.py
│   ├── export_features.py
│   └── report_plotting.py
└── respiration/
    ├── breath_exporter.py
    ├── export_features.py
    └── report_plotting.py
```

Generated files go to `mental-stress/ecg/output/` and
`mental-stress/respiration/output/`; both are gitignored.

## Input data convention

One folder per recording session:

```text
Data/demo/20260101-VR-S01/
├── S01_01_ECG.csv
├── S01_01_Breath.txt
├── S01_02_ECG.csv
├── S01_02_Breath.txt
└── ...
```

Folder names follow:

```text
YYYYMMDD-TestType-Sxx
```

- `YYYYMMDD` — recording date
- `TestType` — experiment label, e.g. `VR`, `Gaming`, `Presentation`
- `Sxx` — anonymised subject code; names are never part of the public format

Signals are split into numbered segments:

- ECG: `<subject>_<segment>_ECG.csv`, with two metadata rows followed by time
  and voltage.
- Breath: `<subject>_<segment>_Breath.txt`, a single-column optical-encoder
  export on the 0–255 scale.

The default segment length is 1800 s. The synthetic demo uses 300 s segments
and overrides the duration in the demo runner.

## ECG workflow

### 1. Export RR intervals

`ecg_rri_exporter.py` reads each ECG segment, optionally inverts the polarity,
removes baseline wander with a `db4` wavelet, applies a 0.5–45 Hz Butterworth
band-pass filter, detects R peaks, cleans the RR series, and exports:

```text
mental-stress/ecg/output/<experiment>/rri/
├── RRI_VR_S01_Seg01.csv
├── RRI_VR_S01_Seg01.png
└── ...
```

Each CSV includes metadata headers (`experiment_type`, `segment_index`,
`subject_id`, `fs_hz`, segment bounds, creation time).

### 2. Export features and report

`export_features.py` reads the RRI files and writes:

```text
mental-stress/ecg/output/<experiment>/features/
├── BeatSeries_VR_S01.csv
├── MetricSeries_VR_S01.csv
└── Features_VR_S01.csv
```

- `BeatSeries` — beat-level RR and HR values.
- `MetricSeries` — rolling metric series used by the report figure.
- `Features` — wide feature table for downstream analysis/ML.

It then calls `report_plotting.py` to produce:

```text
mental-stress/ecg/output/<experiment>/report/
└── Report_VR_S01.png
```

Metrics include HR, RMSSD, pNN50, Poincaré SD1/SD2, VLF/LF/HF power,
LF/HF ratio, and the Baevsky stress index.

### Run the ECG pipeline

For the synthetic demo:

```bash
python examples/generate_synthetic_data.py
python examples/run_ecg_demo.py
```

For a real 30-minute recording, set `input_folder`, `duration_sec = 1800.0`,
and `show_plots` in `ecg_rri_exporter.CONFIG`, then run:

```bash
python mental-stress/ecg/ecg_rri_exporter.py
python mental-stress/ecg/export_features.py
```

## Respiration workflow

### 1. Export breath intervals and filtered signals

`breath_exporter.py` reads the raw breath text files, selects the configured
segment, band-pass filters at 0.1–0.5 Hz, detects inspiration peaks and
expiration troughs, cleans the intervals, and writes:

```text
mental-stress/respiration/output/<experiment>/
├── intervals/
│   ├── Peak Intervals_VR_S01_Seg01.csv
│   ├── Trough Intervals_VR_S01_Seg01.csv
│   └── ...
└── signals/
    └── Filtered Signal_VR_S01_Seg01.csv
```

The script also runs clipping QA against the 0–255 optical-encoder range and
reports the percentage of clipped samples.

### 2. Export features and report

`export_features.py` reads the interval and signal files and writes:

```text
mental-stress/respiration/output/<experiment>/features/
├── BreathSeries_VR_S01.csv
├── MetricSeries_VR_S01.csv
└── Features_VR_S01.csv
```

It then calls `report_plotting.py` to produce the respiration report figure.
Features include breath rate, interval variability (RMSSD, SD, CV), amplitude
statistics, and inspiration/expiration amplitude ratios.

### Run the respiration pipeline

For the synthetic demo:

```bash
python examples/generate_synthetic_data.py
python examples/run_respiration_demo.py
```

For a real 30-minute recording, set `input_folder`, `duration_sec = 1800.0`,
and the plot flags in `breath_exporter.CONFIG`, then run:

```bash
python mental-stress/respiration/breath_exporter.py
python mental-stress/respiration/export_features.py
```

## Configuration

Each script has a `CONFIG` dictionary near the top. The main fields are:

- `input_folder` / `experiment_output_dir` — paths relative to the repository
  root.
- `start_sec`, `end_sec`, `duration_sec` — segment extraction bounds.
- `fs` / `FS` — sampling frequency (250 Hz ECG, 50 Hz respiration).
- `show_plots` / `show_peak_plots` / `show_trough_plots` — interactive
  inspection windows; off by default for headless runs.
- Window sizes for rolling metrics and feature tables.

The default paths point at the synthetic demo. Change them to a folder under
`Data/` to process a real recording. The demo runners override `duration_sec`
to 300 s and shorten the feature windows; the production defaults remain
1800 s.

## Dependencies

```bash
pip install -r requirements.txt
```

This covers NumPy, Pandas, Matplotlib, SciPy, and PyWavelets.

## Data privacy

This repository contains code only. Subject folders use anonymised codes
(`S01`, `S02`, ...), and no names, raw recordings, or exported participant
tables are committed. The synthetic demo data is generated locally and
gitignored.
