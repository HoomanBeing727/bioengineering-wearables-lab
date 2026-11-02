# Data layout

No raw or participant data is committed to this repository. The scripts read
and write under `Data/`, which is gitignored.

## Mental-stress recordings

One experiment folder per session:

```text
Data/demo/20260101-VR-S01/
├── S01_01_ECG.csv
├── S01_01_Breath.txt
├── S01_02_ECG.csv
├── S01_02_Breath.txt
└── ...
```

Folder convention:

```text
YYYYMMDD-TestType-Sxx
```

- `YYYYMMDD` — recording date
- `TestType` — e.g. `VR`, `Gaming`, `Presentation`
- `Sxx` — anonymised subject code (no names)

File convention:

```text
<subject>_<segment>_<signal>.<ext>
```

- ECG files are CSV with two metadata rows, then time and voltage columns.
- Breath files are single-column text exports from the optical encoder.
- Each segment is expected to be 1800 s by default. The synthetic demo uses
  300 s segments and overrides the duration in the demo runner.

## Electrochemistry

Each experiment has its own subfolder:

```text
Data/<experiment>/
├── <experiment>_Psess/   # optional PalmSens .pssession files
├── <experiment>_Raw/     # converted CSVs or direct instrument exports
├── <experiment>_Tidy/    # generated tidy data
└── <experiment>_Final/   # generated summaries, slopes, plots
```

The raw filename encodes the concentration, unit, analyte, sample ID, and
(for OCP) the step number. See the README in each project for the exact
patterns.

Randles–Ševčík data uses a separate layout:

```text
Data/randles/
├── raw/
│   ├── Bare_Electrode/
│   │   ├── 10 mVs Bare_1.txt
│   │   └── ...
│   └── AuND/
└── tidy/
    ├── Bare_Electrode_Tidy/
    └── AuND_Tidy/
```

The synthetic demo uses `Data/demo_randles/` with the same raw/tidy structure,
so regenerating it can never touch real `Data/randles/` recordings.

## Generated data

`examples/generate_synthetic_data.py` creates a deterministic, entirely
synthetic demo dataset under `Data/`. It is safe to delete and regenerate.
