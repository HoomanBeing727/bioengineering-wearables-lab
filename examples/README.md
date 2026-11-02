# Synthetic demo data

These scripts generate a small, deterministic dataset and run each pipeline
without any real participant or laboratory data.

## Generate

From the repository root:

```bash
python examples/generate_synthetic_data.py
```

This creates:

```text
Data/demo/20260101-VR-S01/        # ECG + breath, 4 x 300 s segments
Data/demo_ion/demo_ion_Raw/       # chronoamperometry traces
Data/demo_ocp/demo_ocp_Raw/       # stepped OCP traces
Data/demo_randles/raw/            # cyclic voltammograms, two batches
```

All of `Data/` is gitignored. Re-running without `--force` refuses to
overwrite existing demo data:

```bash
python examples/generate_synthetic_data.py --force
```

## Run the demos

```bash
python examples/run_ecg_demo.py
python examples/run_respiration_demo.py
python examples/run_electrochem_demo.py
```

Expected outputs:

- `mental-stress/ecg/output/20260101-VR-S01/rri/`, `features/`, `report/`
- `mental-stress/respiration/output/20260101-VR-S01/intervals/`, `signals/`,
  `features/`, `report/`
- `Data/demo_ion/demo_ion_Final/`
- `Data/demo_ocp/demo_ocp_Final/`
- `Data/demo_randles/tidy/Bare_Electrode_Tidy/` and `AuND_Tidy/`

The demo runners override only the segment durations and window sizes needed
for the shorter synthetic recordings. The production defaults remain 1800 s
segments and the original feature windows.

## R figures

After the electrochemistry demo has run, the R scripts can be tested from the
repository root:

```bash
Rscript electrochemistry/ion_transport_charge/plotting.R
Rscript electrochemistry/open_circuit_potential/plotting.R

RANDLES_TIDY_DIR=Data/demo_randles/tidy \
RANDLES_FIG_ROOT=Data/demo_randles/figures \
Rscript electrochemistry/randles_sevcik/plotting.R

RANDLES_TIDY_DIR=Data/demo_randles/tidy \
RANDLES_FIG_ROOT=Data/demo_randles/figures \
Rscript electrochemistry/randles_sevcik/CV.R
```

The Randles scripts default to `Data/randles/tidy`; the environment variables
above point them at the synthetic demo instead. They require `tidyverse`,
`ggsci`, and `scales`.

## Clean up

```bash
rm -rf Data mental-stress/ecg/output mental-stress/respiration/output
```

The generated data is safe to delete; nothing in it is required by the source
code.
