import pandas as pd
import numpy as np
import os
from io import StringIO
from scipy.stats import linregress

def _group_cols(df, base_cols):
    if 'Batch' in df.columns and df['Batch'].notna().any():
        return ['Batch'] + base_cols
    return base_cols


def validate_steps(df):
    issues = []
    group_keys = ['ID']
    if 'Batch' in df.columns and df['Batch'].notna().any():
        group_keys = ['Batch', 'ID']

    for keys, group in df.groupby(group_keys):
        keys = keys if isinstance(keys, tuple) else (keys,)
        label = ', '.join(str(k) for k in keys)

        steps = sorted(group['Step'].unique())
        min_step, max_step = min(steps), max(steps)
        expected = set(range(min_step, max_step + 1))
        actual = set(steps)

        missing = expected - actual
        if missing:
            issues.append(f"  {label}: missing step(s) {sorted(missing)}")

        step_conc = group[['Step', 'Concentration']].drop_duplicates()
        dup = step_conc[step_conc.duplicated(subset='Step', keep=False)]
        if not dup.empty:
            for step in dup['Step'].unique():
                concs = dup[dup['Step'] == step]['Concentration'].unique()
                issues.append(f"  {label}: step {step} has multiple concentrations {list(concs)}")

        if len(steps) >= 3:
            seq = step_conc.sort_values('Step')['Concentration'].values
            bottom = np.argmin(seq)
            desc = seq[:bottom + 1]
            asc = seq[bottom:]
            if len(desc) > 1 and not all(desc[i] > desc[i + 1] for i in range(len(desc) - 1)):
                issues.append(f"  {label}: descending segment not monotonic (steps {steps[:bottom + 1]})")
            if len(asc) > 1 and not all(asc[i] < asc[i + 1] for i in range(len(asc) - 1)):
                issues.append(f"  {label}: ascending segment not monotonic (steps {steps[bottom:]})")

        if len(steps) <= 2:
            pairs = step_conc.sort_values('Step')
            info = ', '.join(f"step={r['Step']}({r['Concentration']}mM)" for _, r in pairs.iterrows())
            issues.append(f"  {label}: only {len(steps)} step(s) [{info}] — possible filename typo")

    return issues


def compute_ocp_summary(subfolder, tidy_folder, final_folder, sep_direction, window):
    all_dfs = []
    tidy_path = os.path.join("Data", subfolder, tidy_folder)
    final_path = os.path.join("Data", subfolder, final_folder)
    os.makedirs(final_path, exist_ok=True)

    for root, dirs, files in os.walk(tidy_path):
        for filename in files:
            if not filename.endswith(".csv"):
                continue
            file_path = os.path.join(root, filename)
            df = pd.read_csv(file_path)
            all_dfs.append(df)

    if not all_dfs:
        print("No tidy data found.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    issues = validate_steps(df_all)
    if issues:
        print("\n=== FILE VALIDATION ISSUES FOUND ===")
        for issue in issues:
            print(issue)
        print("====================================\n")
    else:
        print("File validation: all OK")

    if sep_direction: 
        conc_by_step = df_all.groupby('Step')['Concentration'].mean()
        v_bottom_step = conc_by_step.idxmin()
        df_all['Direction'] = df_all['Step'].apply(
            lambda s: 'Descending' if s <= v_bottom_step else 'Ascending'
        )
        bottom_rows = df_all[df_all['Step'] == v_bottom_step].copy()
        bottom_rows['Direction'] = 'Ascending'
        df_all = pd.concat([df_all, bottom_rows], ignore_index=True)
        group_cols = _group_cols(df_all, ['Direction', 'ID', 'Concentration', 'Units', 'Step'])
   
    else: 
        group_cols = _group_cols(df_all, ['ID', 'Concentration', 'Units', 'Step'])
        
    final = (
        df_all
        .groupby(group_cols)
        .tail(window)
        .groupby(group_cols, as_index=False)['Potential/V']
        .agg(Mean_OCP='mean', SD_OCP='std')
    )
    
    os.makedirs(final_path, exist_ok=True)
    final.to_csv(os.path.join(final_path, "ocp_final.csv"), index=False)

    tidy_cols = ['Time/sec', 'Potential/V']
    if 'Batch' in df_all.columns:
        tidy_cols = ['Time/sec', 'Potential/V', 'Batch']
    if sep_direction:
        tidy_cols.append('Direction')
        print(f"  V-bottom detected at Step {v_bottom_step}")
        print(f"  Direction counts: {df_all['Direction'].value_counts().to_dict()}")

    df_all[tidy_cols + ['ID', 'Concentration', 'Units', 'Step']].to_csv(
        os.path.join(final_path, "ocp_tidy_long.csv"), index=False
    )

    print("\n")
    print(f"OCP final saved to {final_path}/ocp_final.csv")
    print(f"OCP tidy long data saved to {final_path}/ocp_tidy_long.csv")
    
    compute_ocp_slopes(final_path)


def compute_ocp_slopes(final_path):
    df = pd.read_csv(os.path.join(final_path, "ocp_final.csv"))

    base = ['ID']
    if 'Direction' in df.columns:
        base = ['Direction'] + base
    group_cols = base
    if 'Batch' in df.columns and df['Batch'].notna().any():
        group_cols = ['Batch'] + group_cols
    groupby_key = group_cols[0] if len(group_cols) == 1 else group_cols

    slopes = []
    for group_keys, group in df.groupby(groupby_key):
        keys = group_keys if isinstance(group_keys, tuple) else (group_keys,)
        row = dict(zip(group_cols, keys))

        x = np.log10(group['Concentration'].values)
        y = group['Mean_OCP'].values
        slope, intercept, r_value, _, _ = linregress(x, y)
        row.update({
            'Slope_V_per_decade': slope,
            'Intercept_V': intercept,
            'R_squared': r_value ** 2,
            'n_points': len(x)
        })
        slopes.append(row)

    slopes_df = pd.DataFrame(slopes)
    slopes_df.to_csv(os.path.join(final_path, "slopes.csv"), index=False)
    print(f"  Slopes saved to {final_path}/slopes.csv")


def process_4channel_ocp(subfolder, raw_folder, final_folder, n_channels, concentrations):
    uniq = sorted(set(concentrations), reverse=True)
    v_shape = uniq + uniq[-2::-1] # Builds V-shaped concentration array for OCP

    raw_path = os.path.join("Data", subfolder, raw_folder)
    final_path = os.path.join("Data", subfolder, final_folder)
    os.makedirs(final_path, exist_ok=True)

    txt_files = [f for f in os.listdir(raw_path) if f.endswith('.txt')]
    if not txt_files:
        raise FileNotFoundError(f"No .txt file found in {raw_path}")
    file_path = os.path.join(raw_path, txt_files[0])

    with open(file_path, 'r', encoding='utf-16') as f:
        for line in f:
            clean = line.strip()
            if 'Time/sec' in clean and 'i1/V' in clean:
                remaining = line + f.read()
                stream = StringIO(remaining)
                df_raw = pd.read_csv(stream, sep=',', skipinitialspace=True)
                break

    long_dfs = []
    for ch in range(1, n_channels + 1):
        col = f'i{ch}/V'
        if col not in df_raw.columns:
            print(f"Channel {ch} column '{col}' not found, skipping")
            continue
        df_ch = df_raw[['Time/sec', col]].dropna().copy()
        df_ch.columns = ['Time/sec', 'Potential/V']
        df_ch['ID'] = f'Ch{ch}'
        long_dfs.append(df_ch)

    df_all = pd.concat(long_dfs, ignore_index=True)
    df_all['Time/sec'] = pd.to_numeric(df_all['Time/sec'], errors='coerce')
    df_all['Potential/V'] = pd.to_numeric(df_all['Potential/V'], errors='coerce')
    df_all = df_all.dropna(subset=['Time/sec', 'Potential/V'])
    df_all = df_all.sort_values(['ID', 'Time/sec']).reset_index(drop=True)

    # Detect segment boundaries from Potential drops
    # 0.017 is ideal drop -> 0.059 / decade 
    # 0.014 is plausible non-ideal Nernst slope ~0.0465 / decade 
    threshold = 0.014
    boundary_times = set()
    for ch in range(1, n_channels + 1):
        df_ch = df_all[df_all['ID'] == f'Ch{ch}'].copy()
        if df_ch.empty:
            continue
        dV = df_ch['Potential/V'].diff().abs()
        dV_smooth = dV.rolling(3, center=True).mean()
        drop_times = df_ch.loc[dV_smooth > threshold, 'Time/sec'].tolist()
        boundary_times.update(drop_times)

    boundary_times = sorted(boundary_times)
    n_segments = len(boundary_times) + 1

    if n_segments != len(v_shape):
        raise ValueError(
            f"Detected {n_segments} segments from Potential drops, "
            f"but v_shape has {len(v_shape)} concentrations ({v_shape}). "
            "Adjust threshold or check raw data."
        )

    # Assign step globally by counting boundaries before each row's time
    boundaries = np.array(boundary_times)
    df_all['Step'] = np.searchsorted(boundaries, df_all['Time/sec'].values, side='right') + 1

    conc_map = {i + 1: v_shape[i] for i in range(len(v_shape))}
    df_all['Concentration'] = df_all['Step'].map(conc_map)
    df_all['Units'] = 'mM'

    df_all.to_csv(os.path.join(final_path, "ocp_tidy_long.csv"), index=False)

    summary = (
        df_all
        .groupby(['ID', 'Concentration', 'Units', 'Step'], as_index=False)['Potential/V']
        .agg(Mean_OCP='mean', SD_OCP='std')
    )
    summary.to_csv(os.path.join(final_path, "ocp_final.csv"), index=False)

    compute_ocp_slopes(final_path)

    print(f"\n4-channel OCP data saved to {final_path}/")
    print(f"  Segments detected: {n_segments}")
    print(f"  Concentration sequence: {v_shape}")
