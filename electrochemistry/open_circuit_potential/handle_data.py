import os
from io import StringIO

import pandas as pd


def _normalise_columns(df_temp):
    """Map PalmSens/CHI column variants onto the pipeline's canonical names."""
    rename_map = {
        "s": "Time/sec",
        "V": "Potential/V",
        "Voltage/V": "Potential/V",
    }
    return df_temp.rename(columns=rename_map)


def _process_files(source_dir, tidy_dir, file_pattern, filetype, batch_name=None):
    all_dataframes = []

    for filename in os.listdir(source_dir):
        if not filename.endswith(filetype):
            continue

        match = file_pattern.match(filename)
        if not match:
            print(f"Skipping {filename}: did not match filename pattern")
            continue

        file_id = match.group("id")
        conc = float(match.group("concentration"))
        units = str(match.group("units"))
        step = int(match.group("step"))

        full_path = os.path.join(source_dir, filename)

        try:
            with open(full_path, "r") as f:
                found_header = False
                for line in f:
                    clean_line = line.strip()
                    if "Time/sec" in clean_line and (
                        "Potential/V" in clean_line or "Voltage/V" in clean_line
                    ):
                        remaining_data = line + f.read()
                        df_temp = pd.read_csv(
                            StringIO(remaining_data), sep=",", skipinitialspace=True
                        )
                        found_header = True
                        break

                    # Native PalmSens CSV uses the short "s,V" header.
                    first_col = clean_line.split(",", 1)[0].strip()
                    if first_col == "s" and "V" in clean_line:
                        remaining_data = line + f.read()
                        df_temp = pd.read_csv(
                            StringIO(remaining_data), sep=",", skipinitialspace=True
                        )
                        found_header = True
                        break

                if not found_header:
                    print(
                        f"Skipping {filename}: could not find Time/sec and "
                        "Potential/V (or Voltage/V) header"
                    )
                    continue

            df_temp = _normalise_columns(df_temp)
            df_temp["ID"] = file_id
            df_temp["Concentration"] = conc
            df_temp["Units"] = units
            df_temp["Step"] = step
            if batch_name is not None:
                df_temp["Batch"] = batch_name
            df_temp["Time/sec"] = pd.to_numeric(df_temp["Time/sec"], errors="coerce")
            df_temp["Potential/V"] = pd.to_numeric(
                df_temp["Potential/V"], errors="coerce"
            )
            df_temp = df_temp.dropna(subset=["Time/sec", "Potential/V"])

            all_dataframes.append(df_temp)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not all_dataframes:
        print(f"No data found in {source_dir} - check file naming and folder path")
        return

    master_df = pd.concat(all_dataframes, ignore_index=True)
    cols = ["Time/sec", "Potential/V", "ID", "Concentration", "Units", "Step"]
    if batch_name is not None:
        cols = ["Time/sec", "Potential/V", "Batch"] + cols[2:]
    master_df = master_df[cols]

    os.makedirs(tidy_dir, exist_ok=True)

    for id_val, group_df in master_df.groupby("ID"):
        final_file = f"ID_{str(id_val).strip()}.csv"
        output_path = os.path.join(tidy_dir, final_file)
        group_df.to_csv(output_path, index=False)
        label = f"[{batch_name}] " if batch_name else ""
        print(f"  {label}Saved data for ID '{id_val}' to {output_path}")


def format_data(subfolder, raw_data_folder, tidy_data_folder, file_pattern, filetype):
    raw_data_folder = os.path.join("Data", subfolder, raw_data_folder)
    tidy_data_folder = os.path.join("Data", subfolder, tidy_data_folder)

    entries = sorted(os.listdir(raw_data_folder))
    batch_dirs = [
        e for e in entries if os.path.isdir(os.path.join(raw_data_folder, e))
    ]

    if batch_dirs:
        print(f"Batch mode: detected {len(batch_dirs)} subfolder(s): {batch_dirs}")
        for batch_name in batch_dirs:
            source = os.path.join(raw_data_folder, batch_name)
            dest = os.path.join(tidy_data_folder, batch_name)
            print(f"  Processing batch '{batch_name}'...")
            _process_files(
                source, dest, file_pattern, filetype, batch_name=batch_name
            )
    else:
        print("No batch subfolders detected - using flat mode")
        _process_files(raw_data_folder, tidy_data_folder, file_pattern, filetype)
