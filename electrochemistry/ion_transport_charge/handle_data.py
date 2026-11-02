import pandas as pd 
import os 
from io import StringIO
 
def format_data(subfolder, raw_data_folder, tidy_data_folder, file_pattern, filetype):
    all_dataframes = []
    
    raw_data_folder = os.path.join("Data", subfolder, raw_data_folder)
    tidy_data_folder = os.path.join("Data", subfolder, tidy_data_folder)
    for filename in os.listdir(raw_data_folder):
        if filename.endswith(filetype):
        
            match = file_pattern.match(filename)
            
            if match:
                id = match.group("id")    
                conc = float(match.group("concentration"))     
                units = str(match.group("units"))     

                full_path = os.path.join(raw_data_folder, filename)
                
                try: 
                    with open(full_path, 'r') as f:
                        found_header = False
                        for line in f:
                            clean_line = line.strip()
                            if 'Time/sec' in clean_line and 'Current/A' in clean_line:
                    
                                remaining_data = line + f.read()
                                file_stream = StringIO(remaining_data)
                                
                                found_header = True
                                df_temp = pd.read_csv(file_stream, sep=',', skipinitialspace=True)
                                break
                            
                            # Handle data from PalmSense
                            elif 's' in clean_line and 'µA' in clean_line:
                                remaining_data = line + f.read()
                                file_stream = StringIO(remaining_data)
                                
                                found_header = True
                                df_temp = pd.read_csv(file_stream, sep=',', skipinitialspace=True)
                                df_temp.rename(columns={'s': 'Time/sec', 'µA': 'Current/A'}, inplace=True)
                                df_temp['Current/A'] = df_temp['Current/A'] * 1e-6
                                break
                        
                        if not found_header:
                            print(f"Skipping {filename}: Could not find header row containing 'Time/sec' and 'Current/A'")
                            continue

                    df_temp['ID'] = id
                    df_temp['Concentration'] = conc
                    df_temp['Units'] = units
                    all_dataframes.append(df_temp)               
                    
                except Exception as e: 
                    print(f"Error processing {filename}: {e}")

    if not all_dataframes:
        print("No data found - check file naming and folder path")
    else: 
        master_df = pd.concat(all_dataframes, ignore_index=True)
        master_df = master_df[['Time/sec', 'Current/A', 'ID', 'Concentration', 'Units']]
        
        os.makedirs(tidy_data_folder, exist_ok=True)
        
        for id, group_df in master_df.groupby('ID'):
            final_file = f"ID_{str(id).strip()}.csv"
            output_path = os.path.join(tidy_data_folder, final_file)
            
            group_df.to_csv(output_path, index=False)
            print(f"Saved data for ID '{id}' to {output_path}")
            

