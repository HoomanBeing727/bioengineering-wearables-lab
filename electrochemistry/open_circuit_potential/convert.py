import pypalmsens as pyp
import pandas as pd 
import os 

def convert_pssession_to_csv(subfolder, pssess_data, raw_data):
    input_path = os.path.join("Data", subfolder, pssess_data)
    output_path = os.path.join("Data", subfolder, raw_data)
    os.makedirs(output_path, exist_ok=True)
    for filename in os.listdir(input_path):
        if filename.endswith('.pssession'):
            print(f"Found session file: {filename}")

            measurements = pyp.load_session_file(f'{input_path}/{filename}')

            for m in measurements:
                time_array = m.dataset['Time'].to_list()
                current_array = m.dataset['Voltage'].to_list()
                
                time = pd.DataFrame(time_array, columns=['Time/sec'])
                current = pd.DataFrame(current_array, columns=['Voltage/V'])
                
                array = pd.concat([time, current], axis=1)
                
                array.to_csv(f"{output_path}/{filename[:-10]}.csv", index=False)
        