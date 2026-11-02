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
            # Load data as array
            time_array = m.dataset['Time'].to_list()
            current_array = m.dataset['Current'].to_list()
            print(time_array[:5])
            print(current_array[:5])
            # Create df and assign col.
            time = pd.DataFrame(time_array, columns=['Time/sec'])
            current = pd.DataFrame(current_array, columns=['Current/A'])
            
            array = pd.concat([time, current], axis=1)
            array['Current/A'] = array['Current/A'] * 1e-6 # current data is stored as uA by PalmSens
            
            array.to_csv(f"{output_path}/{filename[:-10]}.csv", index=False)
            

# convert_pssession_to_csv("9_July", "testin", "testout")