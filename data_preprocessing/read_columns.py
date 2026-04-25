import os
import pandas as pd

def display_data_columns(data_folder):
    if not os.path.exists(data_folder):
        print(f"The directory '{data_folder}' does not exist.")
        return

    print(f"Reading files from: {data_folder}\n")
    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)
        
        # The 14 data files are in CSV format
        if os.path.isfile(filepath) and filename.endswith('.csv'):
            try:
                # nrows=0 ensures we only read the header, making it extremely fast
                df = pd.read_csv(filepath, nrows=0)
                print(f"--- {filename} ({len(df.columns)} columns) ---")
                print(", ".join(df.columns.tolist()) + "\n")
            except Exception as e:
                print(f"Failed to read {filename}: {e}\n")

if __name__ == "__main__":
    # Go up one directory level from data_preprocessing/ to find the 'data' folder
    folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    display_data_columns(folder_path)