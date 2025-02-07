from concurrent.futures import ThreadPoolExecutor
import time
import pandas as pd
from numpy.typing import DTypeLike
import csv
from zipfile import ZipFile
from io import TextIOWrapper

def load_data(path: str, columns: dict[str,DTypeLike], missing: dict[str,set[str]]) -> pd.DataFrame:
    """Loads the raw dataset from files using parameters from the config file.
    
    Positional Arguments:
    path    - The path to the directory or file where your data is located
    columns - The columns from the dataset you plan to use mapped to their data types
    missing - The columns from the dataset mapped to values that indicate data is missing

    returns a DataFrame loaded from the filepath given with the specified columns and types
    """
    start_time = time.time()
    series = []

    with ZipFile(path, 'r') as zar:
        file_names = zar.namelist()
        print(f"Files in archive: {file_names}")

        with zar.open(file_names[0], 'r') as bin_fin:
            fin = TextIOWrapper(bin_fin, encoding="utf-8")
            csv_reader = csv.DictReader(fin)
            with ThreadPoolExecutor() as executor:  # Use ProcessPoolExecutor for CPU-bound
                series = list(executor.map(lambda row: pd.Series(row), csv_reader))

    df = pd.concat(series, axis=1).T  # Transpose to correct orientation
    df.columns = columns.keys()
    df = df.astype(columns)  # Enforce correct data types
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    return df
    

def save_data(df: pd.DataFrame, path: str) -> None:
    """Saves the transformed dataset into the specified output file"""
    df.to_csv(path)