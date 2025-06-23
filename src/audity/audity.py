# MIT License
# Copyright (c) 2025 espehon

#region: imports
import os
import sys
import argparse
import importlib.metadata
from typing import Optional, Union, List, Dict, Any

from colorama import Fore, Back, Style, init as colorama_init
import questionary
from halo import Halo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl


#endregion: imports
#region: startup

try:
    __version__ = f"tasky {importlib.metadata.version('tasky_cli')} from tasky_cli"
except importlib.metadata.PackageNotFoundError:
    __version__ = "Package not installed..."

colorama_init(autoreset=True)


# Set argument parsing
parser = argparse.ArgumentParser(
    description="Audit, inspect, and survey data from the terminal.",
    epilog="Examples: \nHomepage: https://github.com/espehon/audity",
    allow_abbrev=False,
    add_help=False,
    usage="audity [option] <arguments>    'try: audity --help'",
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument('-?', '--help', action='help', help='Show this help message and exit.')
parser.add_argument('-v', '--version', action='version', version=__version__, help="Show package version and exit.")


#endregion: startup
#region: functions


def browse_files(start_path: str=".") -> Union[str, None]:
    if start_path is None:
        start_path = "."
    current_path = os.path.abspath(start_path)
    while True:
        entries = os.listdir(current_path)
        entries = sorted(entries, key=lambda x: (not os.path.isdir(os.path.join(current_path, x)), x.lower()))
        choices = []
        if os.path.dirname(current_path) != current_path:
            choices.append("[Go up] ..")
        for entry in entries:
            full_path = os.path.join(current_path, entry)
            if os.path.isdir(full_path):
                choices.append(f"[DIR] {entry}")
            else:
                choices.append(entry)
        selected = questionary.select(
            f"Select dataset: {current_path}",
            choices=choices + ["[Cancel]"],
        ).ask()
        if selected is None or selected == "[Cancel]":
            return None
        if selected == "[Go up] ..":
            current_path = os.path.dirname(current_path)
        elif selected.startswith("[DIR] "):
            current_path = os.path.join(current_path, selected[6:])
        else:
            return os.path.join(current_path, selected)


def print_sample_with_dtypes(df: pd.DataFrame, n: int = 5):
    # Create a DataFrame with one row: the dtypes as strings
    dtypes_row = pd.DataFrame([df.dtypes.astype(str).values], columns=df.columns)
    dtypes_row.index = ['index/type']
    # Get the sample
    sample = df.sample(n)
    # Concatenate dtypes row and sample
    preview = pd.concat([dtypes_row, sample])
    print(preview)


def load_dataset(file_path: str) -> pd.DataFrame:
    try:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = os.path.splitext(file_path)[1].lower()
    
        if file_extension in ['.csv']:
            raw_data = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            raw_data = pd.read_excel(file_path)
        elif file_extension in ['.json']:
            raw_data = pd.read_json(file_path)
        elif file_extension in ['.parquet']:
            raw_data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        print(f"Error loading dataset\n{e}")
        sys.exit(1)
    
    print_sample_with_dtypes(raw_data, n=5)





def main(argv=None):
    data_file = browse_files()
    if data_file is None:
        print("No file selected. Exiting.")
        return 1
    load_dataset(data_file)
    print("End of file...")
