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
    print()
    print(preview)


def load_dataset(file_path: str) -> pd.DataFrame:
    try:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension in ['.csv']:
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension in ['.json']:
            df = pd.read_json(file_path)
        elif file_extension in ['.parquet']:
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        print(f"{Back.LIGHTYELLOW_EX}{Fore.BLACK}Error loading dataset{Style.RESET_ALL}\n{Fore.LIGHTYELLOW_EX}{e}")
        sys.exit(1)
    return df


def header_rename(df: pd.DataFrame, old_header: str) -> pd.DataFrame:
    new_header = questionary.text(
            f"Enter new header name for '{old_header}':",
            default=old_header
        ).ask()
    if new_header is None or new_header.strip() == "":
        print(f"{Fore.LIGHTYELLOW_EX}Header name cannot be empty. Operation cancelled.")
        return df
    if new_header == old_header:
        print(f"{Fore.LIGHTYELLOW_EX}New header name is the same as the old one. No changes made.")
        return df
    if new_header in df.columns:
        print(f"{Fore.LIGHTYELLOW_EX}Header '{new_header}' already exists. Operation cancelled.")
        return df
    df.rename(columns={old_header: new_header}, inplace=True)
    print(f"{Fore.GREEN}Header '{old_header}' renamed to '{new_header}'.")
    return df


def header_type_change(df: pd.DataFrame, header_name: str) -> pd.DataFrame:
    new_type = questionary.select(
            f"Select new type for '{header_name}':",
            choices=[
                "int64",
                "float64",
                "object",
                "datetime64[ns]",
                "category",
                "[Cancel]"
            ]
        ).ask()
    if new_type is None or new_type == "[Cancel]":
        print(f"Header type change cancelled.")
        return df
    if new_type not in ['int64', 'float64', 'object', 'datetime64[ns]', 'category']:
        print(f"{Fore.LIGHTYELLOW_EX}Unsupported type '{new_type}'. Operation cancelled.")
        return df
    try:
        if new_type == 'int64':
            df[header_name] = pd.to_numeric(df[header_name], errors='coerce').astype('Int64')
        elif new_type == 'float64':
            df[header_name] = df[header_name].astype('float64')
        elif new_type == 'object':
            df[header_name] = df[header_name].astype('object')
        elif new_type == 'datetime64[ns]':
            df[header_name] = pd.to_datetime(df[header_name], errors='coerce')
        elif new_type == 'category':
            df[header_name] = df[header_name].astype('category')
        print(f"{Fore.GREEN}Header '{header_name}' type changed to '{new_type}'.")
    except Exception as e:
        print(f"{Back.LIGHTYELLOW_EX}{Fore.BLACK}Error changing header type:{Style.RESET_ALL}\n{Fore.LIGHTYELLOW_EX}{e}")
    return df


def header_delete(df: pd.DataFrame, header_name: str) -> pd.DataFrame:
    confirm = questionary.confirm(
        f"Are you sure you want to delete the header '{header_name}'?",
        default=False
    ).ask()
    if not confirm:
        print("Header deletion cancelled.")
        return df
    if header_name not in df.columns:
        print(f"Header '{header_name}' does not exist. No changes made.")
        return df
    df.drop(columns=[header_name], inplace=True)
    print(f"{Fore.GREEN}Header '{header_name}' deleted.")
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    def exit_prompt():
        answer = questionary.confirm(
                "Do you want to exit the header editing mode?",
                default=True
            ).ask()
        return answer
    
    print_sample_with_dtypes(df, n=10)
    
    edit_mode = True
    while edit_mode:
        user = questionary.confirm(
        "Do you want to edit the dataset headers?",
        default=True
        ).ask()
        if not user:
            print("Exiting header editing mode.")
            edit_mode = False
            continue
        
        # Header selection
        header_types = df.dtypes.items()
        selected_header = questionary.select(
            "Select header to edit",
            choices=[f"{name} ({dtype})" for name, dtype in header_types] + ["[Finish]"]
        ).ask()
        if selected_header is None or selected_header == "[Finish]":
            if exit_prompt():
                edit_mode = False
            continue
        header_name = selected_header.split(" (")[0]

        # Action selection 
        user = questionary.select(
            f"What do you want to do with {selected_header}?",
            choices=[
                "Edit header name",
                "Change header type",
                "Delete header",
                "[Cancel]"
            ]
        ).ask()
        if user is None or user == "[Cancel]":
            if exit_prompt():
                edit_mode = False
            continue
        
        if user == "Edit header name":
            df = header_rename(df, header_name)
        elif user == "Change header type":
            df = header_type_change(df, header_name)
        elif user == "Delete header":
            df = header_delete(df, header_name)
        else:
            print(f"{Fore.LIGHTYELLOW_EX}Unknown action. Please try again.")
            continue
        print_sample_with_dtypes(df, n=5)
    print(f"\n {Fore.LIGHTGREEN_EX}Header editing completed.\n")
    return df
        










def main(argv=None):
    data_file = browse_files()
    if data_file is None:
        print("No file selected. Exiting.")
        return 1
    df = prepare_data(load_dataset(data_file))

    print("\nFinal dataset preview:")
    print_sample_with_dtypes(df, n=10)
    print("\nEnd of file...")
