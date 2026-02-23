# MIT License
# Copyright (c) 2025 espehon

#region: imports
import os
import sys
import argparse
import importlib.metadata
from typing import Optional, Union, List, Dict, Any


from colorama import Fore, Back, Style, init as colorama_init
colorama_init(autoreset=True)
import questionary
from halo import Halo
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# import openpyxl 


#endregion: imports
#region: startup

try:
    __version__ = f"tasky {importlib.metadata.version('audity')} from audity"
except importlib.metadata.PackageNotFoundError:
    __version__ = "Package not installed..."



spinner = Halo("Loading...", spinner="dots")


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
    dtypes_row = pd.DataFrame([df.dtypes.astype(str).values], columns=df.columns, index=['idx/type'])
    # Get the sample
    sample = df.sample(n)
    sample.sort_index(inplace=True)  # Sort sample by index for better readability
    # Concatenate dtypes row and sample
    preview = pd.concat([dtypes_row, sample])
    
    # Estimate width: assume 12 chars per column (adjust as needed)
    est_width = len(preview.columns) * 12
    try:
        term_width = os.get_terminal_size().columns
    except OSError:
        term_width = 80  # fallback if terminal size can't be determined

    print()
    if est_width > term_width:
        print(preview.T)
        print(f"\nShape: {df.shape} | Output was transposed to fit on screen.")
    else:
        print(preview)
        print(f"\nShape: {df.shape}")


def print_describe_with_dtypes(df: pd.DataFrame) -> None:
    """
    Print a describe() summary with dtypes as the second row under each header.
    """
    desc = df.describe(include='all')
    # Create a DataFrame with one row: the dtypes as strings
    dtypes_row = pd.DataFrame([df.dtypes.astype(str).values], columns=df.columns, index=['dtype'])
    # Reindex desc to start with dtype row, then the rest of describe
    combined = pd.concat([dtypes_row, desc])

    # Desired row order (add other stats as needed)
    row_order = ['dtype', 'count', 'unique', 'top', 'freq', 'mean', 'std', 'max', '75%', '50%', '25%', 'min']
    # Only keep rows that exist in combined
    row_order = [row for row in row_order if row in combined.index]
    combined = combined.reindex(row_order)

    # Estimate width: assume 12 chars per column (adjust as needed)
    est_width = len(combined.columns) * 12
    try:
        term_width = os.get_terminal_size().columns
    except OSError:
        term_width = 80  # fallback if terminal size can't be determined

    print()
    if est_width > term_width:
        print(combined.T)
        print(f"\nShape: {df.shape} | Output was transposed to fit on screen.")
    else:
        print(combined)
        print(f"\nShape: {df.shape}")



def load_dataset(file_path: str) -> pd.DataFrame:
    try:
        spinner.start(text=f"Loading dataset...")
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
        spinner.fail(f"{Back.LIGHTYELLOW_EX}{Fore.BLACK}Error loading dataset{Style.RESET_ALL}")
        print(f"\n{Fore.LIGHTYELLOW_EX}{e}")
        sys.exit(1)
    spinner.succeed("Dataset loaded successfully.")
    return df


def header_rename(df: pd.DataFrame) -> pd.DataFrame:
    old_header = questionary.select(
        "Select header to rename:",
        choices=[{"name": col} for col in df.columns] + ["[Cancel]"]
    ).ask()
    if old_header is None or old_header == "[Cancel]":
        print(f"{Fore.LIGHTYELLOW_EX}Header renaming cancelled.")
        return df
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


def header_type_change(df: pd.DataFrame) -> pd.DataFrame:
    headers = questionary.checkbox(
        "Select headers to change type (leave empty to cancel):",
        choices=[{"name": col} for col in df.columns]
    ).ask()
    if headers is None or len(headers) == 0:
        print(f"{Fore.LIGHTYELLOW_EX}No headers selected for type change. Operation cancelled.")
        return df
    changes = 0
    if len(headers) >= 1:
        for header in headers:
            new_type = questionary.select(
                    f"Select new type for '{header}':",
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
                continue
            if new_type not in ['int64', 'float64', 'object', 'datetime64[ns]', 'category']:
                print(f"{Fore.LIGHTYELLOW_EX}Unsupported type '{new_type}'. Operation cancelled.")
                continue
            try:
                if new_type == 'int64':
                    df[header] = pd.to_numeric(df[header], errors='coerce').astype('Int64')
                elif new_type == 'float64':
                    df[header] = df[header].astype('float64')
                elif new_type == 'object':
                    df[header] = df[header].astype('object')
                elif new_type == 'datetime64[ns]':
                    df[header] = pd.to_datetime(df[header], errors='coerce')
                elif new_type == 'category':
                    df[header] = df[header].astype('category')
                print(f"{Fore.GREEN}Header '{header}' type changed to '{new_type}'.")
                changes += 1
            except Exception as e:
                print(f"{Back.LIGHTYELLOW_EX}{Fore.BLACK}Error changing header type:{Style.RESET_ALL}\n{Fore.LIGHTYELLOW_EX}{e}")
    print(f"{Fore.GREEN}Changed types for {changes} headers.")
    return df


def header_delete(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_delete = questionary.checkbox(
        "Select headers to delete (leave empty to cancel):",
        choices=[{"name": col} for col in df.columns]
    ).ask()
    if columns_to_delete is None or len(columns_to_delete) == 0:
        print(f"{Fore.LIGHTYELLOW_EX}No headers selected for deletion. Operation cancelled.")
        return df
    if len(columns_to_delete) >= 1:
        user = questionary.confirm(
            f"You selected {len(columns_to_delete)} headers to delete. Are you sure?",
            default=False
        ).ask()
        if user is None or not user:
            print(f"{Fore.LIGHTYELLOW_EX}Operation cancelled.")
            return df
        df.drop(columns=columns_to_delete, inplace=True)
        print(f"{Fore.GREEN}Deleted {len(columns_to_delete)} headers: {', '.join(columns_to_delete)}.")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.
    """
    columns_to_check = questionary.checkbox(
        "Select columns to check for duplicates (leave empty to check all columns):",
        choices=[{"name": col} for col in df.columns]
    ).ask()
    if columns_to_check is None:
        user = questionary.confirm(
            "No columns selected. Do you want to check all columns for duplicates?",
            default=False
        ).ask()
        if user is None or not user:
            print(f"{Fore.LIGHTYELLOW_EX}No columns selected. No duplicates will be checked.")
            return df
        columns_to_check = df.columns.tolist()

    initial_count = len(df)
    df = df.drop_duplicates(subset=columns_to_check, keep='first')
    final_count = len(df)
    if initial_count == final_count:
        print(f"{Fore.LIGHTYELLOW_EX}No duplicates found.")
    else:
        print(f"{Fore.GREEN}Removed {initial_count - final_count} duplicate rows.")
    return df


def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing values from the DataFrame.
    """
    columns_to_check = questionary.checkbox(
        "Select columns to check for missing values (leave empty to check all columns):",
        choices=[{"name": col} for col in df.columns]
    ).ask()
    if columns_to_check is None:
        user = questionary.confirm(
            "No columns selected. Do you want to check all columns for missing values?",
            default=False
        ).ask()
        if user is None or not user:
            print(f"{Fore.LIGHTYELLOW_EX}No columns selected. No missing values will be checked.")
            return df
        columns_to_check = df.columns.tolist()
    
    initial_count = len(df)
    df = df.dropna(subset=columns_to_check, how='any')
    final_count = len(df)
    if initial_count == final_count:
        print(f"{Fore.LIGHTYELLOW_EX}No missing values found.")
    else:
        print(f"{Fore.GREEN}Removed {initial_count - final_count} rows with missing values.")
    return df


def remove_outliers_IQR(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers from the DataFrame using the IQR method.
    """
    columns_to_check = questionary.checkbox(
        "Select columns to check for outliers (leave empty to check all numeric columns):",
        choices=[{"name": col} for col in df.select_dtypes(include=[np.number]).columns]
    ).ask()
    if columns_to_check is None:
        user = questionary.confirm(
            "No columns selected. Do you want to check all numeric columns for outliers?",
            default=False
        ).ask()
        if user is None or not user:
            print(f"{Fore.LIGHTYELLOW_EX}No columns selected. No outliers will be checked.")
            return df
        columns_to_check = df.select_dtypes(include=[np.number]).columns.tolist()

    initial_count = len(df)
    for col in columns_to_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    final_count = len(df)
    if initial_count == final_count:
        print(f"{Fore.LIGHTYELLOW_EX}No outliers found.")
    else:
        print(f"{Fore.GREEN}Removed {initial_count - final_count} rows with outliers.")
    return df


def melt_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Melt (un-pivot) columns into long format.
    This function asks the user to select 2 or more columns to melt.
    The dataset is then expanded into long format which replaces the selected columns with 2 new columns.
        1. A new column that contains the selected column names as categorical values.
        2. A new column that contains the corresponding values from the selected columns.
    The user is prompted to name the new columns prior to melting.
    '''
    columns_to_melt = questionary.checkbox(
        "Select at least 2 columns to melt",
        choices=[{'name': col} for col in df.columns]
    ).ask()
    if columns_to_melt is None or len(columns_to_melt) < 2:
        print(f"{Fore.LIGHTYELLOW_EX}At least 2 columns must be selected to melt. Operation cancelled.")
        return df
    
    new_category_name = questionary.text(
        "Enter name for the new column that will contain the melted column names:"
    ).ask()
    if new_category_name is None or new_category_name.strip() == "":
        print(f"{Fore.LIGHTYELLOW_EX}New column name cannot be empty. Operation cancelled.")
        return df
    new_category_name = new_category_name.strip()

    new_value_name = questionary.text(
        "Enter name for the new column that will contain the melted values:"
    ).ask()
    if new_value_name is None or new_value_name.strip() == "":
        print(f"{Fore.LIGHTYELLOW_EX}New column name cannot be empty. Operation cancelled.")
        return df
    new_value_name = new_value_name.strip()

    # Melt the dataframe
    try:
        df_melted = pd.melt(
            df,
            value_vars=columns_to_melt,
            var_name=new_category_name,
            value_name=new_value_name
        )
        print(f"{Fore.GREEN}Melted {len(columns_to_melt)} columns into long format.")
        return df_melted
    except Exception as e:
        print(f"{Back.LIGHTYELLOW_EX}{Fore.BLACK}Error melting columns:{Style.RESET_ALL}\n{Fore.LIGHTYELLOW_EX}{e}")
        return df
    


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    def exit_prompt():
        answer = questionary.confirm(
                "Do you want to exit editing mode?",
                default=True
            ).ask()
        return answer
    
    print_describe_with_dtypes(df)
    
    edit_mode = True
    while edit_mode:
        # Action selection 
        user = questionary.select(
            f"What do you want to do?",
            choices=[
                "Sample dataset",
                "Edit header name",
                "Change header type",
                "Delete header",
                "Melt columns (un-pivot into long format)",
                "Remove duplicates",
                "Remove missing values",
                "Remove outliers (IQR method)",
                "[Done] Finish editing",
            ]
        ).ask()
        if user is None or user == "[Done] Finish editing":
            edit_mode = False
        if user == "Sample dataset":
            print_sample_with_dtypes(df)
        elif user == "Edit header name":
            df = header_rename(df)
        elif user == "Change header type":
            df = header_type_change(df)
        elif user == "Delete header":
            df = header_delete(df)
        elif user == "Melt columns (un-pivot into long format)":
            df = melt_columns(df)
        elif user == "Remove duplicates":
            df = remove_duplicates(df)
        elif user == "Remove missing values":
            df = remove_missing_values(df)
        elif user == "Remove outliers (IQR method)":
            df = remove_outliers_IQR(df)
        else:
            print(f"{Fore.LIGHTYELLOW_EX}Unknown action selected. Please try again.")
            continue
        print_describe_with_dtypes(df)
    print(f"\n {Fore.LIGHTGREEN_EX}Header editing completed.\n")
    return df

# selectors and plots are implemented in separate modules for clarity
from .ui import (
    select_axis_headers as select_x_y_headers,
    select_legend,
    get_xtick_rotation,
    select_size,
)
from .plots import (
    line_plot,
    bar_plot,
    scatter_plot,
    pair_plot,
    box_plot,
    violin_plot,
    distribution_plot,
    joint_grid_plot,
    relation_plot,
)







def audity(df: pd.DataFrame) -> None:
    """
    Main loop functions of audity.
    This function loops which lets the user choose and try different options.
    Options include changing headers, types, and plotting different charts
    """
    print('\n' * 3) # Print a new lines for better readability
    print(" Audity : Visualize and Audit Your Data ".center(os.get_terminal_size().columns),'-')
    print_sample_with_dtypes(df, n=10)
    print()

    features = [
        "Sample Dataframe",
        "Describe Dataframe",
        "Edit Dataframe",
        "Distribution Plot",
        "Box Plot",
        "Violin Plot",
        "Line Plot",
        "Bar Plot",
        "Scatter Plot",
        "Joint Grid Plot",
        "Relation Plot",
        # "Facet Plot",
        "Pair Plot",
        "Exit"
    ]

    # Main loop
    while True:
        print()
        user = questionary.select(
            "What do you want to do?",
            choices=features
        ).ask()
        if user is None:
            user = questionary.confirm(
                "Do you want to exit?",
                default=True
            ).ask()
            print("Exiting Audity. Goodbye!")
            break
        if user == "Exit":
            print("Exiting Audity. Goodbye!")
            break
        elif user == "Sample Dataframe":
            print_sample_with_dtypes(df, n=10)
        elif user == "Describe Dataframe":
            print_describe_with_dtypes(df)
        elif user == "Edit Dataframe":
            df = prepare_data(df)
            print("Updated dataframe preview:")
            print_sample_with_dtypes(df, n=5)
        elif user == "Line Plot":
            line_plot(df)
        elif user == "Bar Plot":
            bar_plot(df)
        elif user == "Scatter Plot":
            scatter_plot(df)
        elif user == "Pair Plot":
            pair_plot(df)
        elif user == "Box Plot":
            box_plot(df)
        elif user == "Violin Plot":
            violin_plot(df)
        elif user == "Distribution Plot":
            distribution_plot(df)
        elif user == "Joint Grid Plot":
            joint_grid_plot(df)
        elif user == "Relation Plot":
            relation_plot(df)
        # elif user == "Facet Plot":
        #     facet_plot(df)

        else:
            print(f"{Fore.LIGHTYELLOW_EX}Unknown option '{user}'. Please try again.")
            continue


def main(argv=None):
    data_file = browse_files()
    if data_file is None:
        print("No file selected. Exiting.")
        return 1
    df = load_dataset(data_file)
    audity(df)
    return 0

#endregion: functions