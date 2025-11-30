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
                "Do you want to exit the header editing mode?",
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
            if exit_prompt():
                edit_mode = False
            continue
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


def select_x_y_headers(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    headers = df.columns.tolist()
    x_header = questionary.select(
        "Select X header",
        choices=headers + ["[Cancel]"]
    ).ask()
    if x_header is None or x_header == "[Cancel]":
        print("X header selection cancelled.")
        return None, None
    
    y_header = questionary.select(
        "Select Y header",
        choices=headers + ["[Cancel]"]
    ).ask()
    if y_header is None or y_header == "[Cancel]":
        print("Y header selection cancelled.")
        return None, None
    
    return x_header, y_header


def select_legend(df: pd.DataFrame) -> Optional[str]:
    """
    Select a legend header from the DataFrame.
    Selection is limited to Object type headers
    """
    headers = [col for col in df.columns if df.dtypes[col] == 'object']
    legend = questionary.select(
        "Select legend header (optional)",
        choices=headers + ["[None]"]
    ).ask()
    if legend is None or legend == "[None]":
        return None
    return legend


def get_xtick_rotation(x_values, max_labels=10, max_label_len=8):
    """
    Decide xtick rotation based on label count and length.
    Returns 45 if any label is long or there are many labels, else 0.
    """
    unique_labels = pd.Series(x_values).unique()
    if len(unique_labels) > max_labels or any(len(str(lbl)) > max_label_len for lbl in unique_labels):
        return 45
    return 0


#region: plotting functions

def line_plot(df) -> None:
    """
    Create a line plot for the selected X and Y headers.
    """
    x_header, y_header = select_x_y_headers(df)
    if x_header is None or y_header is None:
        print("Line plot creation cancelled.")
        return
    
    legend = select_legend(df)
    if legend is not None:
        if legend not in df.columns:
            print(f"{Fore.LIGHTYELLOW_EX}Legend header '{legend}' does not exist. No legend will be used.")
            legend = None
    try:
        spinner.start("Creating line plot...")
        plt.figure(figsize=(10, 6))
        if legend:
            sns.lineplot(data=df, x=x_header, y=y_header, hue=legend)
            plt.title(f"Line Plot: {y_header} vs {x_header} by {legend}")
        else:
            sns.lineplot(data=df, x=x_header, y=y_header)
            plt.title(f"Line Plot: {y_header} vs {x_header}")
        plt.xlabel(x_header)
        plt.ylabel(y_header)
        plt.xticks(rotation=get_xtick_rotation(df[x_header]))
        plt.grid(True)
        spinner.succeed("Line plot created successfully. Close window to continue.")
        plt.show()
    except Exception as e:
        spinner.fail(f"{Fore.LIGHTYELLOW_EX}Error creating line plot: {e}")


def bar_plot(df) -> None:
    """
    Create a bar plot for the selected X and Y headers.
    """
    x_header, y_header = select_x_y_headers(df)
    if x_header is None or y_header is None:
        print("Bar plot creation cancelled.")
        return
    
    legend = select_legend(df)
    if legend is not None:
        if legend not in df.columns:
            print(f"{Fore.LIGHTYELLOW_EX}Legend header '{legend}' does not exist. No legend will be used.")
            legend = None
    try:
        spinner.start("Creating bar plot...")
        plt.figure(figsize=(10, 6))
        if legend:
            sns.barplot(data=df, x=x_header, y=y_header, hue=legend)
            plt.title(f"Bar Plot: {y_header} vs {x_header} by {legend}")
        else:
            sns.barplot(data=df, x=x_header, y=y_header)
            plt.title(f"Bar Plot: {y_header} vs {x_header}")
        plt.xlabel(x_header)
        plt.ylabel(y_header)
        plt.xticks(rotation=get_xtick_rotation(df[x_header]))
        plt.grid(True)
        spinner.succeed("Bar plot created successfully. Close window to continue.")
        plt.show()
    except Exception as e:
        spinner.fail(f"{Fore.LIGHTYELLOW_EX}Error creating bar plot: {e}")


def scatter_plot(df) -> None:
    """
    Create a scatter plot for the selected X and Y headers.
    """
    x_header, y_header = select_x_y_headers(df)
    if x_header is None or y_header is None:
        print("Scatter plot creation cancelled.")
        return
    
    legend = select_legend(df)
    if legend is not None:
        if legend not in df.columns:
            print(f"{Fore.LIGHTYELLOW_EX}Legend header '{legend}' does not exist. No legend will be used.")
            legend = None
    try:
        spinner.start("Creating scatter plot...")
        plt.figure(figsize=(10, 6))
        if legend:
            sns.scatterplot(data=df, x=x_header, y=y_header, hue=legend)
            plt.title(f"Scatter Plot: {y_header} vs {x_header} by {legend}")
        else:
            sns.scatterplot(data=df, x=x_header, y=y_header)
            plt.title(f"Scatter Plot: {y_header} vs {x_header}")
        plt.xlabel(x_header)
        plt.ylabel(y_header)
        plt.xticks(rotation=get_xtick_rotation(df[x_header]))
        plt.grid(True)
        spinner.succeed("Scatter plot created successfully. Close window to continue.")
        plt.show()
    except Exception as e:
        spinner.fail(f"{Fore.LIGHTYELLOW_EX}Error creating scatter plot: {e}")


def pair_plot(df: pd.DataFrame) -> None:
    """
    Create a pair plot for the DataFrame.
    This function uses seaborn's pairplot to visualize pairwise relationships in the dataset.
    """
    # Check if there are too many numeric columns
    numerical_threshold = 6
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > numerical_threshold:
        numerical_col_subset = questionary.checkbox(
            f"There are {len(numerical_cols)} numeric columns which may clutter the plot. Anything over {numerical_threshold} is not recommended.\nPlease verify which columns to render:",
            choices = numerical_cols
        ).ask()
        if numerical_col_subset is None or len(numerical_col_subset) < 2:
            user = questionary.confirm(
                "Not enough columns selected. Do you want to use all the columns instead?",
                default=False
            ).ask()
            if user is None or user is False:
                print("Pair plot creation cancelled.")
                return
        else:
            # Find columns that were not selected and drop them
            cols_to_drop = list(set(numerical_cols) - set(numerical_col_subset))
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

    legend = select_legend(df)
    if legend is not None:
        if legend not in df.columns:
            print(f"{Fore.LIGHTYELLOW_EX}Legend header '{legend}' does not exist. No legend will be used.")
            legend = None
    spinner.start("Creating pair plot...")
    try:
        if legend:
            sns.pairplot(df, hue=legend)
            plt.suptitle(f"Pair Plot with Legend: {legend}", y=1.02) # Adjust title position
        else:
            sns.pairplot(df)
            plt.suptitle("Pair Plot", y=1.02)  # Adjust title position
        spinner.succeed("Pair plot created successfully. Close window to continue.")
        plt.show()
    except Exception as e:
        spinner.fail(f"{Fore.LIGHTYELLOW_EX}Error creating pair plot: {e}")


def box_plot(df: pd.DataFrame) -> None:
    """
    Create a box plot for the DataFrame.
    This function uses seaborn's boxplot to visualize the distribution of data across different categories.
    """
    headers = df.columns.tolist()
    x_header = questionary.select(
        "Select X header for box plot",
        choices=headers + ["[None]"]
    ).ask()
    if x_header == "[None]":
        x_header = None
    
    y_header = questionary.select(
        "Select Y header for box plot",
        choices=headers + ["[Cancel]"]
    ).ask()
    if y_header is None or y_header == "[Cancel]":
        print("Box plot creation cancelled.")
        return
    # if x_header not in df.columns or y_header not in df.columns:
    #     print(f"{Fore.LIGHTYELLOW_EX}Selected headers do not exist in the DataFrame. No box plot will be created.")
    #     return
    
    legend = select_legend(df)
    if legend is not None:
        if legend not in df.columns:
            print(f"{Fore.LIGHTYELLOW_EX}Legend header '{legend}' does not exist. No legend will be used.")
            legend = None
    try:
        spinner.start("Creating box plot...")
        plt.figure(figsize=(10, 6))
        if legend:
            sns.boxplot(data=df, x=x_header, y=y_header, hue=legend)
            if x_header:
                plt.title(f"Box Plot: {y_header} by {x_header} with Legend: {legend}")
            else:
                plt.title(f"Box Plot: {y_header} with Legend: {legend}")
        else:
            sns.boxplot(data=df, x=x_header, y=y_header)
            if x_header:
                plt.title(f"Box Plot: {y_header} by {x_header}")
            else:
                plt.title(f"Box Plot: {y_header}")
        if x_header is not None: 
            plt.xlabel(x_header)
        plt.ylabel(y_header)
        plt.xticks(rotation=get_xtick_rotation(df[x_header]))
        plt.grid(True)
        spinner.succeed("Box plot created successfully. Close window to continue.")
        plt.show()
    except Exception as e:
        spinner.fail(f"{Fore.LIGHTYELLOW_EX}Error creating box plot: {e}")


def violin_plot(df: pd.DataFrame) -> None:
    """
    Create a violin plot for the DataFrame.
    This function uses seaborn's violinplot to visualize the distribution of data across different categories.
    """
    headers = df.columns.tolist()
    x_header = questionary.select(
        "Select X header for violin plot",
        choices=headers + ["[None]"]
    ).ask()
    if x_header == "[None]":
        x_header = None
    
    y_header = questionary.select(
        "Select Y header for violin plot",
        choices=headers + ["[Cancel]"]
    ).ask()
    if y_header is None or y_header == "[Cancel]":
        print("Violin plot creation cancelled.")
        return
    # if x_header not in df.columns or y_header not in df.columns:
    #     print(f"{Fore.LIGHTYELLOW_EX}Selected headers do not exist in the DataFrame. No violin plot will be created.")
    #     return
    
    legend = select_legend(df)
    if legend is not None:
        if legend not in df.columns:
            print(f"{Fore.LIGHTYELLOW_EX}Legend header '{legend}' does not exist. No legend will be used.")
            legend = None
    try:
        spinner.start("Creating violin plot...")
        plt.figure(figsize=(10, 6))
        if legend:
            sns.violinplot(data=df, x=x_header, y=y_header, hue=legend)
            if x_header:
                plt.title(f"Violin Plot: {y_header} by {x_header} with Legend: {legend}")
            else:
                plt.title(f"Violin Plot: {y_header} with Legend: {legend}")
        else:
            sns.violinplot(data=df, x=x_header, y=y_header)
            if x_header:
                plt.title(f"Violin Plot: {y_header} by {x_header}")
            else:
                plt.title(f"Violin Plot: {y_header}")
        if x_header is not None:
            plt.xlabel(x_header)
        plt.ylabel(y_header)
        plt.xticks(rotation=get_xtick_rotation(df[x_header]))
        plt.grid(True)
        spinner.succeed("Violin plot created successfully. Close window to continue.")
        plt.show()
    except Exception as e:
        spinner.fail(f"{Fore.LIGHTYELLOW_EX}Error creating violin plot: {e}")


def distribution_plot(df: pd.DataFrame) -> None:
    """
    Create a distribution plot for the DataFrame.
    This function uses seaborn's displot to visualize the distribution of a selected numeric column.
    """
    headers = df.select_dtypes(include=[np.number]).columns.tolist()
    if not headers:
        print(f"{Fore.LIGHTYELLOW_EX}No numeric columns found in the DataFrame. No distribution plot will be created.")
        return
    x_header = questionary.select(
        "Select header for distribution plot",
        choices=headers + ["[Cancel]"]
    ).ask()
    if x_header is None or x_header == "[Cancel]":
        print("Distribution plot creation cancelled.")
        return
    if x_header not in df.columns:
        print(f"{Fore.LIGHTYELLOW_EX}Selected header '{x_header}' does not exist in the DataFrame. No distribution plot will be created.")
        return
    legend = select_legend(df)
    if legend is not None and legend != "[None]":
        if legend not in df.columns:
            print(f"{Fore.LIGHTYELLOW_EX}Legend header '{legend}' does not exist. No legend will be used.")
            legend = None

    spinner.start("Creating distribution plot...")
    try:
        if legend:
            g = sns.displot(df, x=x_header, hue=legend, kde=True, height=6, aspect=1.6)
            g.fig.suptitle(f"Distribution Plot: {x_header} by {legend}", y=0.98)
        else:
            g = sns.displot(df[x_header], kde=True, height=6, aspect=1.6)
            g.fig.suptitle(f"Distribution Plot: {x_header}", y=0.98)
        g.set_axis_labels(x_header, "Density")
        g.fig.tight_layout()  # Helps prevent title cutoff
        spinner.succeed("Distribution plot created successfully. Close window to continue.")
        plt.show()
    except Exception as e:
        spinner.fail(f"{Fore.LIGHTYELLOW_EX}Error creating distribution plot: {e}")


def joint_grid_plot(df: pd.DataFrame) -> None:
    """
    Create a joint grid plot for the DataFrame.
    X-axis can be numeric or categorical, Y-axis must be numeric.
    """
    x_header = questionary.select(
        "Select X header for joint grid plot (numeric or categorical)",
        choices=df.columns.tolist() + ["[Cancel]"]
    ).ask()
    if x_header is None or x_header == "[Cancel]":
        print("Operation cancelled.")
        return
    y_header = questionary.select(
        "Select Y header for joint grid plot (numeric only)",
        choices=df.select_dtypes(include=[np.number]).columns.tolist() + ["[Cancel]"]
    ).ask()
    if y_header is None or y_header == "[Cancel]":
        print("Operation cancelled.")
        return
    if x_header not in df.columns or y_header not in df.columns:
        print(f"{Fore.LIGHTYELLOW_EX}Selected headers do not exist in the DataFrame. No joint grid plot will be created.")
        return
    try:
        spinner.start("Creating joint grid plot...")
        g = sns.jointplot(data=df, x=x_header, y=y_header, kind="hist", height=8)
        g.plot_joint(sns.histplot, cmap=sns.dark_palette("#69d", reverse=True, as_cmap=True), cbar=True)
        g.plot_marginals(sns.histplot, element="step")
        g.figure.suptitle(f"Joint Grid Plot: {y_header} vs {x_header}", y=1.02)  # Adjust title position
        g.set_axis_labels(x_header, y_header)
        spinner.succeed("Joint grid plot created successfully. Close window to continue.")
        plt.show()
    except Exception as e:
        spinner.fail(f"{Fore.LIGHTYELLOW_EX}Error creating joint grid plot: {e}")


def relation_plot(df: pd.DataFrame) -> None:
    """
    Create a relation plot for the DataFrame.
    X and Y axis must be numeric.
    Marker color can be set to a categorical header.
    Marker size can be set to a numeric header.
    """
    x_header = questionary.select(
        "Select X header for relation plot (numeric only)",
        choices=df.select_dtypes(include=[np.number]).columns.tolist() + ["[Cancel]"]
    ).ask()
    if x_header is None or x_header == "[Cancel]":
        print("Operation cancelled.")
        return
    y_header = questionary.select(
        "Select Y header for relation plot (numeric only)",
        choices=df.select_dtypes(include=[np.number]).columns.tolist() + ["[Cancel]"]
    ).ask()
    if y_header is None or y_header == "[Cancel]":
        print("Operation cancelled.")
        return
    if x_header not in df.columns or y_header not in df.columns:
        print(f"{Fore.LIGHTYELLOW_EX}Selected headers do not exist in the DataFrame. No relation plot will be created.")
        return
    legend = select_legend(df)
    if legend == "[None]":
        legend = None
    size_header = questionary.select(
        "Select size header for relation plot (numeric only, optional)",
        choices=df.select_dtypes(include=[np.number]).columns.tolist() + ["[None]"]
    ).ask()
    if size_header is None or size_header == "[None]":
        size_header = None
    elif size_header not in df.columns:
        print(f"{Fore.LIGHTYELLOW_EX}Size header '{size_header}' does not exist. No size will be used.")
        size_header = None
    try:
        spinner.start("Creating relation plot...")
        sns.relplot(
            data=df,
            x=x_header,
            y=y_header,
            hue=legend,
            size=size_header,
            kind="scatter",
            height=6,
            aspect=1.6
        )
        plt.title(f"Relation Plot: {y_header} vs {x_header}" + (f" by {legend}" if legend else ""))
        plt.xlabel(x_header)
        plt.ylabel(y_header)
        plt.grid(True)
        spinner.succeed("Relation plot created successfully. Close window to continue.")
        plt.show()
    except Exception as e:
        spinner.fail(f"{Fore.LIGHTYELLOW_EX}Error creating relation plot: {e}")










#endregion: plotting functions



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