# MIT License
# Copyright (c) 2025 espehon

"""
This is the core file for all plotting related functions.
It includes UI functions for selecting columns and legend lables as well.
"""

# standard library

# third party
from halo import Halo 
import questionary as q
import seaborn as sns
import pandas as pd
import numpy as np





def select_axis( df: pd.DataFrame, mode: str='X', num_only: bool=False, multiple: bool=False):
    """This function is used to select axis for plotting.
    Although rare, it can be used to select multiple columns for categorical plots with overlap (eg date hierarchy)"""
    mode = str.upper(mode)
    prompt = f"Select {mode}-axis {"(Num Only)" if num_only else ""}"

    if num_only:
        column_list = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        column_list = df.columns.tolist()

def select_legend( df: pd.DataFrame, num_only: bool=False, include_size: bool=False):
    """This function is used to select legend (hue) for plotting.
    if include_size is True, it will ask for size column as well (scatter plot)"""
    prompt = "Select Legend (Hue)"
    if num_only:
        column_list = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        column_list = df.columns.tolist()

    if include_size:
        prompt += " and Size"
        column_list.append("None (No Size)")

    legend_col = q.select(prompt, choices=column_list).ask()
    size_col = None
    if include_size:
        size_col = q.select("Select Size Column", choices=column_list).ask()
        if size_col == "None (No Size)":
            size_col = None

    return legend_col, size_col


