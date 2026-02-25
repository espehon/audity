# MIT License
# Copyright (c) 2025 espehon

"""User interface helpers used across the package.

This module provides a small set of functions that present consistent
questionary-based prompts for selecting DataFrame columns.  They are
intentionally lightweight and do not perform any validation beyond what the
caller requests (e.g., restricting to numeric columns).  Having a centralized
implementation keeps the behaviour uniform and makes it easy to extend later
if additional options or styles are required.

"""

from __future__ import annotations

from typing import Optional, Tuple, List, Literal

import pandas as pd
import numpy as np
from colorama import Fore
import questionary


def select_axis_headers(
    df: pd.DataFrame,
    x_label: str = "X",
    y_label: str = "Y",
    x_type: Literal["numeric", "categorical", "any"] = "any",
    y_type: Literal["numeric", "categorical", "any"] = "any",
    allow_none_x: bool = False,
    allow_none_y: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Ask the user to choose headers for X and Y axes.

    Parameters:

    * ``x_type`` and ``y_type`` control column filtering:
      - "numeric": only numeric columns
      - "categorical": only object/category dtype columns
      - "any": all columns
    * ``allow_none_*`` adds a ``"[None]"`` option and returns ``None`` when
      selected (useful for plots that don't require one of the axes).

    On cancellation the pair ``(None, None)`` is returned.
    """

    all_headers: List[str] = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        col for col in all_headers
        if df[col].dtype == "object" or df[col].dtype.name in ("category", "string")
    ]

    def _get_filtered_headers(col_type: Literal["numeric", "categorical", "any"]) -> List[str]:
        if col_type == "numeric":
            return numeric_cols
        elif col_type == "categorical":
            return categorical_cols
        else:
            return all_headers

    def _make_choices(headers: List[str], allow_none: bool) -> List[str]:
        choices = headers.copy()
        if allow_none:
            choices.append("[None]")
        choices.append("[Cancel]")
        return choices

    def _get_prompt(label: str, col_type: Literal["numeric", "categorical", "any"]) -> str:
        prompt = f"Select {label} header"
        if col_type == "numeric":
            prompt += " (numeric only)"
        elif col_type == "categorical":
            prompt += " (categorical only)"
        return prompt

    # select X
    x_headers = _get_filtered_headers(x_type)
    x_header = questionary.select(
        _get_prompt(x_label, x_type),
        choices=_make_choices(x_headers, allow_none_x),
    ).ask()
    if x_header is None or x_header == "[Cancel]":
        return None, None
    if allow_none_x and x_header == "[None]":
        x_header = None

    # select Y
    y_headers = _get_filtered_headers(y_type)
    y_header = questionary.select(
        _get_prompt(y_label, y_type),
        choices=_make_choices(y_headers, allow_none_y),
    ).ask()
    if y_header is None or y_header == "[Cancel]":
        return None, None
    if allow_none_y and y_header == "[None]":
        y_header = None

    return x_header, y_header


def select_legend(
    df: pd.DataFrame,
    col_type: Literal["numeric", "categorical", "any"] = "categorical",
    allow_none: bool = True,
) -> Optional[str]:
    """Ask the user for an optional legend (hue) column.

    ``col_type`` controls column filtering:
    - "categorical": only object/category dtype columns (default)
    - "numeric": only numeric columns
    - "any": all columns

    ``allow_none`` adds a ``[None]`` option and returns ``None`` when chosen.
    """

    all_headers: List[str] = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        col for col in all_headers
        if df[col].dtype == "object" or df[col].dtype.name in ("category", "string")
    ]

    def _get_filtered_headers(c_type: Literal["numeric", "categorical", "any"]) -> List[str]:
        if c_type == "numeric":
            return numeric_cols
        elif c_type == "categorical":
            return categorical_cols
        else:
            return all_headers

    headers = _get_filtered_headers(col_type)
    prompt = "Select legend header"
    if col_type == "numeric":
        prompt += " (numeric only)"
    elif col_type == "categorical":
        prompt += " (categorical only)"
    
    if allow_none:
        headers = headers.copy()
        headers.append("[None]")
    
    choice = questionary.select(prompt, choices=headers + ["[Cancel]"]).ask()
    if choice is None or choice == "[Cancel]" or choice == "[None]":
        return None
    return choice


def select_size(
    df: pd.DataFrame,
    col_type: Literal["numeric", "categorical", "any"] = "numeric",
    allow_none: bool = True,
) -> Optional[str]:
    """Ask the user for an optional size column.

    ``col_type`` controls column filtering:
    - "numeric": only numeric columns (default)
    - "categorical": only object/category dtype columns
    - "any": all columns

    ``allow_none`` adds a ``[None]`` choice.
    """

    all_headers: List[str] = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        col for col in all_headers
        if df[col].dtype == "object" or df[col].dtype.name in ("category", "string")
    ]

    def _get_filtered_headers(c_type: Literal["numeric", "categorical", "any"]) -> List[str]:
        if c_type == "numeric":
            return numeric_cols
        elif c_type == "categorical":
            return categorical_cols
        else:
            return all_headers

    headers = _get_filtered_headers(col_type)
    
    # Display available size columns
    print(f"\n{Fore.CYAN}Available size columns ({len(headers)}):")
    for col in headers:
        print(f"  • {col}")
    print()
    
    prompt = "Select size header"
    if col_type == "numeric":
        prompt += " (numeric only)"
    elif col_type == "categorical":
        prompt += " (categorical only)"
    
    if allow_none:
        headers = headers.copy()
        headers.append("[None]")
    
    choice = questionary.select(prompt, choices=headers + ["[Cancel]"]).ask()
    if choice is None or choice == "[Cancel]" or choice == "[None]":
        return None
    return choice


def get_xtick_rotation(x_values, max_labels: int = 10, max_label_len: int = 8) -> int:
    """Return recommended rotation for x tick labels.

    A rotation of ``45`` degrees is suggested when there are more than
    ``max_labels`` unique values or when any label exceeds ``max_label_len``
    characters.  Otherwise ``0`` is returned.
    """

    unique = pd.Series(x_values).unique()
    if len(unique) > max_labels or any(len(str(l)) > max_label_len for l in unique):
        return 45
    return 0


def select_chart_types(
    available_charts: List[str],
    prompt: str = "Select chart types to display (use Space to select, Enter to confirm)"
) -> Optional[List[str]]:
    """Ask the user to multi-select from available chart types.

    Parameters:
    * ``available_charts``: List of chart type names to choose from
    * ``prompt``: The prompt text to display to the user

    Returns a list of selected chart types, or None if cancelled.
    """
    
    choices = available_charts.copy()
    choices.append("[Cancel]")
    
    selected = questionary.checkbox(
        prompt,
        choices=choices
    ).ask()
    
    if selected is None or "[Cancel]" in selected:
        return None
    
    # Remove [Cancel] if it somehow got selected
    selected = [item for item in selected if item != "[Cancel]"]
    
    if not selected:
        return None
    
    return selected


def filter_column(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows in a DataFrame by a specific column's values.
    
    Supports different filtering methods based on column dtype:
    - Numeric: comparison operators (>, <=)
    - Datetime: date range syntax (yyyy-mm-dd)
    - Categorical/String/Object: list format or checkbox selection
    
    Returns the filtered DataFrame, or the original if cancelled.
    """
    
    all_headers: List[str] = df.columns.tolist()
    
    # Select which column to filter
    column = questionary.select(
        "Select column to filter:",
        choices=all_headers + ["[Cancel]"]
    ).ask()
    
    if column is None or column == "[Cancel]":
        return df
    
    col_dtype = df[column].dtype
    
    # Determine column type
    is_numeric = pd.api.types.is_numeric_dtype(col_dtype)
    is_datetime = pd.api.types.is_datetime64_any_dtype(col_dtype)
    is_categorical = (col_dtype == "object" or 
                      col_dtype.name in ("category", "string"))
    
    try:
        if is_numeric:
            return _filter_numeric(df, column)
        elif is_datetime:
            return _filter_datetime(df, column)
        elif is_categorical:
            return _filter_categorical(df, column)
        else:
            print(f"{Fore.LIGHTYELLOW_EX}Unsupported column type: {col_dtype}")
            return df
    except Exception as e:
        print(f"{Fore.LIGHTYELLOW_EX}Error filtering column: {e}")
        return df


def _filter_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Filter numeric column using > or <= operators."""
    
    print(f"\n{Fore.CYAN}Filter {column} (numeric)")
    print("Examples: '>100', '<=50.5', '<0'")
    
    filter_expr = questionary.text(
        "Enter filter expression (operator followed by value):",
        validate=lambda text: _validate_numeric_filter(text) or "Invalid format"
    ).ask()
    
    if filter_expr is None or filter_expr.strip() == "":
        return df
    
    filter_expr = filter_expr.strip()
    
    # Parse operator and value
    if filter_expr.startswith("<="):
        operator = "<="
        value = float(filter_expr[2:].strip())
        mask = df[column] <= value
    elif filter_expr.startswith("<"):
        operator = "<"
        value = float(filter_expr[1:].strip())
        mask = df[column] < value
    elif filter_expr.startswith(">="):
        operator = ">="
        value = float(filter_expr[2:].strip())
        mask = df[column] >= value
    elif filter_expr.startswith(">"):
        operator = ">"
        value = float(filter_expr[1:].strip())
        mask = df[column] > value
    elif filter_expr.startswith("=="):
        operator = "=="
        value = float(filter_expr[2:].strip())
        mask = df[column] == value
    else:
        raise ValueError("Invalid operator. Use >, >=, <, <=, or ==")
    
    result = df[mask]
    print(f"{Fore.LIGHTGREEN_EX}Filtered: {len(result)} rows remaining (removed {len(df) - len(result)} rows)")
    return result


def _filter_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Filter datetime column using date range syntax."""
    
    print(f"\n{Fore.CYAN}Filter {column} (datetime)")
    print("Examples: '>2024-01-15', '<=2024-12-31', '==2024-06-01'")
    
    filter_expr = questionary.text(
        "Enter filter expression (operator followed by date in yyyy-mm-dd format):",
    ).ask()
    
    if filter_expr is None or filter_expr.strip() == "":
        return df
    
    filter_expr = filter_expr.strip()
    
    try:
        # Parse operator and date
        if filter_expr.startswith("<="):
            operator = "<="
            date_str = filter_expr[2:].strip()
            date_val = pd.to_datetime(date_str)
            mask = df[column] <= date_val
        elif filter_expr.startswith("<"):
            operator = "<"
            date_str = filter_expr[1:].strip()
            date_val = pd.to_datetime(date_str)
            mask = df[column] < date_val
        elif filter_expr.startswith(">="):
            operator = ">="
            date_str = filter_expr[2:].strip()
            date_val = pd.to_datetime(date_str)
            mask = df[column] >= date_val
        elif filter_expr.startswith(">"):
            operator = ">"
            date_str = filter_expr[1:].strip()
            date_val = pd.to_datetime(date_str)
            mask = df[column] > date_val
        elif filter_expr.startswith("=="):
            operator = "=="
            date_str = filter_expr[2:].strip()
            date_val = pd.to_datetime(date_str)
            mask = df[column] == date_val
        else:
            raise ValueError("Invalid operator. Use >, >=, <, <=, or ==")
        
        result = df[mask]
        print(f"{Fore.LIGHTGREEN_EX}Filtered: {len(result)} rows remaining (removed {len(df) - len(result)} rows)")
        return result
    except Exception as e:
        print(f"{Fore.LIGHTYELLOW_EX}Error parsing date: {e}")
        return df


def _filter_categorical(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Filter categorical/string/object column."""
    
    unique_vals = df[column].unique()
    num_unique = len(unique_vals)
    
    print(f"\n{Fore.CYAN}Filter {column} (categorical, {num_unique} unique values)")
    
    # If 33 or fewer unique values, offer checkbox selection
    if num_unique <= 32:
        print("Select values to KEEP:")
        selected = questionary.checkbox(
            "Choose values (use Space to select, Enter to confirm):",
            choices=[str(val) for val in sorted(unique_vals)] + ["[Cancel]"]
        ).ask()
        
        if selected is None or "[Cancel]" in selected:
            return df
        
        selected = [val for val in selected if val != "[Cancel]"]
        
        if not selected:
            return df
        
        # Convert back to original dtype for comparison
        mask = df[column].astype(str).isin(selected)
    else:
        # Use list format for many unique values
        print(f"Too many unique values ({num_unique}) for checkbox. Use list format.")
        print("Example: 'A, B, C' or 'value1, value2'")
        
        list_input = questionary.text(
            "Enter values to KEEP (comma-separated):"
        ).ask()
        
        if list_input is None or list_input.strip() == "":
            return df
        
        # Parse comma-separated list
        values_to_keep = [v.strip() for v in list_input.split(",")]
        mask = df[column].astype(str).isin(values_to_keep)
    
    result = df[mask]
    print(f"{Fore.LIGHTGREEN_EX}Filtered: {len(result)} rows remaining (removed {len(df) - len(result)} rows)")
    return result


def _validate_numeric_filter(text: str) -> bool:
    """Validate numeric filter expression format."""
    if not text or not text.strip():
        return False
    
    text = text.strip()
    
    # Check for valid operators
    if text.startswith("<=") or text.startswith(">=") or text.startswith("=="):
        value_str = text[2:].strip()
    elif text.startswith("<") or text.startswith(">"):
        value_str = text[1:].strip()
    else:
        return False
    
    # Try to convert to float
    try:
        float(value_str)
        return True
    except ValueError:
        return False
