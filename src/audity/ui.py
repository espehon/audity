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
        if df[col].dtype == "object" or df[col].dtype.name == "category"
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
        if df[col].dtype == "object" or df[col].dtype.name == "category"
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
        if df[col].dtype == "object" or df[col].dtype.name == "category"
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
