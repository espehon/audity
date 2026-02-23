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

from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
from colorama import Fore
import questionary


def select_axis_headers(
    df: pd.DataFrame,
    x_label: str = "X",
    y_label: str = "Y",
    require_numeric_x: bool = False,
    require_numeric_y: bool = False,
    allow_none_x: bool = False,
    allow_none_y: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Ask the user to choose headers for X and Y axes.

    Parameters are mostly passed through to the underlying prompts:

    * ``require_numeric_*`` will ensure the chosen column is numeric, printing
      a warning and returning ``(None,None)`` if not.
    * ``allow_none_*`` adds a ``"[None]"`` option and returns ``None`` when
      selected (useful for plots that don't require one of the axes).

    On cancellation the pair ``(None, None)`` is returned.
    """

    headers: List[str] = df.columns.tolist()

    def _make_choices(allow_none: bool) -> List[str]:
        choices = headers.copy()
        if allow_none:
            choices.append("[None]")
        choices.append("[Cancel]")
        return choices

    # select X
    x_header = questionary.select(
        f"Select {x_label} header",
        choices=_make_choices(allow_none_x),
    ).ask()
    if x_header is None or x_header == "[Cancel]":
        return None, None
    if allow_none_x and x_header == "[None]":
        x_header = None
    if require_numeric_x and x_header is not None:
        numeric = df.select_dtypes(include=[np.number]).columns
        if x_header not in numeric:
            print(f"{Fore.LIGHTYELLOW_EX}{x_label} header must be numeric.")
            return None, None

    # select Y
    y_header = questionary.select(
        f"Select {y_label} header",
        choices=_make_choices(allow_none_y),
    ).ask()
    if y_header is None or y_header == "[Cancel]":
        return None, None
    if allow_none_y and y_header == "[None]":
        y_header = None
    if require_numeric_y and y_header is not None:
        numeric = df.select_dtypes(include=[np.number]).columns
        if y_header not in numeric:
            print(f"{Fore.LIGHTYELLOW_EX}{y_label} header must be numeric.")
            return None, None

    return x_header, y_header


def select_legend(
    df: pd.DataFrame,
    only_categorical: bool = True,
    allow_none: bool = True,
) -> Optional[str]:
    """Ask the user for an optional legend (hue) column.

    ``only_categorical`` restricts candidates to object/category dtypes; this is
    sensible for most plots but callers may set it to ``False`` if they wish to
    allow numeric colours.  ``allow_none`` adds a ``[None]`` option and returns
    ``None`` when chosen.
    """

    headers: List[str] = df.columns.tolist()
    if only_categorical:
        headers = [
            col for col in headers
            if df[col].dtype == "object" or df[col].dtype.name == "category"
        ]
    
    # Display available legend options
    print(f"\n{Fore.CYAN}Available legend columns ({len(headers)}):")
    for col in headers:
        print(f"  • {col}")
    print()
    
    if allow_none:
        headers.append("[None]")
    choice = questionary.select("Select legend header", choices=headers + ["[Cancel]"]).ask()
    if choice is None or choice == "[Cancel]" or choice == "[None]":
        return None
    return choice


def select_size(
    df: pd.DataFrame,
    numeric_only: bool = True,
    allow_none: bool = True,
) -> Optional[str]:
    """Ask the user for an optional size column (numeric only).

    ``numeric_only`` filters to numeric dtypes.  ``allow_none`` adds a
    ``[None]`` choice.
    """

    headers: List[str] = df.columns.tolist()
    if numeric_only:
        headers = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Display available size columns
    print(f"\n{Fore.CYAN}Available size columns ({len(headers)}):")
    for col in headers:
        print(f"  • {col}")
    print()
    
    if allow_none:
        headers.append("[None]")
    choice = questionary.select("Select size header", choices=headers + ["[Cancel]"]).ask()
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
