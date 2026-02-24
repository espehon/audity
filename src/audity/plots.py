# MIT License
# Copyright (c) 2025 espehon

"""Plotting routines for the ``audity`` package.

All chart functions accept a :class:`pandas.DataFrame` and take care of
querying the user for required columns via the helpers exported from
:mod:`audity.ui`.  This keeps the logic focussed on the visualisation and
makes it easy to add new plot types without duplicating UI code.

"""

from __future__ import annotations


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import questionary

from halo import Halo

# ``Tuple`` and ``Optional`` were previously imported for annotations but
# are no longer needed; keeping the import list minimal prevents pylance
# "unused import" warnings.

# import selectors from the ui submodule
from .ui import (
    select_axis_headers,
    select_legend,
    select_size,
    get_xtick_rotation,
)


# ---------------------------------------------------------------------------
# utility helpers
# ---------------------------------------------------------------------------

def _make_spinner(text: str = "Processing...") -> Halo:
    return Halo(text=text, spinner="dots")


def _safe_show() -> None:
    """Show the current matplotlib figure with a grid.

    A tiny helper to keep the call site concise and ensure the return type is
    recognised by static checkers.
    """
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def line_plot(df: pd.DataFrame) -> None:
    try:
        x_header, y_header = select_axis_headers(
            df, x_type="numeric", y_type="numeric"
        )
        if x_header is None or y_header is None:
            print("Line plot creation cancelled.")
            return
        assert x_header is not None and y_header is not None

        legend = select_legend(df, col_type="categorical")
        if legend is None and df.select_dtypes(include=["object", "category"]).shape[1] > 0:
            # Legend selection was cancelled
            print("Line plot creation cancelled.")
            return

        spinner = _make_spinner("Creating line plot...")
        try:
            spinner.start()
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
            spinner.succeed("Line plot created successfully. Close window to continue.")
            _safe_show()
        except Exception as e:
            spinner.fail(f"{e}")
    except KeyboardInterrupt:
        print("\nLine plot creation cancelled.")
        return


def bar_plot(df: pd.DataFrame) -> None:
    try:
        x_header, y_header = select_axis_headers(
            df, x_type="any", y_type="numeric"
        )
        if x_header is None or y_header is None:
            print("Bar plot creation cancelled.")
            return
        assert x_header is not None and y_header is not None

        legend = select_legend(df, col_type="categorical")
        if legend is None and df.select_dtypes(include=["object", "category"]).shape[1] > 0:
            # Legend selection was cancelled
            print("Bar plot creation cancelled.")
            return

        spinner = _make_spinner("Creating bar plot...")
        try:
            spinner.start()
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
            spinner.succeed("Bar plot created successfully. Close window to continue.")
            _safe_show()
        except Exception as e:
            spinner.fail(f"{e}")
    except KeyboardInterrupt:
        print("\nBar plot creation cancelled.")
        return


def scatter_plot(df: pd.DataFrame) -> None:
    try:
        x_header, y_header = select_axis_headers(
            df, x_type="numeric", y_type="numeric"
        )
        if x_header is None or y_header is None:
            print("Scatter plot creation cancelled.")
            return
        assert x_header is not None and y_header is not None

        legend = select_legend(df, col_type="categorical")
        if legend is None and df.select_dtypes(include=["object", "category"]).shape[1] > 0:
            # Legend selection was cancelled
            print("Scatter plot creation cancelled.")
            return

        spinner = _make_spinner("Creating scatter plot...")
        try:
            spinner.start()
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
            spinner.succeed("Scatter plot created successfully. Close window to continue.")
            _safe_show()
        except Exception as e:
            spinner.fail(f"{e}")
    except KeyboardInterrupt:
        print("\nScatter plot creation cancelled.")
        return


def pair_plot(df: pd.DataFrame) -> None:
    try:
        numerical_threshold = 6
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) > numerical_threshold:
            numerical_col_subset = questionary.checkbox(
                f"There are {len(numerical_cols)} numeric columns which may clutter the plot. "
                f"Anything over {numerical_threshold} is not recommended.\n"
                "Please verify which columns to render:",
                choices=numerical_cols,
            ).ask()
            if numerical_col_subset is None or len(numerical_col_subset) < 2:
                user = questionary.confirm(
                    "Not enough columns selected. Do you want to use all the columns instead?",
                    default=False,
                ).ask()
                if user is None or not user:
                    print("Pair plot creation cancelled.")
                    return
            else:
                cols_to_drop = list(set(numerical_cols) - set(numerical_col_subset))
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)

        legend = select_legend(df, col_type="categorical")
        spinner = _make_spinner("Creating pair plot...")
        try:
            spinner.start()
            if legend:
                sns.pairplot(df, hue=legend)
                plt.suptitle(f"Pair Plot with Legend: {legend}", y=1.02)
            else:
                sns.pairplot(df)
                plt.suptitle("Pair Plot", y=1.02)
            spinner.succeed("Pair plot created successfully. Close window to continue.")
            _safe_show()
        except Exception as e:
            spinner.fail(f"{e}")
    except KeyboardInterrupt:
        print("\nPair plot creation cancelled.")
        return


def box_plot(df: pd.DataFrame) -> None:
    try:
        x_header, y_header = select_axis_headers(
            df, x_label="X", y_label="Y", x_type="categorical", y_type="numeric", allow_none_x=True
        )
        if y_header is None:
            print("Box plot creation cancelled.")
            return
        # x_header can legitimately be None in the case of a single-axis plot

        legend = select_legend(df, col_type="categorical")
        if legend is None and df.select_dtypes(include=["object", "category"]).shape[1] > 0:
            # Legend selection was cancelled
            print("Box plot creation cancelled.")
            return

        spinner = _make_spinner("Creating box plot...")
        try:
            spinner.start()
            plt.figure(figsize=(10, 6))
            if legend:
                sns.boxplot(data=df, x=x_header, y=y_header, hue=legend)
            else:
                sns.boxplot(data=df, x=x_header, y=y_header)
            title = f"Box Plot: {y_header}"
            if x_header:
                title += f" by {x_header}"
            if legend:
                title += f" with Legend: {legend}"
            plt.title(title)
            if x_header is not None:
                plt.xlabel(x_header)
            plt.ylabel(y_header)
            if x_header is not None:
                plt.xticks(rotation=get_xtick_rotation(df[x_header]))
            spinner.succeed("Box plot created successfully. Close window to continue.")
            _safe_show()
        except Exception as e:
            spinner.fail(f"{e}")
    except KeyboardInterrupt:
        print("\nBox plot creation cancelled.")
        return


def violin_plot(df: pd.DataFrame) -> None:
    try:
        x_header, y_header = select_axis_headers(
            df, x_label="X", y_label="Y", x_type="categorical", y_type="numeric", allow_none_x=True
        )
        if y_header is None:
            print("Violin plot creation cancelled.")
            return
        # x_header may be None here and is handled later

        legend = select_legend(df, col_type="categorical")
        if legend is None and df.select_dtypes(include=["object", "category"]).shape[1] > 0:
            # Legend selection was cancelled
            print("Violin plot creation cancelled.")
            return

        spinner = _make_spinner("Creating violin plot...")
        try:
            spinner.start()
            plt.figure(figsize=(10, 6))
            if legend:
                sns.violinplot(data=df, x=x_header, y=y_header, hue=legend)
            else:
                sns.violinplot(data=df, x=x_header, y=y_header)
            title = f"Violin Plot: {y_header}"
            if x_header:
                title += f" by {x_header}"
            if legend:
                title += f" with Legend: {legend}"
            plt.title(title)
            if x_header is not None:
                plt.xlabel(x_header)
            plt.ylabel(y_header)
            if x_header is not None:
                plt.xticks(rotation=get_xtick_rotation(df[x_header]))
            spinner.succeed("Violin plot created successfully. Close window to continue.")
            _safe_show()
        except Exception as e:
            spinner.fail(f"{e}")
    except KeyboardInterrupt:
        print("\nViolin plot creation cancelled.")
        return


def distribution_plot(df: pd.DataFrame) -> None:
    try:
        # Use select_axis_headers with allow_none to get a single column
        # by asking for X only and setting Y to None
        x_header, _ = select_axis_headers(
            df,
            x_label="Distribution",
            y_label="",
            x_type="any",
            allow_none_y=True,
        )
        if x_header is None:
            print("Distribution plot creation cancelled.")
            return
        
        # Handle non-numeric columns by counting frequencies
        plot_df = df
        is_count_plot = False
        if not pd.api.types.is_numeric_dtype(df[x_header]):
            is_count_plot = True
            plot_df = df[x_header].value_counts().reset_index()
            plot_df.columns = [x_header, 'count']
            plot_df = plot_df.sort_values(x_header)
        
        legend = select_legend(df, col_type="categorical") if not is_count_plot else None
        if legend is None and not is_count_plot and df.select_dtypes(include=["object", "category"]).shape[1] > 0:
            # Legend selection was cancelled
            print("Distribution plot creation cancelled.")
            return

        spinner = _make_spinner("Creating distribution plot...")
        try:
            spinner.start()
            if is_count_plot:
                plt.figure(figsize=(10, 6))
                plt.bar(plot_df[x_header], plot_df['count'])
                plt.title(f"Distribution Plot: {x_header} (count)")
                plt.xlabel(x_header)
                plt.ylabel("Count")
                plt.xticks(rotation=get_xtick_rotation(plot_df[x_header]))
                spinner.succeed("Distribution plot created successfully. Close window to continue.")
                plt.show()
            else:
                if legend:
                    g = sns.displot(df, x=x_header, hue=legend, kde=True, height=6, aspect=1.6)
                    g.fig.suptitle(f"Distribution Plot: {x_header} by {legend}", y=0.98)
                else:
                    g = sns.displot(data=df, x=x_header, kde=True, height=6, aspect=1.6)
                    g.fig.suptitle(f"Distribution Plot: {x_header}", y=0.98)
                g.set_axis_labels(x_header, "Density")
                g.fig.tight_layout()
                spinner.succeed("Distribution plot created successfully. Close window to continue.")
                plt.show()
        except Exception as e:
            spinner.fail(f"{e}")
    except KeyboardInterrupt:
        print("\nDistribution plot creation cancelled.")
        return


def joint_grid_plot(df: pd.DataFrame) -> None:
    try:
        x_header, y_header = select_axis_headers(
            df,
            x_label="X",
            y_label="Y",
            x_type="numeric",
            y_type="numeric",
        )
        if x_header is None or y_header is None:
            print("Joint grid plot creation cancelled.")
            return
        assert x_header is not None and y_header is not None

        spinner = _make_spinner("Creating joint grid plot...")
        try:
            spinner.start()
            g = sns.jointplot(data=df, x=x_header, y=y_header, kind="hist", height=8)
            g.plot_joint(
                sns.histplot,
                cmap=sns.dark_palette("#69d", reverse=True, as_cmap=True),
                cbar=True,
            )
            g.plot_marginals(sns.histplot, element="step")
            g.figure.suptitle(f"Joint Grid Plot: {y_header} vs {x_header}", y=1.02)
            g.set_axis_labels(x_header, y_header)
            spinner.succeed("Joint grid plot created successfully. Close window to continue.")
            plt.show()
        except Exception as e:
            spinner.fail(f"{e}")
    except KeyboardInterrupt:
        print("\nJoint grid plot creation cancelled.")
        return


def relation_plot(df: pd.DataFrame) -> None:
    try:
        x_header, y_header = select_axis_headers(
            df,
            x_label="X",
            y_label="Y",
            x_type="numeric",
            y_type="numeric",
        )
        if x_header is None or y_header is None:
            print("Relation plot creation cancelled.")
            return
        assert x_header is not None and y_header is not None

        legend = select_legend(df, col_type="categorical")
        if legend is None and df.select_dtypes(include=["object", "category"]).shape[1] > 0:
            # Legend selection was cancelled
            print("Relation plot creation cancelled.")
            return

        size_header = select_size(df, col_type="numeric")
        if size_header is None and df.select_dtypes(include=[np.number]).shape[1] > 1:
            # Size selection was cancelled (and there are more numeric cols than just the axes)
            print("Relation plot creation cancelled.")
            return

        spinner = _make_spinner("Creating relation plot...")
        try:
            spinner.start()
            sns.relplot(
                data=df,
                x=x_header,
                y=y_header,
                hue=legend,
                size=size_header,
                kind="scatter",
                height=6,
                aspect=1.6,
            )
            title = f"Relation Plot: {y_header} vs {x_header}"
            if legend:
                title += f" by {legend}"
            plt.title(title)
            plt.xlabel(x_header)
            plt.ylabel(y_header)
            plt.grid(True)
            spinner.succeed("Relation plot created successfully. Close window to continue.")
            plt.show()
        except Exception as e:
            spinner.fail(f"{e}")
    except KeyboardInterrupt:
        print("\nRelation plot creation cancelled.")
        return


