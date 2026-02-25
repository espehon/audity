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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
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


def _format_large_number(value: float) -> str:
    """Format large numbers with K, M, B notation (e.g., 1.3K, 3.4M).
    
    Args:
        value: The numeric value to format
        
    Returns:
        Formatted string representation of the value
    """
    abs_value = abs(value)
    
    if abs_value >= 1e9:
        return f"{value / 1e9:.1f}B"
    elif abs_value >= 1e6:
        return f"{value / 1e6:.1f}M"
    elif abs_value >= 1e3:
        return f"{value / 1e3:.1f}K"
    else:
        # For smaller values, show with one decimal place or as integer if it's whole
        if value == int(value):
            return str(int(value))
        return f"{value:.1f}"


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

# ---------------------------------------------------------------------------
# Delta Analysis Charts
# ---------------------------------------------------------------------------

def delta_chart_single_column(df: pd.DataFrame) -> None:
    """Create a multi-panel delta analysis chart.
    
    Displays delta analysis for a single column with aligned x-axes:
    1. Diverging bar chart (category vs value with color gradient)
    2. Horizontal violin plots (category vs value distribution)
    3. Histogram with KDE (distribution of values)
    
    User can select which charts to display to get more vertical space.
    All panels share the same x-axis range for alignment, sorted from
    best to worst performance (for charts 1 and 2).
    """
    try:
        # Get categorical columns for selection
        categorical_cols = [
            col for col in df.columns
            if df[col].dtype == "object" or df[col].dtype.name == "category"
        ]
        if not categorical_cols:
            print("No categorical columns found for category axis.")
            return
        
        # Select category column
        category_col = questionary.select(
            "Select category column (y-axis):",
            choices=categorical_cols + ["[Cancel]"]
        ).ask()
        if category_col is None or category_col == "[Cancel]":
            print("Delta chart creation cancelled.")
            return
        
        # Get numeric columns for selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("No numeric columns found for value axis.")
            return
        
        # Select value column
        value_col = questionary.select(
            "Select value column (x-axis, delta %):",
            choices=numeric_cols + ["[Cancel]"]
        ).ask()
        if value_col is None or value_col == "[Cancel]":
            print("Delta chart creation cancelled.")
            return
        
        # Select which charts to display
        available_charts = [
            "Diverging Bar Chart",
            "Violin Plots",
            "Histogram with KDE"
        ]
        
        selected_charts = questionary.checkbox(
            "Select charts to display (use Space to select, Enter to confirm):",
            choices=available_charts
        ).ask()
        
        if selected_charts is None or len(selected_charts) == 0:
            print("Delta chart creation cancelled.")
            return
        
        # Calculate sorted order: best to worst (highest to lowest average value)
        avg_by_category = df.groupby(category_col)[value_col].mean().sort_values(ascending=False)
        category_order = avg_by_category.index.tolist()
        
        # If more than 10 categories, ask user if they want to filter
        if len(category_order) > 10:
            filter_response = questionary.text(
                f"{len(category_order)} categories detected. Show top/bottom N (0 = all):",
                default="5"
            ).ask()
            
            if filter_response is None:
                print("Delta chart creation cancelled.")
                return
            
            try:
                n = int(filter_response)
                if n > 0:
                    # Keep top N (highest) and bottom N (lowest)
                    top_n = category_order[:n]
                    bottom_n = category_order[-n:]
                    category_order = top_n + bottom_n
                    # Re-filter the dataframe to only include these categories
                    df = df[df[category_col].isin(category_order)]
            except ValueError:
                print("Invalid input. Showing all categories.")
        
        spinner = _make_spinner("Creating delta analysis chart...")
        try:
            spinner.start()
            
            # Recalculate sorted order with potentially filtered data
            avg_by_category = df.groupby(category_col)[value_col].mean().sort_values(ascending=False)
            category_order = avg_by_category.index.tolist()
            
            # Determine x-axis limits for alignment across all panels
            x_min = df[value_col].min()
            x_max = df[value_col].max()
            x_margin = (x_max - x_min) * 0.05  # 5% margin
            x_limits = [x_min - x_margin, x_max + x_margin]
            
            # Create figure with appropriate number of subplots
            num_charts = len(selected_charts)
            base_height = 6
            fig, axes = plt.subplots(num_charts, 1, figsize=(12, base_height + len(category_order) * 0.3))
            
            # Increase spacing between subplots to prevent title overlap
            fig.subplots_adjust(hspace=0.5)
            
            # Handle single subplot case (axes is not a list)
            if num_charts == 1:
                axes = [axes]
            
            fig.suptitle(f"Delta Analysis: {value_col} by {category_col}", fontsize=14, fontweight='bold')
            
            # Create custom colormap: dark blue -> light blue -> tan -> light orange -> dark orange
            # Avoids white in the middle for better contrast
            colors_list = ['#00008B', '#4169E1', '#D3BF77', '#FFB347', '#FF8C00']  # dark blue, steel blue, tan, light orange, dark orange
            custom_cmap = LinearSegmentedColormap.from_list('blue_tan_orange', colors_list, N=256)
            
            ax_idx = 0
            
            # ===== CHART 1: Diverging Bar Chart =====
            if "Diverging Bar Chart" in selected_charts:
                ax = axes[ax_idx]
                avg_values = df.groupby(category_col)[value_col].mean().reindex(category_order)
                
                # Map values to 0-1 range for colormap
                abs_max = max(abs(avg_values.min()), abs(avg_values.max()))
                normalized_colors = 0.5 + (avg_values.values / (2 * abs_max))  # Map to 0-1 where 0.5 is zero
                colors = custom_cmap(normalized_colors)
                
                bars = ax.barh(range(len(category_order)), avg_values.values, color=colors)
                ax.set_yticks(range(len(category_order)))
                ax.set_yticklabels(list(avg_values.index))
                ax.set_xlabel(value_col)
                ax.set_ylabel(category_col)
                ax.set_title("Diverging Bar Chart (Average)", fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax.set_xlim(x_limits)
                
                ax_idx += 1
            
            # ===== CHART 2: Horizontal Violin Plots =====
            if "Violin Plots" in selected_charts:
                ax = axes[ax_idx]
                sns.violinplot(data=df, y=category_col, x=value_col, order=category_order[::-1], ax=ax)
                ax.set_xlabel(value_col)
                ax.set_ylabel(category_col)
                ax.set_title("Violin Plots (Distribution)", fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
                ax.set_xlim(x_limits)
                
                ax_idx += 1
            
            # ===== CHART 3: Histogram with KDE =====
            if "Histogram with KDE" in selected_charts:
                ax = axes[ax_idx]
                bins = 30
                sns.histplot(data=df, x=value_col, bins=bins, stat='count', kde=True, 
                            color='steelblue', edgecolor='black', ax=ax, line_kws={'linewidth': 2, 'color': 'red'})
                ax.set_xlabel(value_col)
                ax.set_ylabel("Count")
                ax.set_title(f"Histogram with KDE (Distribution, n={bins} bins)", fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
                ax.set_xlim(x_limits)
                
                ax_idx += 1
            
            plt.tight_layout()
            spinner.succeed("Delta analysis chart created successfully. Close window to continue.")
            _safe_show()
        except Exception as e:
            spinner.fail(f"{e}")
    except KeyboardInterrupt:
        print("\nDelta chart creation cancelled.")
        return


def delta_chart_winners(df: pd.DataFrame) -> None:
    """Create a winners analysis chart for delta columns.
    
    Displays a three-panel analysis of winners based on selected criteria:
    1. Bar chart (left): Sum of winning delta values per category, sorted best to worst
    2. Box plot (top right): Distribution of winning values per delta column
    3. Bar chart (bottom right): Total winner count per delta column with data labels
    
    User defines what constitutes a "winner" (higher or lower values).
    """
    try:
        # Get categorical columns for selection
        categorical_cols = [
            col for col in df.columns
            if df[col].dtype == "object" or df[col].dtype.name == "category"
        ]
        if not categorical_cols:
            print("No categorical columns found for category axis.")
            return
        
        # Select category column
        category_col = questionary.select(
            "Select category column:",
            choices=categorical_cols + ["[Cancel]"]
        ).ask()
        if category_col is None or category_col == "[Cancel]":
            print("Winners chart creation cancelled.")
            return
        
        # Get numeric columns for selection (delta columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("No numeric columns found for delta values.")
            return
        
        # Select multiple delta columns
        delta_cols = questionary.checkbox(
            "Select delta columns (use Space to select, Enter to confirm):",
            choices=numeric_cols + ["[Cancel]"]
        ).ask()
        
        if delta_cols is None or "[Cancel]" in delta_cols:
            print("Winners chart creation cancelled.")
            return
        
        delta_cols = [col for col in delta_cols if col != "[Cancel]"]
        
        if len(delta_cols) == 0:
            print("No valid delta columns selected. Chart creation cancelled.")
            return
        
        # Define winner criteria
        criteria = questionary.select(
            "Define a winner:",
            choices=["Higher is better", "Lower is better"]
        ).ask()
        
        if criteria is None:
            print("Winners chart creation cancelled.")
            return
        
        higher_is_better = criteria == "Higher is better"
        
        spinner = _make_spinner("Creating winners analysis chart...")
        try:
            spinner.start()
            
            # Determine winner for each row (category instance)
            # Winner is the delta column with highest/lowest value
            if higher_is_better:
                df['_winner_col'] = df[delta_cols].idxmax(axis=1)
                df['_winner_value'] = df[delta_cols].max(axis=1)
            else:
                df['_winner_col'] = df[delta_cols].idxmin(axis=1)
                df['_winner_value'] = df[delta_cols].min(axis=1)
            
            # Calculate total winner value per category
            winners_by_category = df.groupby(category_col)['_winner_value'].sum().sort_values(ascending=False)
            
            # Filter to only actual winners (positive if higher is better, negative if lower is better)
            if higher_is_better:
                winners_by_category = winners_by_category[winners_by_category > 0]
            else:
                winners_by_category = winners_by_category[winners_by_category < 0]
            
            # Check if any winners remain
            if len(winners_by_category) == 0:
                # Clean up temporary columns
                df.drop(columns=['_winner_col', '_winner_value'], inplace=True)
                spinner.fail("No winners found. Aborting.")
                return
            
            category_order = winners_by_category.index.tolist()
            
            # Get winner delta column for each category (most frequent winner)
            def get_primary_winner(group):
                return group['_winner_col'].mode()[0] if len(group['_winner_col'].mode()) > 0 else group['_winner_col'].iloc[0]
            
            category_winners = df.groupby(category_col).apply(get_primary_winner, include_groups=False)
            
            # ===== Filter dataset: only keep winners per category =====
            # For each category, keep only rows where the winning column won
            winners_only_rows = []
            for cat in category_order:
                winning_col = category_winners[cat]
                cat_data = df[df[category_col] == cat]
                # Keep only rows where this category's winner column was the actual winner
                winners_only_rows.append(cat_data[cat_data['_winner_col'] == winning_col])
            
            if winners_only_rows:
                df_winners_only = pd.concat(winners_only_rows, ignore_index=True)
            else:
                df_winners_only = pd.DataFrame()
            
            # Create color map: each delta column gets a color
            delta_col_colors = {}
            colors_palette = sns.color_palette("husl", len(delta_cols))
            for idx, col in enumerate(delta_cols):
                delta_col_colors[col] = colors_palette[idx]
            
            # Create figure with GridSpec for custom layout
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.2, 0.6], height_ratios=[1.2, 1])
            
            ax1 = fig.add_subplot(gs[:, 0])  # Left: full height
            ax2 = fig.add_subplot(gs[0, 1])  # Top middle
            ax3 = fig.add_subplot(gs[1, 1])  # Bottom middle
            ax4 = fig.add_subplot(gs[:, 2])  # Right: full height
            
            fig.suptitle(
                f"Winners Analysis: {', '.join(delta_cols)} by {category_col}",
                fontsize=14, fontweight='bold'
            )
            
            # ===== CHART 1: Bar chart of summed winner values per category =====
            ax = ax1
            winner_values = winners_by_category.reindex(category_order)
            winner_colors = [delta_col_colors[category_winners[cat]] for cat in category_order]
            
            bars = ax.barh(range(len(category_order)), winner_values.values, color=winner_colors)
            ax.set_yticks(range(len(category_order)))
            ax.set_yticklabels(category_order)
            ax.set_xlabel("Total Winner Value")
            ax.set_ylabel(category_col)
            ax.set_title(f"Winners by {category_col} ({'Higher' if higher_is_better else 'Lower'} is better, Label = Count)", fontweight='bold')
            ax.invert_yaxis()  # Best on top
            
            # Flip x-axis for "lower is better" so bars extend from y-axis
            if not higher_is_better:
                ax.invert_xaxis()
            
            # Add data labels showing count of winning data points per category
            winner_counts = df_winners_only[category_col].value_counts().reindex(category_order, fill_value=0)
            for bar, count in zip(bars, winner_counts.to_numpy()):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2.,
                       f' {int(count)}',
                       ha='left', va='center', fontweight='bold')
            
            # Add legend for delta columns
            legend_handles = [Patch(facecolor=delta_col_colors[col]) for col in delta_cols]
            ax.legend(legend_handles, delta_cols, loc='best', fontsize=9, title='Winner', title_fontsize=9)
            
            # ===== CHART 2: Box plot of winning values per delta column =====
            ax = ax2
            
            # Prepare data: only winning rows, grouped by delta column
            winner_data_list = []
            for col in delta_cols:
                winner_values_for_col = df_winners_only[df_winners_only['_winner_col'] == col][col]
                for val in winner_values_for_col:
                    winner_data_list.append({'Delta Column': col, 'Value': val})
            
            if winner_data_list:
                winner_data_df = pd.DataFrame(winner_data_list)
                sns.boxplot(
                    data=winner_data_df,
                    x='Delta Column',
                    y='Value',
                    hue='Delta Column',
                    ax=ax,
                    palette=[delta_col_colors[col] for col in delta_cols],
                    legend=False
                )
            
            ax.set_title("Box Plot (Winners)", fontweight='bold')
            ax.set_ylabel("Value")
            ax.set_xlabel("Delta Column")
            
            # ===== CHART 3: Total winner sum bar chart =====
            ax = ax3
            
            # Count total winners per delta column (from winners only)
            total_winners = df_winners_only['_winner_col'].value_counts().reindex(delta_cols, fill_value=0)
            # Sum total winner values per delta column (from winners only)
            total_winner_values = df_winners_only.groupby('_winner_col')['_winner_value'].sum().reindex(delta_cols, fill_value=0)
            
            bars = ax.bar(range(len(delta_cols)), total_winners.to_numpy(), 
                         color=[delta_col_colors[col] for col in delta_cols])
            ax.set_xticks(range(len(delta_cols)))
            ax.set_xticklabels(delta_cols, rotation=45, ha='right')
            ax.set_ylabel("Winner Count")
            ax.set_title("Winners Per Delta Column (Label = Total Value)", fontweight='bold')
            
            # Add data labels on bars showing total value
            for bar, count, value in zip(bars, total_winners.values, total_winner_values.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       _format_large_number(value),
                       ha='center', va='bottom', fontweight='bold')
            
            # ===== CHART 4: Stacked vertical bar chart of total values per delta column =====
            ax = ax4
            
            # Sum total values per delta column from winners only
            total_by_delta = df_winners_only.groupby('_winner_col')['_winner_value'].sum().reindex(delta_cols, fill_value=0)
            
            # Create stacked bar chart
            bottom = 0
            bar_positions = [0]
            bars_list = []
            cumulative_heights = []
            
            for col in delta_cols:
                bar = ax.bar(bar_positions, [total_by_delta[col]], bottom=bottom,
                           color=delta_col_colors[col], width=0.5)
                bars_list.append(bar)
                cumulative_heights.append(bottom + total_by_delta[col] / 2)  # For label positioning
                bottom += total_by_delta[col]
            
            # Add data labels for each segment with column name and value
            for col, bar, y_pos in zip(delta_cols, bars_list, cumulative_heights):
                value = total_by_delta[col]
                label_text = f'{col}: {_format_large_number(value)}'
                ax.text(0, y_pos, label_text,
                       ha='center', va='center', fontweight='bold', fontsize=7, color='white')
            
            # Add grand total label at top
            grand_total = total_by_delta.sum()
            ax.text(0, bottom + (grand_total * 0.02),  # Small offset above the bar
                   f'Total: {_format_large_number(grand_total)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(['Winners'])
            ax.set_ylabel("Total Value")
            ax.set_title("Total Winners Value", fontweight='bold')
            
            # Clean up temporary columns
            df.drop(columns=['_winner_col', '_winner_value'], inplace=True)
            
            plt.tight_layout()
            spinner.succeed("Winners analysis chart created successfully. Close window to continue.")
            _safe_show()
        except Exception as e:
            # Clean up temporary columns in case of error
            if '_winner_col' in df.columns:
                df.drop(columns=['_winner_col'], inplace=True)
            if '_winner_value' in df.columns:
                df.drop(columns=['_winner_value'], inplace=True)
            spinner.fail(f"{e}")
    except KeyboardInterrupt:
        print("\nWinners chart creation cancelled.")
        # Clean up temporary columns in case of interruption
        if '_winner_col' in df.columns:
            df.drop(columns=['_winner_col'], inplace=True)
        if '_winner_value' in df.columns:
            df.drop(columns=['_winner_value'], inplace=True)
        return


def delta_chart_multiple_columns(df: pd.DataFrame) -> None:
    """Create a multi-panel delta analysis chart for multiple columns.
    
    Displays delta analysis for multiple columns with three horizontal panels:
    1. Heatmap (categories vs delta columns, color gradient blue to red)
    2. Stacked box plot (categories vs values, colored by delta columns)
    3. KDE histogram overlaid (distribution of values with hue per delta column)
    
    All panels share the same category order, sorted from best to worst
    performance (highest to lowest average value across all delta columns).
    """
    try:
        # Get categorical columns for selection
        categorical_cols = [
            col for col in df.columns
            if df[col].dtype == "object" or df[col].dtype.name == "category"
        ]
        if not categorical_cols:
            print("No categorical columns found for category axis.")
            return
        
        # Select category column
        category_col = questionary.select(
            "Select category column (y-axis):",
            choices=categorical_cols + ["[Cancel]"]
        ).ask()
        if category_col is None or category_col == "[Cancel]":
            print("Delta chart creation cancelled.")
            return
        
        # Get numeric columns for selection (delta columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("No numeric columns found for delta values.")
            return
        
        # Select multiple delta columns
        delta_cols = questionary.checkbox(
            "Select delta columns (use Space to select, Enter to confirm):",
            choices=numeric_cols + ["[Cancel]"]
        ).ask()
        
        if delta_cols is None or "[Cancel]" in delta_cols:
            print("Delta chart creation cancelled.")
            return
        
        if len(delta_cols) == 0:
            print("No delta columns selected. Chart creation cancelled.")
            return
        
        # Remove [Cancel] if it somehow got through
        delta_cols = [col for col in delta_cols if col != "[Cancel]"]
        
        if len(delta_cols) == 0:
            print("No valid delta columns selected. Chart creation cancelled.")
            return
        
        # Calculate sorted order: best to worst (highest to lowest average value)
        # Use mean across all selected delta columns
        avg_across_deltas = df[delta_cols].mean(axis=1)
        avg_by_category = df.groupby(category_col).apply(
            lambda x: x[delta_cols].values.flatten().mean(),
            include_groups=False
        ).sort_values(ascending=False)
        category_order = avg_by_category.index.tolist()
        
        # If more than 10 categories, ask user if they want to filter
        if len(category_order) > 10:
            filter_response = questionary.text(
                f"{len(category_order)} categories detected. Show top/bottom N (0 = all):",
                default="5"
            ).ask()
            
            if filter_response is None:
                print("Delta chart creation cancelled.")
                return
            
            try:
                n = int(filter_response)
                if n > 0:
                    # Keep top N (highest) and bottom N (lowest)
                    top_n = category_order[:n]
                    bottom_n = category_order[-n:]
                    category_order = top_n + bottom_n
                    # Re-filter the dataframe to only include these categories
                    df = df[df[category_col].isin(category_order)]
            except ValueError:
                print("Invalid input. Showing all categories.")
        
        
        spinner = _make_spinner("Creating multi-column delta analysis chart...")
        try:
            spinner.start()
            
            # Recalculate sorted order with potentially filtered data
            avg_by_category = df.groupby(category_col).apply(
                lambda x: x[delta_cols].values.flatten().mean(),
                include_groups=False
            ).sort_values(ascending=False)
            category_order = avg_by_category.index.tolist()
            
            # Determine value range for alignment
            all_values = df[delta_cols].values.flatten()
            x_min = all_values.min()
            x_max = all_values.max()
            x_margin = (x_max - x_min) * 0.05  # 5% margin
            x_limits = [x_min - x_margin, x_max + x_margin]
            
            # Auto-calculate bins using Freedman-Diaconis rule
            n_samples = len(all_values)
            bins = max(int(np.sqrt(n_samples)), 10)  # At least 10 bins
            
            # Create figure with 3 horizontal subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 5 + len(category_order) * 0.2))
            
            fig.suptitle(
                f"Delta Analysis: {', '.join(delta_cols)} by {category_col}",
                fontsize=14, fontweight='bold'
            )
            
            # Create custom colormap: blue to red (temperature gradient)
            colors_list = ['#0000FF', '#4169E1', '#87CEEB', '#FFB6C1', '#FF6347', '#FF0000']
            custom_cmap = LinearSegmentedColormap.from_list('blue_red_temp', colors_list, N=256)
            
            # ===== CHART 1: Heatmap =====
            ax = axes[0]
            
            # Prepare data for heatmap: categories (rows) x delta columns (columns)
            heatmap_data = df.groupby(category_col)[delta_cols].mean().reindex(category_order)
            
            # Normalize data for colormap (0-1 range)
            h_min = heatmap_data.values.min()
            h_max = heatmap_data.values.max()
            h_range = h_max - h_min if h_max != h_min else 1
            heatmap_normalized = (heatmap_data - h_min) / h_range
            
            im = ax.imshow(heatmap_normalized, cmap=custom_cmap, aspect='auto', interpolation='nearest')
            
            # Set ticks and labels
            ax.set_xticks(range(len(delta_cols)))
            ax.set_xticklabels(delta_cols, rotation=45, ha='right')
            ax.set_yticks(range(len(category_order)))
            ax.set_yticklabels(category_order)
            
            ax.set_title("Heatmap (Average by Category)", fontweight='bold')
            ax.set_xlabel("Delta Columns")
            ax.set_ylabel(category_col)
            
            # Add values to heatmap cells
            for i in range(len(category_order)):
                for j in range(len(delta_cols)):
                    value = heatmap_data.iloc[i, j]
                    # Use white text if color is dark, black if light
                    text_color = 'white' if heatmap_normalized.iloc[i, j] < 0.5 else 'black'
                    ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                           color=text_color, fontsize=8, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Value", rotation=270, labelpad=15)
            
            # ===== CHART 2: Stacked Box Plot =====
            ax = axes[1]
            
            # Prepare data for box plot: melt the delta columns
            df_melted = df.melt(
                id_vars=[category_col],
                value_vars=delta_cols,
                var_name='Delta Column',
                value_name='Value'
            )
            
            # Create box plot with categories in same order as heatmap
            sns.boxplot(
                data=df_melted,
                y=category_col,
                x='Value',
                hue='Delta Column',
                order=category_order,
                ax=ax
            )
            
            ax.set_title("Box Plot (Distribution by Delta Column)", fontweight='bold')
            ax.set_xlabel("Value")
            ax.set_ylabel(category_col)
            ax.set_xlim(x_limits)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            # Move legend inside the chart
            ax.legend(title='Delta Column', loc='best', fontsize=8, title_fontsize=8)
            
            # ===== CHART 3: KDE Histogram =====
            ax = axes[2]
            
            # Create separate histograms with KDE for each delta column
            colors_palette = sns.color_palette("husl", len(delta_cols))
            
            for idx, col in enumerate(delta_cols):
                sns.histplot(
                    data=df,
                    x=col,
                    bins=bins,
                    stat='count',
                    kde=True,
                    label=col,
                    ax=ax,
                    color=colors_palette[idx],
                    alpha=0.5,
                    edgecolor='black',
                    line_kws={'linewidth': 2}
                )
            
            ax.set_title(f"KDE Histogram (Count, n={bins} bins)", fontweight='bold')
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.set_xlim(x_limits)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.legend(title='Delta Column', fontsize=8, title_fontsize=8)
            
            plt.tight_layout()
            spinner.succeed("Delta analysis chart created successfully. Close window to continue.")
            _safe_show()
        except Exception as e:
            spinner.fail(f"{e}")
    except KeyboardInterrupt:
        print("\nDelta chart creation cancelled.")
        return

