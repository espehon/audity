# Audity
Audit, inspect, and survey data by generating visuals with matplotlib on the fly. Audity is a CLI tool that loads a data file (csv/xlsx) a offers charting options allowing you to chose x and y axis. Visuals are rendered in a new window and all data manipulation is in memory. 

# Usage
Just run `audity` in the directory of your data file(s). File selection is handled within audity. Simply use the arrow keys and enter on your desired file. Audity will then load the file as a Pandas Dataframe and then the fun begins.

# Features
- Directory navigation for file selection
- Preview or describe Dataframes
- Edit column names
- Edit column data types
- Remove columns
- Remove outliers (IQR)
- Render charts (see [Supported Visuals](#Supported-Visuals) below)

# Supported Visuals
- Box Plot
- Violin Plot
- Line Plot
- Bar Plot
- Scatter Plot
- Joint Grid Plot
- Relation Plot
- Pair Plot

