# Inpatient Data Analysis

This folder contains scripts for analyzing the ACCM inpatient data from `dbo.accm_inpatient.csv`.

## Files

- `analyze_inpatient_data.py` - Main analysis script

## Features

The analysis script provides comprehensive data exploration including:

### 1. Data Loading & Basic Information
- Dataset dimensions (rows × columns)
- Memory usage
- Duplicate rows detection
- Column type identification (numerical, categorical, datetime)

### 2. Missing Data Analysis
- Identifies missing values per column
- Calculates missing data percentages
- Generates visualization of columns with missing data

### 3. Numerical Features Analysis
- Descriptive statistics (mean, median, std, min, max)
- Distribution plots for numerical features
- Correlation matrix heatmap

### 4. Categorical Features Analysis
- Unique value counts for each categorical column
- Top categories analysis
- Visualizations of most frequent categories

### 5. Datetime Features Analysis
- Date range identification
- Valid date counts
- Temporal coverage analysis

### 6. Patient Encounter Analysis
- Encounters per patient statistics
- Admission type distribution
- Department/specialty distribution
- Length of stay analysis (if admission/discharge times available)

### 7. Summary Report
- Overall dataset statistics
- Exports summary to CSV file

## Requirements

```bash
pip install pandas numpy matplotlib seaborn
```

## Usage

### Run the analysis:

```bash
cd /projects/LCICM/Xing_Scripts/QoQ_Med_JHH/analyze_data
python analyze_inpatient_data.py
```

### Output Files

The script generates the following files in the current directory:

1. `missing_data_analysis.png` - Bar chart of missing data by column
2. `numerical_distributions.png` - Histograms of numerical features
3. `correlation_matrix.png` - Correlation heatmap
4. `categorical_analysis.png` - Bar charts of top categorical values
5. `length_of_stay_distribution.png` - LOS distribution histogram
6. `dataset_summary.csv` - Summary statistics in CSV format

## Script Structure

The script uses an object-oriented approach with the `InpatientDataAnalyzer` class:

```python
class InpatientDataAnalyzer:
    - load_data()                    # Load CSV file
    - identify_column_types()        # Categorize columns
    - display_basic_info()           # Show dataset overview
    - analyze_missing_data()         # Missing value analysis
    - analyze_numerical_features()   # Numerical column analysis
    - analyze_categorical_features() # Categorical column analysis
    - analyze_datetime_features()    # Date/time column analysis
    - analyze_patient_encounters()   # Healthcare-specific analysis
    - generate_summary_report()      # Create summary report
    - run_full_analysis()           # Execute all analyses
```

## Customization

To modify the analysis, edit the `analyze_inpatient_data.py` file:

- Change the number of displayed columns in category analysis
- Adjust histogram bins
- Modify plot styles and colors
- Add custom healthcare metrics
- Export additional reports

## Notes

- The script automatically detects datetime columns based on column names
- Visualizations are saved as high-resolution PNG files (300 DPI)
- The script handles missing data gracefully
- Console output provides detailed analysis results
