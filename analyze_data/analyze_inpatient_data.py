#!/usr/bin/env python3
"""
Inpatient Data Analysis Script
================================
This script loads and analyzes the ACCM inpatient data from dbo.accm_inpatient.csv.
It performs comprehensive data exploration, statistical analysis, and generates visualizations.

Author: Data Analysis Team
Date: December 1, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class InpatientDataAnalyzer:
    """Class to handle inpatient data analysis"""
    
    def __init__(self, data_path):
        """
        Initialize the analyzer with data path
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = Path(data_path)
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        
    def load_data(self):
        """Load the CSV data"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Data loaded successfully from: {self.data_path}")
            print(f"✓ Dataset shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            print()
            return True
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return False
    
    def identify_column_types(self):
        """Identify and categorize column types"""
        # Identify datetime columns
        datetime_patterns = ['time', 'date', 'tim']
        for col in self.df.columns:
            if any(pattern in col.lower() for pattern in datetime_patterns):
                self.datetime_cols.append(col)
        
        # Convert datetime columns
        for col in self.datetime_cols:
            try:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            except:
                pass
        
        # Identify numerical and categorical columns
        self.numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove datetime columns from categorical
        self.categorical_cols = [col for col in self.categorical_cols if col not in self.datetime_cols]
        
    def display_basic_info(self):
        """Display basic information about the dataset"""
        print("=" * 80)
        print("BASIC INFORMATION")
        print("=" * 80)
        
        print(f"\nDataset Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Duplicate Rows: {self.df.duplicated().sum():,}")
        
        print(f"\nColumn Types:")
        print(f"  - Numerical columns: {len(self.numerical_cols)}")
        print(f"  - Categorical columns: {len(self.categorical_cols)}")
        print(f"  - Datetime columns: {len(self.datetime_cols)}")
        
        print("\n" + "-" * 80)
        print("First 5 rows:")
        print("-" * 80)
        print(self.df.head())
        print()
    
    def analyze_missing_data(self):
        """Analyze missing data in the dataset"""
        print("=" * 80)
        print("MISSING DATA ANALYSIS")
        print("=" * 80)
        
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum().values,
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df) * 100).values
        })
        
        missing_data = missing_data.sort_values('Missing_Percentage', ascending=False)
        
        total_missing = missing_data['Missing_Count'].sum()
        total_cells = len(self.df) * len(self.df.columns)
        
        print(f"\nTotal Missing Values: {total_missing:,} ({total_missing/total_cells*100:.2f}% of all cells)")
        print(f"\nColumns with Missing Data (showing top 20):")
        print("-" * 80)
        
        cols_with_missing = missing_data[missing_data['Missing_Percentage'] > 0].head(20)
        if len(cols_with_missing) > 0:
            for idx, row in cols_with_missing.iterrows():
                print(f"  {row['Column']:<40} {row['Missing_Count']:>10,} ({row['Missing_Percentage']:>6.2f}%)")
        else:
            print("  ✓ No missing data found!")
        
        print()
        
        # Visualize missing data
        if len(cols_with_missing) > 0:
            self._plot_missing_data(cols_with_missing)
    
    def _plot_missing_data(self, missing_data):
        """Plot missing data visualization"""
        plt.figure(figsize=(12, max(6, len(missing_data) * 0.3)))
        plt.barh(range(len(missing_data)), missing_data['Missing_Percentage'].values)
        plt.yticks(range(len(missing_data)), missing_data['Column'].values)
        plt.xlabel('Missing Percentage (%)')
        plt.ylabel('Column Name')
        plt.title('Columns with Missing Data')
        plt.tight_layout()
        plt.savefig('missing_data_analysis.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved visualization: missing_data_analysis.png")
        plt.close()
    
    def analyze_numerical_features(self):
        """Analyze numerical features"""
        print("=" * 80)
        print("NUMERICAL FEATURES ANALYSIS")
        print("=" * 80)
        
        if len(self.numerical_cols) == 0:
            print("  No numerical columns found.")
            return
        
        print(f"\nFound {len(self.numerical_cols)} numerical columns")
        print("-" * 80)
        
        # Display statistics
        stats = self.df[self.numerical_cols].describe()
        print("\nDescriptive Statistics:")
        print(stats)
        print()
        
        # Plot distributions
        self._plot_distributions()
        
        # Correlation analysis
        if len(self.numerical_cols) > 1:
            self._plot_correlation_matrix()
    
    def _plot_distributions(self):
        """Plot distributions of numerical features"""
        n_cols = min(9, len(self.numerical_cols))
        if n_cols == 0:
            return
        
        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(self.numerical_cols[:n_cols]):
            data = self.df[col].dropna()
            if len(data) > 0:
                axes[idx].hist(data, bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('numerical_distributions.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved visualization: numerical_distributions.png")
        plt.close()
    
    def _plot_correlation_matrix(self):
        """Plot correlation matrix"""
        corr_data = self.df[self.numerical_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved visualization: correlation_matrix.png")
        plt.close()
    
    def analyze_categorical_features(self):
        """Analyze categorical features"""
        print("=" * 80)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("=" * 80)
        
        if len(self.categorical_cols) == 0:
            print("  No categorical columns found.")
            return
        
        print(f"\nFound {len(self.categorical_cols)} categorical columns")
        print("-" * 80)
        
        # Display unique value counts
        print("\nUnique values per categorical column:")
        for col in self.categorical_cols[:15]:  # Show first 15
            unique_count = self.df[col].nunique()
            print(f"  {col:<40} {unique_count:>10,} unique values")
        
        if len(self.categorical_cols) > 15:
            print(f"  ... and {len(self.categorical_cols) - 15} more columns")
        
        print()
        
        # Analyze top categorical columns
        self._analyze_top_categories()
        
        # Plot top categories
        self._plot_top_categories()
    
    def _analyze_top_categories(self):
        """Analyze top categories in categorical columns"""
        print("\nTop categories for selected columns:")
        print("-" * 80)
        
        for col in self.categorical_cols[:5]:  # Show first 5
            print(f"\n{col}:")
            value_counts = self.df[col].value_counts().head(10)
            for idx, (value, count) in enumerate(value_counts.items(), 1):
                pct = count / len(self.df) * 100
                print(f"  {idx:2d}. {str(value):<30} {count:>10,} ({pct:>5.2f}%)")
        
        print()
    
    def _plot_top_categories(self):
        """Plot top categories"""
        n_cols = min(3, len(self.categorical_cols))
        if n_cols == 0:
            return
        
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
        if n_cols == 1:
            axes = [axes]
        
        for idx, col in enumerate(self.categorical_cols[:n_cols]):
            top_values = self.df[col].value_counts().head(10)
            axes[idx].barh(range(len(top_values)), top_values.values)
            axes[idx].set_yticks(range(len(top_values)))
            axes[idx].set_yticklabels([str(x)[:30] for x in top_values.index])
            axes[idx].set_xlabel('Count')
            axes[idx].set_title(f'Top 10 Values in {col}')
            axes[idx].invert_yaxis()
            axes[idx].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved visualization: categorical_analysis.png")
        plt.close()
    
    def analyze_datetime_features(self):
        """Analyze datetime features"""
        print("=" * 80)
        print("DATETIME FEATURES ANALYSIS")
        print("=" * 80)
        
        if len(self.datetime_cols) == 0:
            print("  No datetime columns found.")
            return
        
        print(f"\nFound {len(self.datetime_cols)} datetime columns")
        print("-" * 80)
        
        for col in self.datetime_cols[:10]:  # Show first 10
            valid_dates = self.df[col].dropna()
            if len(valid_dates) > 0:
                print(f"\n{col}:")
                print(f"  Valid dates: {len(valid_dates):,}")
                print(f"  Date range: {valid_dates.min()} to {valid_dates.max()}")
        
        print()
    
    def analyze_patient_encounters(self):
        """Specific analysis for patient encounters"""
        print("=" * 80)
        print("PATIENT ENCOUNTER ANALYSIS")
        print("=" * 80)
        
        # Analyze encounters per patient
        if 'osler_id' in self.df.columns:
            encounters_per_patient = self.df.groupby('osler_id').size()
            print(f"\nEncounters per Patient:")
            print(f"  Total unique patients: {len(encounters_per_patient):,}")
            print(f"  Mean encounters per patient: {encounters_per_patient.mean():.2f}")
            print(f"  Median encounters per patient: {encounters_per_patient.median():.0f}")
            print(f"  Max encounters per patient: {encounters_per_patient.max()}")
        
        # Analyze admission types
        if 'hosp_admsn_type_c' in self.df.columns:
            print(f"\nAdmission Types:")
            adm_types = self.df['hosp_admsn_type_c'].value_counts()
            for adm_type, count in adm_types.head(10).items():
                pct = count / len(self.df) * 100
                print(f"  Type {adm_type}: {count:,} ({pct:.2f}%)")
        
        # Analyze departments
        if 'dep_speciality' in self.df.columns:
            print(f"\nTop Departments/Specialties:")
            depts = self.df['dep_speciality'].value_counts()
            for dept, count in depts.head(10).items():
                pct = count / len(self.df) * 100
                print(f"  {str(dept):<30} {count:>10,} ({pct:>5.2f}%)")
        
        # Calculate length of stay if possible
        if 'hosp_admsn_time' in self.df.columns and 'hosp_disch_time' in self.df.columns:
            self._analyze_length_of_stay()
        
        print()
    
    def _analyze_length_of_stay(self):
        """Analyze length of stay"""
        # Convert to datetime if not already
        adm_time = pd.to_datetime(self.df['hosp_admsn_time'], errors='coerce')
        disch_time = pd.to_datetime(self.df['hosp_disch_time'], errors='coerce')
        
        # Calculate length of stay in days
        los = (disch_time - adm_time).dt.total_seconds() / (24 * 3600)
        los = los[los.notna() & (los >= 0)]  # Filter valid values
        
        if len(los) > 0:
            print(f"\nLength of Stay Analysis:")
            print(f"  Valid records: {len(los):,}")
            print(f"  Mean LOS: {los.mean():.2f} days")
            print(f"  Median LOS: {los.median():.2f} days")
            print(f"  Std Dev: {los.std():.2f} days")
            print(f"  Min LOS: {los.min():.2f} days")
            print(f"  Max LOS: {los.max():.2f} days")
            
            # Plot LOS distribution
            plt.figure(figsize=(12, 6))
            plt.hist(los[los <= los.quantile(0.95)], bins=50, edgecolor='black', alpha=0.7)
            plt.xlabel('Length of Stay (days)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Length of Stay (95th percentile)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('length_of_stay_distribution.png', dpi=300, bbox_inches='tight')
            print("  ✓ Saved visualization: length_of_stay_distribution.png")
            plt.close()
    
    def generate_summary_report(self):
        """Generate a summary report"""
        print("=" * 80)
        print("SUMMARY REPORT")
        print("=" * 80)
        
        summary = {
            'Total Rows': len(self.df),
            'Total Columns': len(self.df.columns),
            'Numerical Columns': len(self.numerical_cols),
            'Categorical Columns': len(self.categorical_cols),
            'Datetime Columns': len(self.datetime_cols),
            'Memory Usage (MB)': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f}",
            'Duplicate Rows': self.df.duplicated().sum(),
            'Total Missing Values': self.df.isnull().sum().sum(),
            'Missing Value Percentage': f"{self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100:.2f}%"
        }
        
        print("\nDataset Summary:")
        print("-" * 80)
        for key, value in summary.items():
            print(f"  {key:<30} {value}")
        
        print()
        
        # Save summary to file
        summary_df = pd.DataFrame(summary, index=['Value']).T
        summary_df.to_csv('dataset_summary.csv')
        print("  ✓ Saved summary: dataset_summary.csv")
        print()
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 80)
        print(" " * 20 + "INPATIENT DATA ANALYSIS")
        print("=" * 80)
        print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()
        
        # Load data
        if not self.load_data():
            return
        
        # Identify column types
        self.identify_column_types()
        
        # Run analyses
        self.display_basic_info()
        self.analyze_missing_data()
        self.analyze_numerical_features()
        self.analyze_categorical_features()
        self.analyze_datetime_features()
        self.analyze_patient_encounters()
        self.generate_summary_report()
        
        print("=" * 80)
        print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()


def main():
    """Main function"""
    # Define the data path
    data_path = Path(__file__).parent.parent / 'datasets' / 'dbo.accm_inpatient.csv'
    
    # Check if file exists
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    # Create analyzer and run analysis
    analyzer = InpatientDataAnalyzer(data_path)
    analyzer.run_full_analysis()
    
    print("✓ All visualizations and reports have been saved to the current directory.")
    print()


if __name__ == "__main__":
    main()
