import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

class ComplaintProfiling:
    """
    Performs data profiling optimized for the Consumer Complaint dataset,
    focusing on temporal patterns, categorical distributions, and text presence.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Convert date columns to datetime objects automatically
        date_cols = ['Date received', 'Date sent to company']
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

    # 1️⃣ DATA OVERVIEW
    def overview(self):
        print("====== CONSUMER COMPLAINT DATA OVERVIEW ======")
        print(f"Total Complaints: {self.df.shape[0]}")
        print(f"Total Features: {self.df.shape[1]}")
        print("\nData Types Summary:")
        display(self.df.dtypes.value_counts())
        print("\nFirst 3 Rows:")
        display(self.df.head(3))

    # 2️⃣ SUMMARY STATISTICS (Categorical & Logical)
    def summary_statistics(self):
        print("\n====== SUMMARY STATISTICS ======")
        
        # Categorical Breakdown (Focus on Products and Issues)
        core_categories = ['Product', 'Sub-product', 'Issue', 'State', 'Submitted via']
        existing_cats = [c for c in core_categories if c in self.df.columns]

        if existing_cats:
            print("\n--- Core Category Distribution (Top 5) ---")
            for col in existing_cats:
                print(f"\nTop 5 values for {col}:")
                display(self.df[col].value_counts().head(5))

        # Binary/Boolean check (Timely Response, Consumer Disputed)
        binary_cols = ['Timely response?', 'Consumer disputed?', 'Consumer consent provided?']
        existing_binary = [c for c in binary_cols if c in self.df.columns]
        
        if existing_binary:
            print("\n--- Process & Response Status ---")
            for col in existing_binary:
                display(self.df[col].value_counts(dropna=False))

    # 3️⃣ MISSING VALUES (COUNT + PERCENTAGE)
    def missing_value_summary(self):
        null_count = self.df.isnull().sum()
        null_percent = (null_count / len(self.df)) * 100
        missing_summary = pd.DataFrame({
            'Missing Values': null_count,
            'Percentage (%)': null_percent
        })
        missing_summary = (
            missing_summary[missing_summary['Missing Values'] > 0]
            .sort_values(by='Percentage (%)', ascending=False)
        )
        print("\n====== MISSING DATA SUMMARY ======")
        if missing_summary.empty:
            print("No missing values found.")
        else:
            display(missing_summary)
            print("> Note: Missing 'Consumer complaint narrative' is common if consent wasn't given.")

    # 4️⃣ DUPLICATE CHECK
    def duplicate_check(self):
        # We check duplicates based on Complaint ID if it exists, else full row
        print("\n====== DUPLICATE CHECK ======")
        if 'Complaint ID' in self.df.columns:
            id_dups = self.df['Complaint ID'].duplicated().sum()
            print(f"Duplicate Complaint IDs Found: {id_dups}")
        
        row_dups = self.df.duplicated().sum()
        print(f"Exact Duplicate Rows Found: {row_dups}")

    # 5️⃣ TEMPORAL ANALYSIS (Date Range Detection)
    def temporal_summary(self):
        print("\n====== TEMPORAL SUMMARY (DATE RANGE) ======")
        date_cols = self.df.select_dtypes(include=['datetime64']).columns
        
        if not date_cols.empty:
            for col in date_cols:
                print(f"\nRange for {col}:")
                print(f"Min: {self.df[col].min()}")
                print(f"Max: {self.df[col].max()}")
        else:
            print("No date columns detected for temporal analysis.")

    # 6️⃣ RUN ALL
    def run_all(self):
        self.overview()
        self.summary_statistics()
        self.missing_value_summary()
        self.duplicate_check()
        self.temporal_summary()
        return self.df