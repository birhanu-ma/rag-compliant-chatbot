import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ComplaintDeliverableEDA:
    """
    Focused EDA for Consumer Complaints to analyze Product trends
    and Narrative characteristics for RAG pipeline planning.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.narrative_col = 'Consumer complaint narrative'
        self.product_col = 'Product'
        sns.set_theme(style="whitegrid", palette="muted")

    # 1️⃣ PRODUCT DISTRIBUTION ANALYSIS
    def analyze_product_distribution(self):
        print("\n[1/3] Analyzing Product Distribution...")
        plt.figure(figsize=(12, 6))
        
        product_counts = self.df[self.product_col].value_counts()
        sns.barplot(x=product_counts.values, y=product_counts.index, hue=product_counts.index, legend=False, palette='viridis')
        
        plt.title('Distribution of Complaints Across Products', fontsize=15)
        plt.xlabel('Number of Complaints')
        plt.ylabel('Product Category')
        plt.tight_layout()
        plt.show()
        
        # Display as a table for the deliverable
        dist_df = pd.DataFrame({'Count': product_counts, 'Percentage': (product_counts / len(self.df) * 100).round(2)})
        print(dist_df)

    # 2️⃣ NARRATIVE PRESENCE ANALYSIS
    def analyze_narrative_presence(self):
        print("\n[2/3] Identifying Narrative Availability...")
        
        # Check for presence (True if not null, False if null)
        has_narrative = self.df[self.narrative_col].notnull()
        counts = has_narrative.value_counts()
        labels = {True: 'With Narrative', False: 'No Narrative'}
        counts.index = counts.index.map(labels)

        # Plot Pie Chart
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#ff9999'], explode=(0.05, 0))
        plt.title('Complaints With vs. Without Narratives')
        plt.show()

        print(f"Total Complaints: {len(self.df)}")
        print(f"Complaints with Narrative: {counts.get('With Narrative', 0)}")
        print(f"Complaints without Narrative: {counts.get('No Narrative', 0)}")

    # 3️⃣ NARRATIVE WORD COUNT ANALYSIS
    def analyze_narrative_length(self):
        print("\n[3/3] Calculating Narrative Word Counts...")
        
        # Filter for rows that have a narrative
        narratives = self.df[self.narrative_col].dropna()
        
        # Calculate word counts (split by whitespace)
        word_counts = narratives.apply(lambda x: len(str(x).split()))

        # Visualization
        plt.figure(figsize=(12, 6))
        sns.histplot(word_counts, bins=50, kde=True, color='teal')
        
        # Add vertical lines for mean and median
        plt.axvline(word_counts.mean(), color='red', linestyle='--', label=f'Mean: {word_counts.mean():.0f}')
        plt.axvline(word_counts.median(), color='blue', linestyle='-', label=f'Median: {word_counts.median():.0f}')
        
        plt.title('Distribution of Narrative Word Counts', fontsize=15)
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        # Statistics for the report
        print("--- Word Count Statistics ---")
        print(f"Shortest Narrative: {word_counts.min()} words")
        print(f"Longest Narrative:  {word_counts.max()} words")
        print(f"Average Words:      {word_counts.mean():.2f}")
        print(f"Median Words:       {word_counts.median()}")
        
        if word_counts.max() > 500:
            print("\n> Insight: Long narratives detected. Chunking is recommended for RAG.")

    def run_deliverable_analysis(self):
        self.analyze_product_distribution()
        self.analyze_narrative_presence()
        self.analyze_narrative_length()

