"""
ML Performance Evaluation Script
=================================
Analyzes predictions logged in SQLite and generates a performance summary.
Created by: Jai (Backend Lead)
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime

# Path Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "logs", "predictions", "predictions.db")

def generate_report():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        print("Run some predictions first to generate data.")
        return

    print("=" * 60)
    print(f"NALAS ML PERFORMANCE REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        conn = sqlite3.connect(DB_PATH)
        
        # 1. Overall volume
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        total_count = len(df)
        success_count = len(df[df['status'] == 'success'])
        error_count = len(df[df['status'] == 'error'])
        
        print(f"Total Predictions Logged: {total_count}")
        print(f"Success Rate: {(success_count/total_count*100):.1f}%")
        print(f"Error Rate:   {(error_count/total_count*100):.1f}%")
        print("-" * 60)

        # 2. Method Distribution
        method_counts = df['method'].value_counts()
        print("Prediction Methods:")
        for method, count in method_counts.items():
            print(f"  - {method:15}: {count} ({(count/success_count*100):.1f}%)")
        print("-" * 60)

        # 3. Outlier Statistics
        outliers = df[df['is_outlier'] == 1]
        outlier_count = len(outliers)
        print(f"Outliers Detected: {outlier_count} ({(outlier_count/max(success_count,1)*100):.1f}% of successful predictions)")
        if outlier_count > 0:
            print("Top Outlier Items:")
            top_outliers = outliers['menu_item_id'].value_counts().head(5)
            for item, count in top_outliers.items():
                print(f"  - {item:15}: {count} occurrences")
        print("-" * 60)

        # 4. Latency Analysis
        avg_lat = df['latency_ms'].mean()
        p95_lat = df['latency_ms'].quantile(0.95)
        print("Latency Analysis (ms):")
        print(f"  - Average: {avg_lat:.2f} ms")
        print(f"  - P95:     {p95_lat:.2f} ms")
        print("-" * 60)

        # 5. Prediction Ranges
        print("Total Cost Summary (Rs.):")
        print(f"  - Min:  {df['total_cost'].min():.2f}")
        print(f"  - Max:  {df['total_cost'].max():.2f}")
        print(f"  - Mean: {df['total_cost'].mean():.2f}")

        conn.close()
    except Exception as e:
        print(f"Failed to generate report: {e}")

if __name__ == "__main__":
    generate_report()
