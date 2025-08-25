#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Statistical Analysis and Plotting Script
Input: Processed data from all participants (features, physiological data)
Process: Group comparisons, statistical tests, effect size calculation, plotting.
Output: Statistical results (Tables 8,9,10), figures, and summary.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

def load_all_data(participants_info_path, features_path):
    """
    Load the master spreadsheet with participant info and all extracted features.
    Args:
        participants_info_path (str): Path to Excel/CSV with participant demographics and physiological data.
        features_path (str): Path to CSV containing all extracted movement features for all participants.
    Returns:
        merged_df (DataFrame): Merged dataframe with all info and features for analysis.
    """
    df_info = pd.read_excel(participants_info_path) # Or pd.read_csv
    df_features = pd.read_csv(features_path)
    
    # Merge on participant ID
    merged_df = pd.merge(df_info, df_features, on='participant_id')
    return merged_df

def perform_group_comparisons(df, group_var='age_group', young_label='18-30', old_label='>45'):
    """
    Perform t-tests or Mann-Whitney U tests between two groups for all outcome measures.
    Args:
        df (DataFrame): The merged dataframe.
        group_var (str): Column name that defines the groups.
        young_label: Value for the young group.
        old_label: Value for the older group.
    Returns:
        results_df (DataFrame): A dataframe containing p-values, effect sizes, and CIs for each variable.
    """
    young_group = df[df[group_var] == young_label]
    old_group = df[df[group_var] == old_label]
    
    # List of continuous outcome variables to compare
    outcome_vars = ['PF', 'CIA', 'PF_BC', 'PF_AC', 'CIA_BC', 'CIA_AC', 'HRmax', 'VO2max', 'METS', 'EEm', 'RPE']
    
    results = []
    for var in outcome_vars:
        y_data = young_group[var].dropna()
        o_data = old_group[var].dropna()
        
        # Check normality (Shapiro-Wilk) on each group
        _, p_young = stats.shapiro(y_data)
        _, p_old = stats.shapiro(o_data)
        
        if p_young < 0.05 or p_old < 0.05:
            # Use non-parametric test (Mann-Whitney U)
            stat, p_val = stats.mannwhitneyu(y_data, o_data, alternative='two-sided')
            test_used = "Mann-Whitney U"
            # Calculate effect size r = z / sqrt(N)
            n1, n2 = len(y_data), len(o_data)
            z = stats.norm.ppf(1 - (p_val / 2)) # Approximate z from U stat
            r_effect = z / np.sqrt(n1 + n2)
            ci_low, ci_high = None, None # MWU doesn't provide a straightforward CI for the difference in medians easily
        else:
            # Use independent t-test
            stat, p_val = stats.ttest_ind(y_data, o_data, equal_var=False) # Welch's t-test
            test_used = "Welch's t-test"
            # Calculate Cohen's d
            pooled_std = np.sqrt(((len(y_data)-1)*np.var(y_data, ddof=1) + (len(o_data)-1)*np.var(o_data, ddof=1)) / (len(y_data)+len(o_data)-2))
            d_effect = (np.mean(y_data) - np.mean(o_data)) / pooled_std
            # Calculate 95% CI for Cohen's d (approximate)
            se_d = np.sqrt((len(y_data) + len(o_data)) / (len(y_data)*len(o_data)) + (d_effect**2) / (2*(len(y_data)+len(o_data))))
            ci_low = d_effect - 1.96 * se_d
            ci_high = d_effect + 1.96 * se_d
            r_effect = None
        
        # Store results
        result_row = {
            'Variable': var,
            'Test': test_used,
            'Statistic': stat,
            'p_value': p_val,
            'Effect_Size_Type': 'Cohen\'s d' if test_used.startswith('t') else 'r',
            'Effect_Size_Value': d_effect if test_used.startswith('t') else r_effect,
            'CI_Lower': ci_low,
            'CI_Upper': ci_high
        }
        results.append(result_row)
    
    results_df = pd.DataFrame(results)
    return results_df

def create_summary_table(df, group_var='age_group', young_label='18-30', old_label='>45'):
    """
    Create a summary table (like Table 8 and 9 combined) with means and standard deviations per group.
    """
    summary = df.groupby(group_var).agg({
        'PF': ['mean', 'std'],
        'CIA': ['mean', 'std'],
        'PF_BC': ['mean', 'std'],
        'PF_AC': ['mean', 'std'],
        'CIA_BC': ['mean', 'std'],
        'CIA_AC': ['mean', 'std'],
        'HRmax': ['mean', 'std'],
        'VO2max': ['mean', 'std'],
        'METS': ['mean', 'std'],
        'EEm': ['mean', 'std'],
        'RPE': ['mean', 'std']
    }).round(2).T # Transpose for a better view
    
    return summary

def plot_key_findings(df):
    """
    Generate the key plots for the manuscript.
    """
    # Example 1: Boxplot of CIA-AC by Group
    plt.figure(figsize=(8,6))
    sns.boxplot(x='age_group', y='CIA_AC', data=df)
    plt.title('Complexity Index Average after Change (CIA-AC) by Age Group')
    plt.ylabel('CIA-AC')
    plt.xlabel('Age Group')
    plt.tight_layout()
    plt.savefig('cia_ac_by_group.png', dpi=300)
    plt.close()

    # Example 2: Scatter plot of CIA-AC vs %HRR
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='%HRR', y='CIA_AC', hue='age_group', data=df)
    plt.title('CIA-AC vs Percentage of Heart Rate Reserve (%HRR)')
    plt.xlabel('%HRR')
    plt.ylabel('CIA-AC')
    plt.legend(title='Age Group')
    plt.tight_layout()
    plt.savefig('cia_ac_vs_hrr.png', dpi=300)
    plt.close()

    # ... Add more plots as needed (STFT spectrograms, MSE curves for representative participants) ...

if __name__ == "__main__":
    # --- Configuration ---
    PARTICIPANTS_PATH = "path/to/your/participants_master_data.xlsx"
    FEATURES_PATH = "path/to/your/all_participants_features.csv"
    OUTPUT_DIR = "path/to/your/analysis/results"

    # --- Load Data ---
    print("Loading all data...")
    df_full = load_all_data(PARTICIPANTS_PATH, FEATURES_PATH)

    # --- Perform Statistical Analysis ---
    print("Performing group comparisons...")
    results_table = perform_group_comparisons(df_full)
    summary_table = create_summary_table(df_full)

    # --- Save Results to CSV ---
    results_table.to_csv(os.path.join(OUTPUT_DIR, 'statistical_test_results.csv'), index=False)
    summary_table.to_csv(os.path.join(OUTPUT_DIR, 'summary_statistics.csv'))

    # --- Generate Plots ---
    print("Generating plots...")
    plot_key_findings(df_full)

    # --- Print Key Results to Console ---
    print("\n--- Key Results ---")
    print(summary_table)
    print("\n--- Statistical Significance ---")
    print(results_table[['Variable', 'Test', 'p_value', 'Effect_Size_Type', 'Effect_Size_Value']])

    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}")