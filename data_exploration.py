import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

def load_and_explore_cancer_data():
    print("\n=== Cancer Dataset Exploration ===")
    df = pd.read_csv('Global Cancer Patients/global_cancer_patients_2015_2024.csv')
    
    # Basic information
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nData Types:\n", df.dtypes)
    
    # Summary statistics
    print("\nSummary Statistics:\n", df.describe())
    
    # Check for class imbalance
    if 'target' in df.columns:
        print("\nClass Distribution:\n", df['target'].value_counts(normalize=True))
    
    return df

def load_and_explore_bankruptcy_data():
    print("\n=== Bankruptcy Dataset Exploration ===")
    df = pd.read_csv('Company Bankruptcy/company_bankruptcy_data.csv')
    
    # Basic information
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nData Types:\n", df.dtypes)
    
    # Summary statistics
    print("\nSummary Statistics:\n", df.describe())
    
    # Check for class imbalance
    if 'target' in df.columns:
        print("\nClass Distribution:\n", df['target'].value_counts(normalize=True))
    
    return df

def plot_correlation_matrix(df, title):
    # Only use numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 8))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
    plt.title(f'Correlation Matrix - {title}')
    plt.tight_layout()
    plt.savefig(f'correlation_matrix_{title.lower().replace(" ", "_")}.png')
    plt.close()

def engineer_cancer_severity_label(df):
    # Use median as threshold for high/low severity
    threshold = df['Target_Severity_Score'].median()
    df['High_Severity'] = (df['Target_Severity_Score'] > threshold).astype(int)
    print(f"\nHigh_Severity threshold: {threshold}")
    print("Class balance (High_Severity):\n", df['High_Severity'].value_counts(normalize=True))
    return df

def visualize_class_balance(df, target_col, title):
    plt.figure(figsize=(6,4))
    sns.countplot(x=target_col, data=df)
    plt.title(f'Class Balance: {title}')
    plt.savefig(f'class_balance_{title.lower().replace(" ", "_")}.png')
    plt.close()

def prepare_and_save_datasets():
    # Cancer dataset
    cancer_df = pd.read_csv('Global Cancer Patients/global_cancer_patients_2015_2024.csv')
    cancer_df = engineer_cancer_severity_label(cancer_df)
    visualize_class_balance(cancer_df, 'High_Severity', 'Cancer Severity')
    cancer_df.to_csv('Global Cancer Patients/cancer_processed.csv', index=False)

    # Bankruptcy dataset
    bankruptcy_df = pd.read_csv('Company Bankruptcy/company_bankruptcy_data.csv')
    visualize_class_balance(bankruptcy_df, 'Bankrupt?', 'Bankruptcy')
    bankruptcy_df.to_csv('Company Bankruptcy/bankruptcy_processed.csv', index=False)

def main():
    # Load and explore both datasets
    cancer_df = load_and_explore_cancer_data()
    bankruptcy_df = load_and_explore_bankruptcy_data()
    
    # Plot correlation matrices
    plot_correlation_matrix(cancer_df, 'Cancer Dataset')
    plot_correlation_matrix(bankruptcy_df, 'Bankruptcy Dataset')
    
    # Save basic statistics to files
    cancer_df.describe().to_csv('cancer_statistics.csv')
    bankruptcy_df.describe().to_csv('bankruptcy_statistics.csv')

    # Prepare and save processed datasets, visualize class balance
    prepare_and_save_datasets()

if __name__ == "__main__":
    main() 