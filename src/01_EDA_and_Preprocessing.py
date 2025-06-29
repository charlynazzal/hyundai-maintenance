import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    df = pd.read_csv(filepath)
    print("Data loaded successfully.")
    return df

def initial_inspection(df):
    """Performs an initial inspection of the dataframe."""
    print("Data Info:")
    df.info()
    print("\nDescriptive Statistics:")
    print(df.describe())
    
def plot_anomaly_distribution(df):
    """Plots the distribution of the 'Anomaly Indication'."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Anomaly Indication', data=df)
    plt.title('Distribution of Anomaly Indication')
    plt.savefig('images/anomaly_distribution.png')
    print("\nSaved anomaly distribution plot to images/anomaly_distribution.png")
    plt.close()

def plot_feature_distributions(df):
    """Plots distributions of numerical and categorical features."""
    print("\nPlotting feature distributions...")
    # Numerical features
    numerical_features = ['Engine Temperature (°C)', 'Brake Pad Thickness (mm)', 'Tire Pressure (PSI)']
    for col in numerical_features:
        plt.figure(figsize=(10, 5))
        sns.histplot(df, x=col, hue='Anomaly Indication', kde=True, multiple="stack")
        plt.title(f'Distribution of {col} by Anomaly Indication')
        plt.savefig(f'images/{col.replace(" ", "_").replace("(°C)", "C").replace("(mm)", "mm").replace("(PSI)", "PSI")}_distribution.png')
        plt.close()

    # Categorical feature
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Maintenance Type', hue='Anomaly Indication')
    plt.title('Maintenance Type vs. Anomaly Indication')
    plt.xticks(rotation=15)
    plt.savefig('images/maintenance_type_analysis.png')
    plt.close()
    print("Feature distribution plots saved to images/")

def plot_correlation_heatmap(df):
    """Plots the correlation heatmap for numerical features."""
    print("\nPlotting correlation heatmap...")
    plt.figure(figsize=(8, 6))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=np.number)
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig('images/correlation_heatmap.png')
    plt.close()
    print("Correlation heatmap saved to images/")

def main():
    """Main function to run the EDA script."""
    sns.set_style('whitegrid')
    
    # Load data
    df = load_data('data/cars_hyundai.csv')
    
    # Display first 5 rows
    print("First 5 rows of the dataset:")
    print(df.head())
    
    # Initial data inspection
    initial_inspection(df)
    
    # Plot target variable distribution
    plot_anomaly_distribution(df)
    
    # Plot feature distributions
    plot_feature_distributions(df)

    # Plot correlation heatmap
    plot_correlation_heatmap(df)
    
    print("\nEDA script finished.")

if __name__ == '__main__':
    main() 