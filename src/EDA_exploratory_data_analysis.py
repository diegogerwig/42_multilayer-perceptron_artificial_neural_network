import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from ydata_profiling import ProfileReport
import webbrowser
import platform
import subprocess

def load_data():
    column_names = [
        "ID number", "Diagnosis", "Radius mean", "Radius se", "Radius worst",
        "Texture mean", "Texture se", "Texture worst", "Perimeter mean", 
        "Perimeter se", "Perimeter worst", "Area mean", "Area se", "Area worst",
        "Smoothness mean", "Smoothness se", "Smoothness worst", "Compactness mean",
        "Compactness se", "Compactness worst", "Concavity mean", "Concavity se",
        "Concavity worst", "Concave points mean", "Concave points se", 
        "Concave points worst", "Symmetry mean", "Symmetry se", "Symmetry worst",
        "Fractal dimension mean", "Fractal dimension se", "Fractal dimension worst"
    ]
    return pd.read_csv('./data/data.csv', names=column_names)

def analyze_class_distribution(df):
    class_dist = df['Diagnosis'].value_counts(normalize=True)
    plt.figure(figsize=(8, 6))
    class_dist.plot(kind='bar')
    plt.title('Class Distribution')
    plt.ylabel('Percentage')
    plt.tight_layout()
    save_plot('class_distribution.png')

def analyze_correlations(df):
    correlations = df.corr()['Diagnosis'].sort_values(ascending=False)
    print("\nTop 10 correlations with target:")
    print(correlations[:10])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    save_plot('correlation_matrix.png')

def analyze_feature_distributions(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Diagnosis', y=col, data=df)
        plt.title(f'{col} Distribution by Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_plot(f'dist_{col}.png')

def detect_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers[col] = len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)])
    return outliers

def analyze_multicollinearity(df, threshold=0.8):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i,j])
                 for i, j in zip(*np.where(upper > threshold))]
    return high_corr

def save_plot(filename):
    os.makedirs('./plots', exist_ok=True)
    plt.savefig(f'./plots/{filename}')
    plt.close()

def generate_summary_report(df, outliers, high_corr):
    os.makedirs('./reports', exist_ok=True)
    with open('./reports/summary_report.txt', 'w') as f:
        f.write("Data Analysis Summary Report\n")
        f.write("==========================\n\n")
        
        f.write("1. Dataset Overview\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Features: {len(df.columns)}\n")
        f.write(f"Missing values: {df.isnull().sum().sum()}\n\n")
        
        f.write("2. Class Distribution\n")
        class_dist = df['Diagnosis'].value_counts()
        f.write(str(class_dist) + "\n\n")
        
        f.write("3. Feature Statistics\n")
        f.write(str(df.describe()) + "\n\n")
        
        f.write("4. Outliers Count\n")
        for col, count in outliers.items():
            f.write(f"{col}: {count}\n")
        f.write("\n")
        
        f.write("5. Highly Correlated Features\n")
        for feat1, feat2, corr in high_corr:
            f.write(f"{feat1} - {feat2}: {corr:.3f}\n")

def perform_eda(df):
    print("Starting Exploratory Data Analysis...")
    print("Initial columns:", df.columns.tolist())
    
    df = df.copy()
    print("After copy columns:", df.columns.tolist())
    
    # First convert diagnosis
    df['Diagnosis'] = pd.to_numeric(df['Diagnosis'].map({'M': 1, 'B': 0}))
    print("After diagnosis conversion:", df.dtypes['Diagnosis'])
    
    # Then drop ID
    df = df.drop('ID number', axis=1)
    print("After ID drop columns:", df.columns.tolist())
    
    # Finally convert remaining columns
    for col in df.columns:
        if col != 'Diagnosis':
            df[col] = pd.to_numeric(df[col])
    print("Final dtypes:", df.dtypes)
    
    analyze_class_distribution(df)
    analyze_correlations(df)
    analyze_feature_distributions(df)
    
    outliers = detect_outliers(df)
    high_corr = analyze_multicollinearity(df)
    
    generate_summary_report(df, outliers, high_corr)
    profile = ProfileReport(df, title="Breast Cancer Data Analysis")
    profile.to_file('./reports/profile_report.html')

def open_reports():
    report_path = os.path.abspath('./reports/profile_report.html')
    if platform.system().lower() == 'linux':
        if 'microsoft' in platform.uname().release.lower():
            try:
                windows_path = subprocess.check_output(['wslpath', '-w', report_path], 
                                                     stderr=subprocess.PIPE).decode().strip()
                subprocess.Popen(['cmd.exe', '/c', 'start', windows_path])
            except subprocess.CalledProcessError:
                subprocess.Popen(['xdg-open', report_path])
        else:
            subprocess.Popen(['xdg-open', report_path])
    else:
        webbrowser.open('file://' + report_path)

def main():
    try:
        df = load_data()
        print("Data loaded successfully")
        
        perform_eda(df)
        print("EDA completed. Check ./reports and ./plots directories")
        
        open_reports()
        print("Opening reports in browser...")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()