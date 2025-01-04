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
    numeric_cols = numeric_cols.drop('Diagnosis')  # Exclude target column
    
    # Calculate grid dimensions
    n_features = len(numeric_cols)
    n_cols = 4  # Can be adjusted as needed
    n_rows = (n_features + n_cols - 1) // n_cols  # Round up
    
    # Create large figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    fig.suptitle('Feature Distributions by Class', fontsize=16, y=1.02)
    
    # Flatten axes array for easier iteration
    axes_flat = axes.flatten()
    
    # Create boxplots
    for idx, (col, ax) in enumerate(zip(numeric_cols, axes_flat)):
        sns.boxplot(x='Diagnosis', y=col, data=df, ax=ax)
        ax.set_title(col)
        ax.tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for idx in range(len(numeric_cols), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_plot('feature_distributions.png')

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

def generate_html_report(df, outliers, high_corr):
    os.makedirs('./reports', exist_ok=True)
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis Summary Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #34495e;
                margin-top: 30px;
            }
            .stats-table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }
            .stats-table th, .stats-table td {
                padding: 12px;
                border: 1px solid #ddd;
                text-align: left;
            }
            .stats-table th {
                background-color: #f8f9fa;
            }
            .stats-table tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            .correlation-item {
                padding: 8px;
                margin: 4px 0;
                background-color: #f8f9fa;
                border-radius: 4px;
            }
            pre {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 4px;
                overflow-x: auto;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Data Analysis Summary Report</h1>
            
            <h2>1. Dataset Overview</h2>
            <table class="stats-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total samples</td>
                    <td>{total_samples}</td>
                </tr>
                <tr>
                    <td>Features</td>
                    <td>{num_features}</td>
                </tr>
                <tr>
                    <td>Missing values</td>
                    <td>{missing_values}</td>
                </tr>
            </table>

            <h2>2. Class Distribution</h2>
            <pre>{class_distribution}</pre>

            <h2>3. Feature Statistics</h2>
            <pre>{feature_stats}</pre>

            <h2>4. Outliers Count</h2>
            <table class="stats-table">
                <tr>
                    <th>Feature</th>
                    <th>Number of Outliers</th>
                </tr>
                {outliers_rows}
            </table>

            <h2>5. Highly Correlated Features</h2>
            <div class="correlation-items">
                {correlation_items}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Prepare data for HTML
    outliers_rows = "\n".join([
        f'<tr><td>{col}</td><td>{count}</td></tr>'
        for col, count in outliers.items()
    ])
    
    correlation_items = "\n".join([
        f'<div class="correlation-item">{feat1} - {feat2}: {corr:.3f}</div>'
        for feat1, feat2, corr in high_corr
    ])
    
    # Format HTML with data
    html_content = html_content.format(
        total_samples=len(df),
        num_features=len(df.columns),
        missing_values=df.isnull().sum().sum(),
        class_distribution=df['Diagnosis'].value_counts().to_string(),
        feature_stats=df.describe().to_string(),
        outliers_rows=outliers_rows,
        correlation_items=correlation_items
    )
    
    # Save HTML report
    with open('./reports/summary_report.html', 'w') as f:
        f.write(html_content)

def generate_summary_report(df, outliers, high_corr):
    # Generate text report
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
    
    # Generate HTML report
    generate_html_report(df, outliers, high_corr)

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
    reports = [
        'profile_report.html',
        'summary_report.html'
    ]
    
    for report in reports:
        report_path = os.path.abspath(f'./reports/{report}')
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