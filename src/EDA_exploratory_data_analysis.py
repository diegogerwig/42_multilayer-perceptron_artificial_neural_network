import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from ydata_profiling import ProfileReport
import webbrowser
import platform
import subprocess
from pathlib import Path
import sys
from contextlib import redirect_stdout, redirect_stderr
import base64
from io import BytesIO
from datetime import datetime
from colorama import init, Fore, Style

init(autoreset=True)  # Initialize colorama

def encode_image_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def load_data():
    print(f"{Fore.YELLOW}📂 Loading dataset...")
    column_names = [
        "ID number", 
        "Diagnosis", 
        "Radius mean", "Radius se", "Radius worst",
        "Texture mean", "Texture se", "Texture worst", 
        "Perimeter mean", "Perimeter se", "Perimeter worst", 
        "Area mean", "Area se", "Area worst",
        "Smoothness mean", "Smoothness se", "Smoothness worst", 
        "Compactness mean", "Compactness se", "Compactness worst", 
        "Concavity mean", "Concavity se", "Concavity worst", 
        "Concave points mean", "Concave points se", "Concave points worst", 
        "Symmetry mean", "Symmetry se", "Symmetry worst",
        "Fractal dimension mean", "Fractal dimension se", "Fractal dimension worst"
    ]
    return pd.read_csv('./data/raw/data.csv', names=column_names)

def create_violin_plots(df):
    """Create violin plots for the most important features"""
    print(f"{Fore.YELLOW}📊 Generating violin plots...")
    top_features = ['Radius mean', 'Texture mean', 'Perimeter mean', 'Area mean']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribution of Key Features by Diagnosis', fontsize=16)
    
    for idx, feature in enumerate(top_features):
        row = idx // 2
        col = idx % 2
        sns.violinplot(x='Diagnosis', y=feature, data=df, ax=axes[row, col])
        axes[row, col].set_title(feature)
        axes[row, col].set_xlabel('Diagnosis (0=Benign, 1=Malignant)')
    
    plt.tight_layout()
    return fig

def create_density_plots(df):
    """Create density plots for mean features"""
    print(f"{Fore.YELLOW}📈 Generating density plots...")
    mean_features = [col for col in df.columns if 'mean' in col]
    
    # Calculate grid dimensions
    n_features = len(mean_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    fig.suptitle('Density Plots of Mean Features by Diagnosis', fontsize=16)
    
    # Flatten axes for easier iteration if we have multiple rows
    if n_rows > 1:
        axes = axes.flatten()
    
    # Create density plots
    for idx, feature in enumerate(mean_features):
        ax = axes[idx] if n_rows > 1 else axes[idx % n_cols]
        sns.kdeplot(data=df, x=feature, hue='Diagnosis', ax=ax)
        ax.set_title(feature)
        ax.tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for idx in range(len(mean_features), len(axes) if n_rows > 1 else n_cols):
        if n_rows > 1:
            axes[idx].set_visible(False)
        else:
            axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

def analyze_correlations(df):
    print(f"{Fore.YELLOW}🔄 Analyzing correlations...")
    correlations = df.corr()['Diagnosis'].sort_values(ascending=False)
    print(f"\n{Fore.WHITE}Top 10 correlations with target:")
    for feat, corr in correlations[1:11].items():
        print(f"{Fore.BLUE}   {feat}: {corr:.3f}")
    return correlations

def detect_outliers(df):
    print(f"{Fore.YELLOW}🔍 Detecting outliers...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers[col] = len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)])
    return outliers

def analyze_multicollinearity(df, threshold=0.8):
    print(f"{Fore.YELLOW}🔗 Analyzing feature relationships...")
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i,j])
                 for i, j in zip(*np.where(upper > threshold))]
    return high_corr

def generate_html_report(df, outliers, high_corr_sorted, feature_stats, target_corrs):
    print(f"{Fore.YELLOW}📝 Generating HTML report...")
    
    # Generate all plots and convert to base64
    class_dist_fig = plt.figure(figsize=(8, 6))
    df['Diagnosis'].value_counts(normalize=True).plot(kind='bar')
    plt.title('Class Distribution')
    plt.ylabel('Percentage')
    class_dist_base64 = encode_image_to_base64(class_dist_fig)
    plt.close()

    # Correlation matrix
    corr_fig = plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    corr_base64 = encode_image_to_base64(corr_fig)
    plt.close()

    # Additional plots
    violin_base64 = encode_image_to_base64(create_violin_plots(df))
    density_base64 = encode_image_to_base64(create_density_plots(df))
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    class_dist = df['Diagnosis'].value_counts()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis Summary Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }}
            .container {{ background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            h3 {{ color: #34495e; margin-top: 20px; }}
            .stats-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            .stats-table th, .stats-table td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
            .stats-table th {{ background-color: #f8f9fa; }}
            .stats-table tr:nth-child(even) {{ background-color: #f8f9fa; }}
            .plot-container {{ margin: 20px 0; padding: 15px; background-color: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            .plot-container img {{ max-width: 100%; height: auto; }}
            .plot-title {{ color: #2c3e50; margin-bottom: 10px; }}
            .highlight {{ background-color: #fff3cd; padding: 15px; border-radius: 4px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Data Analysis Summary Report</h1>
            <p>Generated on: {current_time}  by: dgerwig</p>
            
            <h2>1. Dataset Overview</h2>
            <table class="stats-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total samples</td><td>{len(df)}</td></tr>
                <tr><td>Features</td><td>{len(df.columns)}</td></tr>
                <tr><td>Missing values</td><td>{df.isnull().sum().sum()}</td></tr>
                <tr><td>Data balance ratio (Benign/Malignant)</td>
                    <td>{(df['Diagnosis'] == 0).sum() / (df['Diagnosis'] == 1).sum():.2f}</td></tr>
            </table>

            <h2>2. Class Distribution</h2>
            <div class="plot-container">
                <h3 class="plot-title">Class Distribution Plot</h3>
                <img src="data:image/png;base64,{class_dist_base64}" alt="Class Distribution">
            </div>

            <table class="stats-table">
                <tr><th>Class</th><th>Count</th><th>Percentage</th></tr>
                <tr><td>0 (Benign)</td><td>{class_dist[0]}</td><td>{(class_dist[0]/len(df)*100):.1f}%</td></tr>
                <tr><td>1 (Malignant)</td><td>{class_dist[1]}</td><td>{(class_dist[1]/len(df)*100):.1f}%</td></tr>
            </table>

            <h2>3. Feature Correlations</h2>
            <div class="plot-container">
                <h3 class="plot-title">Correlation Matrix</h3>
                <img src="data:image/png;base64,{corr_base64}" alt="Correlation Matrix">
            </div>

            <h2>4. Feature Distributions</h2>
            <div class="plot-container">
                <h3 class="plot-title">Violin Plots of Key Features</h3>
                <img src="data:image/png;base64,{violin_base64}" alt="Violin Plots">
            </div>

            <div class="plot-container">
                <h3 class="plot-title">Density Plots of Mean Features</h3>
                <img src="data:image/png;base64,{density_base64}" alt="Density Plots">
            </div>

            <h2>5. Data Quality Analysis</h2>
            <h3>Outliers Summary</h3>
            <div class="highlight">
                <p>Total outliers in dataset: {feature_stats['total_outliers']}</p>
                <p>Average outliers per feature: {feature_stats['mean_outliers_per_feature']:.2f}</p>
                <p>Feature with most outliers: {feature_stats['most_outliers'][0]} 
                   ({feature_stats['most_outliers'][1]} outliers)</p>
            </div>

            <h3>Top Feature Correlations with Diagnosis</h3>
            <table class="stats-table">
                <tr><th>Feature</th><th>Correlation</th></tr>
                {''.join(f'<tr><td>{feat}</td><td>{corr:.3f}</td></tr>' 
                        for feat, corr in target_corrs[1:6].items())}
            </table>
        </div>
    </body>
    </html>
    """
    
    os.makedirs('./report', exist_ok=True)
    with open('./report/summary_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"{Fore.GREEN}✅ HTML report generated successfully")

def perform_eda(df):
    print(f"\n{Fore.YELLOW}🔬 Starting Exploratory Data Analysis...")
    
    print(f"{Fore.WHITE}   Initial shape: {Fore.BLUE}{df.shape}")
    
    df = df.copy()
    
    # Convert diagnosis
    df['Diagnosis'] = pd.to_numeric(df['Diagnosis'].map({'M': 1, 'B': 0}))
    print(f"{Fore.WHITE}   Diagnosis values converted to: {Fore.BLUE}[0: Benign, 1: Malignant]")
    
    # Drop ID
    df = df.drop('ID number', axis=1)
    print(f"{Fore.WHITE}   ID column removed")
    
    # Convert remaining columns
    for col in df.columns:
        if col != 'Diagnosis':
            df[col] = pd.to_numeric(df[col])
    print(f"{Fore.WHITE}   All features converted to numeric type")
    
    # Perform analysis
    correlations = analyze_correlations(df)
    outliers = detect_outliers(df)
    high_corr = analyze_multicollinearity(df)
    
    # Calculate feature stats
    feature_stats = {
        'mean_outliers_per_feature': sum(outliers.values()) / len(outliers),
        'most_outliers': max(outliers.items(), key=lambda x: x[1]),
        'total_outliers': sum(outliers.values())
    }
    
    # Calculate target correlations
    target_corrs = df.corr()['Diagnosis'].sort_values(key=abs, ascending=False)
    
    # Sort high correlations
    high_corr_sorted = sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)
    
    generate_html_report(df, outliers, high_corr_sorted, feature_stats, target_corrs)
    
    # print(f"\n{Fore.YELLOW}📊 Generating profile report...")
    # profile = ProfileReport(df, title="Breast Cancer Data Analysis")
    # profile.to_file('./report/profile_report.html')

def open_reports():
    try:
        import webbrowser
        
        paths = [
            os.path.abspath('./report/summary_report.html'),
            # os.path.abspath('./report/profile_report.html')
        ]
        
        # Use the default browser without trying to force Chrome
        for path in paths:
            if os.path.exists(path):
                url = 'file://' + path
                webbrowser.open(url)
                print(f"{Fore.GREEN}✅ Opened {path}")
            else:
                print(f"{Fore.RED}❌ File not found: {path}")
            
    except Exception as e:
        print(f"{Fore.RED}❌ Error: {str(e)}")
        print(f"{Fore.YELLOW}💡 Los archivos están en el directorio ./report - puedes abrirlos manualmente:")
        
def main():
    try:
        print(f"\n{Fore.YELLOW}🔬 Breast Cancer Dataset Analysis Tool")
        print(f"{Fore.WHITE}   This tool performs exploratory data analysis on breast cancer data\n")
        
        df = load_data()
        print(f"{Fore.GREEN}✅ Data loaded successfully")
        
        perform_eda(df)
        print(f"\n{Fore.GREEN}✨ EDA completed. Reports generated in ./report directory")
        
        import time
        time.sleep(1)
        
        open_reports()
        
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {str(e)}")

if __name__ == "__main__":
    main()