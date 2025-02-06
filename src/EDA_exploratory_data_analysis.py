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
    plt.style.use('dark_background')
    df['Diagnosis'].value_counts(normalize=True).plot(kind='bar', color=['#00CED1', '#ff3366'])
    plt.title('Class Distribution')
    plt.ylabel('Percentage')
    class_dist_base64 = encode_image_to_base64(class_dist_fig)
    plt.close()

    corr_fig = plt.figure(figsize=(12, 8))
    plt.style.use('dark_background')
    sns.heatmap(df.corr(), cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    corr_base64 = encode_image_to_base64(corr_fig)
    plt.close()

    violin_base64 = encode_image_to_base64(create_violin_plots(df))
    density_base64 = encode_image_to_base64(create_density_plots(df))
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    class_dist = df['Diagnosis'].value_counts()
    
    html_content = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis Summary Report</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-gray-100">
        <div class="container mx-auto px-4 py-8">
            <h1 class="text-3xl font-bold mb-8 text-blue-400">Data Analysis Summary Report</h1>
            <div class="bg-gray-800 p-4 rounded-lg italic text-gray-400 mb-8">
                <strong>Generated on:</strong> {current_time} | <strong>Analyst:</strong> dgerwig
            </div>
            
            <div class="mb-12">
                <h2 class="text-2xl font-bold mb-6 text-blue-400">1. Dataset Overview</h2>
                <div class="bg-gray-800 rounded-lg overflow-hidden">
                    <table class="w-full">
                        <tr class="border-b border-gray-700">
                            <th class="bg-gray-700 text-blue-400 p-4 text-left">Metric</th>
                            <th class="bg-gray-700 text-blue-400 p-4 text-left">Value</th>
                        </tr>
                        <tr class="border-b border-gray-700 hover:bg-gray-700">
                            <td class="p-4">Total samples</td>
                            <td class="p-4 text-blue-400 font-bold">{len(df)}</td>
                        </tr>
                        <tr class="border-b border-gray-700 hover:bg-gray-700">
                            <td class="p-4">Features</td>
                            <td class="p-4 text-blue-400 font-bold">{len(df.columns)-1}</td>
                        </tr>
                        <tr class="border-b border-gray-700 hover:bg-gray-700">
                            <td class="p-4">Missing values</td>
                            <td class="p-4 text-blue-400 font-bold">{df.isnull().sum().sum()}</td>
                        </tr>
                        <tr class="hover:bg-gray-700">
                            <td class="p-4">Data balance ratio (Benign/Malignant)</td>
                            <td class="p-4 text-blue-400 font-bold">{(df['Diagnosis'] == 0).sum() / (df['Diagnosis'] == 1).sum():.2f}</td>
                        </tr>
                    </table>
                </div>
            </div>

            <div class="mb-12">
                <h2 class="text-2xl font-bold mb-6 text-blue-400">2. Class Distribution</h2>
                <div class="bg-gray-800 p-6 rounded-lg shadow-lg mb-6">
                    <h3 class="text-xl font-bold mb-4 text-blue-400">Class Distribution Plot</h3>
                    <img src="data:image/png;base64,{class_dist_base64}" alt="Class Distribution" class="w-full rounded-lg border border-gray-700">
                </div>

                <div class="bg-gray-800 rounded-lg overflow-hidden">
                    <table class="w-full">
                        <tr class="border-b border-gray-700">
                            <th class="bg-gray-700 text-blue-400 p-4 text-left">Class</th>
                            <th class="bg-gray-700 text-blue-400 p-4 text-left">Count</th>
                            <th class="bg-gray-700 text-blue-400 p-4 text-left">Percentage</th>
                        </tr>
                        <tr class="border-b border-gray-700 hover:bg-gray-700">
                            <td class="p-4">0 (Benign)</td>
                            <td class="p-4 text-blue-400 font-bold">{class_dist[0]}</td>
                            <td class="p-4 text-blue-400 font-bold">{(class_dist[0]/len(df)*100):.1f}%</td>
                        </tr>
                        <tr class="hover:bg-gray-700">
                            <td class="p-4">1 (Malignant)</td>
                            <td class="p-4 text-blue-400 font-bold">{class_dist[1]}</td>
                            <td class="p-4 text-blue-400 font-bold">{(class_dist[1]/len(df)*100):.1f}%</td>
                        </tr>
                    </table>
                </div>
            </div>

            <div class="mb-12">
                <h2 class="text-2xl font-bold mb-6 text-blue-400">3. Feature Correlations</h2>
                <div class="bg-gray-800 p-6 rounded-lg shadow-lg">
                    <h3 class="text-xl font-bold mb-4 text-blue-400">Correlation Matrix</h3>
                    <img src="data:image/png;base64,{corr_base64}" alt="Correlation Matrix" class="w-full rounded-lg border border-gray-700">
                </div>
            </div>

            <div class="mb-12">
                <h2 class="text-2xl font-bold mb-6 text-blue-400">4. Feature Distributions</h2>
                <div class="bg-gray-800 p-6 rounded-lg shadow-lg mb-6">
                    <h3 class="text-xl font-bold mb-4 text-blue-400">Violin Plots of Key Features</h3>
                    <img src="data:image/png;base64,{violin_base64}" alt="Violin Plots" class="w-full rounded-lg border border-gray-700">
                </div>

                <div class="bg-gray-800 p-6 rounded-lg shadow-lg">
                    <h3 class="text-xl font-bold mb-4 text-blue-400">Density Plots of Mean Features</h3>
                    <img src="data:image/png;base64,{density_base64}" alt="Density Plots" class="w-full rounded-lg border border-gray-700">
                </div>
            </div>

            <div class="mb-12">
                <h2 class="text-2xl font-bold mb-6 text-blue-400">5. Data Quality Analysis</h2>
                <div class="bg-gray-800 p-6 rounded-lg mb-8 border-l-4 border-pink-500">
                    <h3 class="text-xl font-bold mb-4 text-blue-400">Outliers Summary</h3>
                    <p class="mb-2">
                        <strong>Total outliers in dataset:</strong> 
                        <span class="text-blue-400 font-bold">{feature_stats['total_outliers']}</span>
                    </p>
                    <p class="mb-2">
                        <strong>Average outliers per feature:</strong> 
                        <span class="text-blue-400 font-bold">{feature_stats['mean_outliers_per_feature']:.2f}</span>
                    </p>
                    <p class="mb-2">
                        <strong>Feature with most outliers:</strong> 
                        <span class="text-blue-400 font-bold">{feature_stats['most_outliers'][0]}</span> 
                        (<span class="text-pink-500 font-bold">{feature_stats['most_outliers'][1]} outliers</span>)
                    </p>
                </div>

                <h3 class="text-xl font-bold mb-4 text-blue-400">Top Feature Correlations with Diagnosis</h3>
                <div class="bg-gray-800 rounded-lg overflow-hidden">
                    <table class="w-full">
                        <tr class="border-b border-gray-700">
                            <th class="bg-gray-700 text-blue-400 p-4 text-left">Feature</th>
                            <th class="bg-gray-700 text-blue-400 p-4 text-left">Correlation</th>
                        </tr>"""
    
    # Add table rows separately
    for feat, corr in target_corrs[1:6].items():
        html_content += f"""
                        <tr class="border-b border-gray-700 hover:bg-gray-700">
                            <td class="p-4">{feat}</td>
                            <td class="p-4 text-blue-400 font-bold">{corr:.3f}</td>
                        </tr>"""
    
    html_content += """
                    </table>
                </div>
            </div>

            <div class="mt-12 pt-6 border-t border-gray-700 text-gray-400 text-center">
                <p>Analysis completed successfully | Generated with Python Data Analysis Tools</p>
            </div>
        </div>
    </body>
    </html>"""
    
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