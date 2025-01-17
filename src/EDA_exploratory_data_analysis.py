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

def load_data():
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

def generate_html_report(df, outliers, high_corr_sorted, feature_stats, target_corrs):
    from datetime import datetime
    
    os.makedirs('./report', exist_ok=True)
    
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
            .correlation-item {{ padding: 8px; margin: 4px 0; background-color: #f8f9fa; border-radius: 4px; }}
            pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; }}
            .highlight {{ background-color: #fff3cd; padding: 15px; border-radius: 4px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Data Analysis Summary Report</h1>
            <p>Generated on: {current_time}</p>
            
            <h2>1. Dataset Overview</h2>
            <table class="stats-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total samples</td>
                    <td>{len(df)}</td>
                </tr>
                <tr>
                    <td>Features</td>
                    <td>{len(df.columns)}</td>
                </tr>
                <tr>
                    <td>Missing values</td>
                    <td>{df.isnull().sum().sum()}</td>
                </tr>
                <tr>
                    <td>Data balance ratio (Benign/Malignant)</td>
                    <td>{(df['Diagnosis'] == 0).sum() / (df['Diagnosis'] == 1).sum():.2f}</td>
                </tr>
            </table>

            <h2>2. Class Distribution</h2>
            <table class="stats-table">
                <tr>
                    <th>Class</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                <tr>
                    <td>0 (Benign)</td>
                    <td>{class_dist[0]}</td>
                    <td>{(class_dist[0]/len(df)*100):.1f}%</td>
                </tr>
                <tr>
                    <td>1 (Malignant)</td>
                    <td>{class_dist[1]}</td>
                    <td>{(class_dist[1]/len(df)*100):.1f}%</td>
                </tr>
            </table>

            <h2>3. Feature Statistics</h2>
            <pre>{df.describe().to_string()}</pre>

            <h2>4. Data Quality Analysis</h2>
            <h3>Outliers Summary</h3>
            <div class="highlight">
                <p>Total outliers in dataset: {feature_stats['total_outliers']}</p>
                <p>Average outliers per feature: {feature_stats['mean_outliers_per_feature']:.2f}</p>
                <p>Feature with most outliers: {feature_stats['most_outliers'][0]} ({feature_stats['most_outliers'][1]} outliers)</p>
            </div>

            <h3>Detailed Outliers Count</h3>
            <table class="stats-table">
                <tr>
                    <th>Feature</th>
                    <th>Number of Outliers</th>
                </tr>
                {''.join(f'<tr><td>{col}</td><td>{count}</td></tr>' for col, count in sorted(outliers.items(), key=lambda x: x[1], reverse=True))}
            </table>

            <h2>5. Feature Correlations</h2>
            <h3>Top Correlations with Target (Diagnosis)</h3>
            <table class="stats-table">
                <tr>
                    <th>Feature</th>
                    <th>Correlation with Diagnosis</th>
                </tr>
                {''.join(f'<tr><td>{feat}</td><td>{corr:.3f}</td></tr>' for feat, corr in target_corrs[1:6].items())}
            </table>

            <h3>Highly Correlated Feature Pairs</h3>
            <table class="stats-table">
                <tr>
                    <th>Feature Pair</th>
                    <th>Correlation</th>
                </tr>
                {''.join(f'<tr><td>{feat1} - {feat2}</td><td>{corr:.3f}</td></tr>' for feat1, feat2, corr in high_corr_sorted)}
            </table>

            <h2>6. Feature Importance</h2>
            <table class="stats-table">
                <tr>
                    <th>Feature</th>
                    <th>Absolute Correlation with Target</th>
                </tr>
                {''.join(f'<tr><td>{feat}</td><td>{abs(corr):.3f}</td></tr>' 
                        for feat, corr in target_corrs[1:11].items())}
            </table>
        </div>
    </body>
    </html>
    """
    
    with open('./report/summary_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

def generate_summary_report(df, outliers, high_corr):
    from datetime import datetime
    
    # Sort high correlations by absolute correlation value
    high_corr_sorted = sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)
    
    # Calculate basic statistics for features
    feature_stats = {
        'mean_outliers_per_feature': sum(outliers.values()) / len(outliers),
        'most_outliers': max(outliers.items(), key=lambda x: x[1]),
        'total_outliers': sum(outliers.values())
    }
    
    os.makedirs('./report', exist_ok=True)
    with open('./report/summary_report.txt', 'w') as f:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Data Analysis Summary Report\n")
        f.write(f"Generated on: {current_time}\n")
        f.write("============================\n\n")
        
        f.write("1. Dataset Overview\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Features: {len(df.columns)}\n")
        f.write(f"Missing values: {df.isnull().sum().sum()}\n")
        f.write(f"Data balance ratio (Benign/Malignant): {(df['Diagnosis'] == 0).sum() / (df['Diagnosis'] == 1).sum():.2f}\n\n")
        
        f.write("2. Class Distribution\n")
        class_dist = df['Diagnosis'].value_counts()
        f.write("Diagnosis values:\n")
        f.write("0 (Benign)  : " + str(class_dist[0]) + f" ({(class_dist[0]/len(df)*100):.1f}%)\n")
        f.write("1 (Malignant): " + str(class_dist[1]) + f" ({(class_dist[1]/len(df)*100):.1f}%)\n\n")
        
        f.write("3. Feature Statistics\n")
        f.write(str(df.describe()) + "\n\n")
        
        f.write("4. Data Quality Analysis\n")
        f.write("a) Outliers Summary:\n")
        f.write(f"Total outliers in dataset: {feature_stats['total_outliers']}\n")
        f.write(f"Average outliers per feature: {feature_stats['mean_outliers_per_feature']:.2f}\n")
        f.write(f"Feature with most outliers: {feature_stats['most_outliers'][0]} ({feature_stats['most_outliers'][1]} outliers)\n\n")
        
        f.write("b) Detailed Outliers Count:\n")
        sorted_outliers = sorted(outliers.items(), key=lambda x: x[1], reverse=True)
        for col, count in sorted_outliers:
            f.write(f"{col}: {count}\n")
        f.write("\n")
        
        f.write("5. Feature Correlations\n")
        f.write("a) Highly Correlated Features (sorted by correlation strength):\n")
        for feat1, feat2, corr in high_corr_sorted:
            f.write(f"{feat1} - {feat2}: {corr:.3f}\n")
        f.write("\n")
        
        f.write("b) Strong Correlations with Target (Diagnosis):\n")
        target_corrs = df.corr()['Diagnosis'].sort_values(key=abs, ascending=False)
        for feat, corr in target_corrs[1:6].items():  # Skip first as it's Diagnosis itself
            f.write(f"{feat}: {corr:.3f}\n")
        f.write("\n")
        
        f.write("6. Statistical Tests\n")
        # Perform Shapiro-Wilk test for normality on numerical features
        from scipy import stats
        f.write("Normality Test Results (Shapiro-Wilk, p-values):\n")
        for column in df.select_dtypes(include=['float64']).columns:
            _, p_value = stats.shapiro(df[column])
            f.write(f"{column}: {p_value:.4f}\n")

    # Generate HTML report
    generate_html_report(df, outliers, high_corr_sorted, feature_stats, target_corrs)

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
    profile.to_file('./report/profile_report.html')

def open_reports():
    """
    Opens generated reports in the default browser.
    Handles different environments: native Linux and WSL.
    Silences Gtk warnings and browser messages.
    """


    # Redirect stderr and stdout to devnull to silence all messages
    devnull = open(os.devnull, 'w')

    reports = [
        'summary_report.html',
        'profile_report.html'
    ]
    
    def is_wsl():
        """Check if running under Windows Subsystem for Linux"""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except:
            return False

    for report in reports:
        report_path = os.path.abspath(f'./report/{report}')
        
        if not Path(report_path).is_file():
            print(f"Warning: Report {report} not found at {report_path}")
            continue

        try:
            if platform.system().lower() == 'linux':
                if is_wsl():
                    print(f"Opening {report} in WSL environment...")
                    try:
                        windows_path = subprocess.check_output(['wslpath', '-w', report_path]).decode().strip()
                        with redirect_stdout(devnull), redirect_stderr(devnull):
                            subprocess.run(['cmd.exe', '/c', 'start', windows_path], 
                                        check=True, 
                                        stderr=subprocess.DEVNULL, 
                                        stdout=subprocess.DEVNULL)
                    except subprocess.CalledProcessError as e:
                        print("WSL path conversion failed, trying xdg-open as fallback...")
                        with redirect_stdout(devnull), redirect_stderr(devnull):
                            subprocess.run(['xdg-open', report_path], 
                                        check=True, 
                                        stderr=subprocess.DEVNULL, 
                                        stdout=subprocess.DEVNULL)
                else:
                    print(f"Opening {report} in native Linux...")
                    try:
                        with redirect_stdout(devnull), redirect_stderr(devnull):
                            subprocess.run(['xdg-open', report_path], 
                                        check=True, 
                                        stderr=subprocess.DEVNULL, 
                                        stdout=subprocess.DEVNULL)
                    except subprocess.CalledProcessError:
                        try:
                            with redirect_stdout(devnull), redirect_stderr(devnull):
                                subprocess.run(['google-chrome', report_path], 
                                            check=True, 
                                            stderr=subprocess.DEVNULL, 
                                            stdout=subprocess.DEVNULL)
                        except subprocess.CalledProcessError:
                            try:
                                with redirect_stdout(devnull), redirect_stderr(devnull):
                                    subprocess.run(['firefox', report_path], 
                                                check=True, 
                                                stderr=subprocess.DEVNULL, 
                                                stdout=subprocess.DEVNULL)
                            except subprocess.CalledProcessError:
                                print(f"Could not open {report}. Please open it manually at: {report_path}")
            else:
                print(f"Opening {report} using default browser...")
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    webbrowser.open('file://' + report_path)
                
        except Exception as e:
            print(f"Error opening {report}: {str(e)}")
            print(f"You can find the report at: {report_path}")

    # Close devnull
    devnull.close()

def main():
    try:
        df = load_data()
        print("✅ Data loaded successfully")
        
        perform_eda(df)
        print("\n💫 EDA completed. Reports generated in ./report directory")
        
        # Small pause to ensure files are completely written
        import time
        time.sleep(1)
        
        open_reports()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()