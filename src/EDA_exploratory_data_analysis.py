import pandas as pd
from ydata_profiling import ProfileReport
import os
import webbrowser
import platform
import subprocess
import time

def load_data():
    with open('./data/data_colums_names.txt', 'r') as f:
        column_names = [line.strip() for line in f.readlines()]
    return pd.read_csv('./data/data.csv', names=column_names)

def generate_profile_report(df):
    os.makedirs('./reports', exist_ok=True)
    report_path = './reports/profile_report.html'
    
    profile = ProfileReport(df, title="Data Profile Report")
    profile.to_file(report_path)
    return report_path

def open_html_report(report_path):
    abs_path = os.path.abspath(report_path)
    
    if platform.system().lower() == 'linux':
        if 'microsoft' in platform.uname().release.lower():
            try:
                windows_path = subprocess.check_output(['wslpath', '-w', abs_path], 
                                                     stderr=subprocess.PIPE).decode().strip()
                subprocess.Popen(['cmd.exe', '/c', 'start', windows_path])
            except subprocess.CalledProcessError:
                subprocess.Popen(['xdg-open', abs_path])
        else:
            subprocess.Popen(['xdg-open', abs_path])
    else:
        webbrowser.open('file://' + abs_path)
    
    time.sleep(2)

def main():
    try:
        df = load_data()
        print("Data loaded successfully")

        report_path = generate_profile_report(df)
        print(f"Report generated: {report_path}")
        
        open_html_report(report_path)
        print("Opening report in browser...")
        time.sleep(3)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'report_path' in locals():
            print(f"Report location: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    main()