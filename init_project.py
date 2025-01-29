#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
from pathlib import Path

class ProjectManager:
    def __init__(self):
        self.home = os.environ['HOME']
        self.conda_path = f"{self.home}/sgoinfre/miniforge"
        self.conda_bin = f"{self.conda_path}/bin/conda"
        self.env_name = "mlp_env"
        self.env_path = f"{self.conda_path}/envs/{self.env_name}"

    def run_command(self, command, shell=False):
        try:
            if shell:
                subprocess.run(command, shell=True, check=True)
            else:
                subprocess.run(command.split(), check=True)
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Error executing: {command}")
            return False

    def remove_venv(self):
        print('🧹 Removing conda environment')
        self.run_command(f"{self.conda_bin} env remove -n {self.env_name} -y", shell=True)
        if os.path.exists(self.env_path):
            shutil.rmtree(self.env_path, ignore_errors=True)

    def create_venv(self):
        print('🔆 Creating new conda environment')
        return self.run_command(f"{self.conda_bin} create -y -n {self.env_name} python=3.11")

    def activate_venv(self):
        print('Activating conda environment')
        activate_command = f". {self.conda_path}/etc/profile.d/conda.sh && conda activate {self.env_name}"
        return self.run_command(activate_command, shell=True)

    def install_dependencies(self):
        print('🔗 Installing dependencies')
        if not os.path.exists('requirements.txt'):
            print("❌ requirements.txt not found")
            return False
        return self.run_command(f"{self.env_path}/bin/pip install -r requirements.txt")

    def run_project(self):
        if not os.path.exists("./src/split.py"):
            print("❌ File not found")
            return False

        python_bin = f"{self.env_path}/bin/python"
        commands = [
            (f"{python_bin} ./src/split.py", '📂 Splitting dataset'),
            (f"{python_bin} ./src/train.py --early_stopping false", '💪 Training model'),
            (f"{python_bin} ./src/predict.py", '🔮 Making predictions')
        ]

        for cmd, msg in commands:
            print(f"\n{msg}")
            if not self.run_command(cmd):
                return False
        return True

    def clean_project(self):
        print('🧹 Starting project cleanup...')
        
        # Files to clean
        files = [
            "./data/processed/data_training.csv",
            "./data/processed/data_validation.csv",
            "./data/processed/data_test.csv",
            "./data/processed/predictions.csv"
        ]
        
        # Directories to clean
        dirs = ["./models", "./plots", "./report"]
        
        # Clean files
        for file in files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed: {file}")
            else:
                print(f"Skipping: {file} (not found)")

        # Clean directories
        for dir_path in dirs:
            if os.path.exists(dir_path):
                for item in os.listdir(dir_path):
                    if item != '.gitkeep':
                        item_path = os.path.join(dir_path, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
            else:
                os.makedirs(dir_path, exist_ok=True)
                Path(os.path.join(dir_path, '.gitkeep')).touch()

def main():
    if len(sys.argv) != 2:
        print("Usage: python init_project.py [-option]")
        print("Options:")
        print("  -init     : Create env, activate env, install dependencies and run project")
        print("  -run      : Activate env and run project")
        print("  -clean    : Remove unnecessary files and directories")
        print("  -remove   : Remove env and unnecessary files and directories")
        return

    pm = ProjectManager()
    option = sys.argv[1]

    if option == "-init":
        pm.remove_venv()
        pm.clean_project()
        if not pm.create_venv():
            return
        if not pm.activate_venv():
            return
        if not pm.install_dependencies():
            return
        pm.run_project()
    elif option == "-run":
        pm.clean_project()
        if not pm.activate_venv():
            return
        if not pm.install_dependencies():
            return
        pm.run_project()
    elif option == "-clean":
        pm.clean_project()
    elif option == "-remove":
        pm.remove_venv()
        pm.clean_project()
    else:
        print(f"❌ Invalid option: {option}")

if __name__ == "__main__":
    main()