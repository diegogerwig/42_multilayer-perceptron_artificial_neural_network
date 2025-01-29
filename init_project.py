import os
import sys
import shutil
import subprocess
import time
from pathlib import Path

class ProjectManager:
    def __init__(self):
        self.home = os.environ['HOME']
        self.conda_path = f"{self.home}/sgoinfre/miniforge"
        self.conda_bin = f"{self.conda_path}/bin/conda"
        self.env_name = "mlp_env"
        self.env_path = f"{self.conda_path}/envs/{self.env_name}"

    def run_command(self, command, shell=False, capture_output=False):
        try:
            if shell:
                if capture_output:
                    return subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
                return subprocess.run(command, shell=True, check=True)
            else:
                if capture_output:
                    return subprocess.run(command.split(), check=True, capture_output=True, text=True)
                return subprocess.run(command.split(), check=True)
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Error executing: {command}")
            return False

    def remove_venv(self):
        print('\n🧹 Removing conda environment')
        self.run_command(f"{self.conda_bin} env remove -n {self.env_name} -y", shell=True)
        if os.path.exists(self.env_path):
            shutil.rmtree(self.env_path, ignore_errors=True)

    def create_venv(self):
        print('\n🔆 Creating new conda environment')
        return self.run_command(f"{self.conda_bin} create -y -n {self.env_name} python=3.11")

    def activate_venv(self):
        print('\n💿 Activating conda environment')
        activate_command = f". {self.conda_path}/etc/profile.d/conda.sh && conda activate {self.env_name}"
        return self.run_command(activate_command, shell=True)

    def install_dependencies(self):
        print('\n🔗 Installing dependencies')
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
            # (f"{python_bin} ./src/EDA_exploratory_data_analysis.py", '🔎 EDA Exploratory Data Analysis'),
            (f"{python_bin} ./src/split.py", '📂 Splitting dataset\n'),
            (f"{python_bin} ./src/train.py --early_stopping false", '💪 Training model\n'),
            (f"{python_bin} ./src/predict.py", '🔮 Making predictions\n')
        ]

        for cmd, msg in commands:
            print(f"\n{msg}")
            if not self.run_command(cmd):
                return False
        return True

    def eval_project(self):
        if not os.path.exists("./src/split.py"):
            print("❌ File not found")
            return False

        python_bin = f"{self.env_path}/bin/python"
        cycles = 5
        loss_values = []
        accuracy_values = []

        for i in range(1, cycles + 1):
            print(f"\n\n🔄 Evaluation Cycle {i} of {cycles}")
            print("=" * 40)
            
            print('\n🎯 Starting Evaluation')
            print('\n📂 Splitting dataset with "evaluation.py" script (random split)')
            self.run_command(f"{python_bin} ./src/evaluation.py")
            
            print('\n💪 Training Model\n')
            self.run_command(f"{python_bin} ./src/train.py --early_stopping true --skip-input")
            
            print('\n\n🔮 Making Predictions\n')
            result = self.run_command(f"{python_bin} ./src/predict.py --skip-input", capture_output=True)
            
            if result:
                output_lines = result.stdout.split('\n')
                loss = None
                accuracy = None
                
                for line in output_lines:
                    if "LOSS:" in line:
                        loss = float(line.split()[2])
                    if "Accuracy:" in line:
                        accuracy = float(line.split()[2])
                
                if loss is not None:
                    loss_values.append(loss)
                if accuracy is not None:
                    accuracy_values.append(accuracy)
                
                print(f"\n✅ Cycle {i} completed - LOSS: {loss}")
                print("=" * 40)
                time.sleep(2)

        # Print evaluation summary
        print("\n\n")
        print("=" * 40)
        print("📊 Evaluation Summary:")
        print("=" * 40)

        min_loss = min(loss_values)
        for i, (loss, accuracy) in enumerate(zip(loss_values, accuracy_values), 1):
            if loss == min_loss:
                print(f"Cycle {i} -> LOSS: \033[32m{loss}\033[0m & ACCURACY: {accuracy*100:.4f}% (🏆 BEST CYCLE)")
            else:
                print(f"Cycle {i} -> LOSS: {loss} & ACCURACY: {accuracy*100:.4f}%")

        return True

    def visualize_project(self):
        print('\n📂 Visualize')
        python_bin = f"{self.env_path}/bin/python"
        return self.run_command(f"{python_bin} ./src/utils/visualizer.py")

    def clean_project(self):
        print('\n🧹 Starting project cleanup...')
        
        total_deleted = 0
        
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
        files_deleted = 0
        print("\nCleaning individual files:")
        print("=" * 40)
        for file in files:
            if os.path.exists(file):
                os.remove(file)
                print(f"✓ Removed: {file}")
                files_deleted += 1
            else:
                print(f"✕ Skipping: {file} (not found)")
        total_deleted += files_deleted

        # Clean directories
        print("\nCleaning directories:")
        print("=" * 40)
        for dir_path in dirs:
            dir_files_deleted = 0
            if os.path.exists(dir_path):
                print(f"\nProcessing directory: {dir_path}")
                for item in os.listdir(dir_path):
                    if item != '.gitkeep':
                        item_path = os.path.join(dir_path, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                            print(f"✓ Removed file: {item}")
                            dir_files_deleted += 1
                        elif os.path.isdir(item_path):
                            items_in_subdir = sum(len(files) for _, _, files in os.walk(item_path))
                            shutil.rmtree(item_path)
                            print(f"✓ Removed directory: {item} (containing {items_in_subdir} files)")
                            dir_files_deleted += items_in_subdir
                if dir_files_deleted == 0:
                    print("📂 Directory was already clean")
                else:
                    print(f"📊 {dir_files_deleted} items removed from {dir_path}")
            else:
                os.makedirs(dir_path, exist_ok=True)
                Path(os.path.join(dir_path, '.gitkeep')).touch()
                print(f"📁 Created directory: {dir_path}")
            total_deleted += dir_files_deleted

        # Print summary
        print("\n✨ Cleanup Summary:")
        print("=" * 40)
        print(f"🗑️  Total items removed: {total_deleted}")
        print(f"📄 Individual files removed: {files_deleted}")
        print(f"📁 Directories processed: {len(dirs)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python init_project.py [-option]")
        print("Options:")
        print("  -init     : Create env & activate env & install dependencies & run project")
        print("  -run      : Activate env & run project")
        print("  -eval     : Activate env & run evaluation")
        print("  -visualize: Activate env & visualize network")
        print("  -clean    : Remove unnecessary files and directories")
        print("  -remove   : Remove env & Remove unnecessary files and directories")
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
        pm.visualize_project()
    elif option == "-eval":
        pm.clean_project()
        if not pm.activate_venv():
            return
        if not pm.install_dependencies():
            return
        pm.eval_project()
    elif option == "-visualize":
        if not pm.activate_venv():
            return
        if not pm.install_dependencies():
            return
        pm.visualize_project()
    elif option == "-clean":
        pm.clean_project()
    elif option == "-remove":
        pm.remove_venv()
        pm.clean_project()
    else:
        print(f"❌ Invalid option: {option}")

if __name__ == "__main__":
    main()