#!/bin/bash

remove_venv() {
    echo '🧹 Removing venv'
    rm -rf ~/sgoinfre/mlp_venv
}

create_venv() {
    echo '🧹 Removing venv'
    rm -rf ~/sgoinfre/mlp_venv
    echo '🔆 Creating venv'
    python3 -m venv ~/sgoinfre/mlp_venv
}

activate_venv() {
    source ~/sgoinfre/mlp_venv/bin/activate
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo -e '\n✅ Virtual environment activated'
        echo "💻 Python location: $(which python) | Version: $(python --version 2>&1)"
    else
        echo '❌ Failed to activate virtual environment.'
    fi
}

install_dependencies () {
    if [ -f "requirements.txt" ]; then
        echo '🔗 Installing dependencies'
        pip install -r requirements.txt
        echo "✅ Dependencies installed"
    else
        echo "❌ requirements.txt not found"
    fi
}

grid_search() {
    if [ -f "./src/hyperparam.py" ]; then
        echo -e '\n🔍 Grid search'
        python ./src/hyperparam.py --data ./data/data.csv
    else
        echo "❌ File not found"
    fi
}

run_project() {
    if [ -f "./src/split.py" ]; then
        # echo -e '\n🔎 EDA Exploratory Data Analysis'
        # python ./src/EDA_exploratory_data_analysis.py --dataset ./data/data.csv

        echo -e '\n📂 Split dataset'
        python ./src/split.py --dataset ./data/data.csv

        echo -e '\n💪 Trainig'
        python ./src/train.py --train_data ./data/data_training.csv --val_data ./data/data_validation.csv
        # python ./src/train.py --train_data ./data/data_training.csv --val_data ./data/data_validation.csv --layer 16 8 4 --learning_rate 0.01

        echo -e '\n🔮 Predict'
        python ./src/predict.py --test_data ./data/data_test.csv

    else
        echo "❌ File not found"
    fi
}

eval_project() {
    CYCLES=10

    if [ -f "./src/split.py" ]; then
        # Array to store LOSS values
        declare -a loss_values

        GREEN='\033[0;32m'
        NC='\033[0m' # No Color
        
        for i in $(seq 1 $CYCLES); do
            echo -e "\n\n🔄 Evaluation Cycle $i of $CYCLES"
            echo "========================================"
            
            echo -e '\n🎯 Evaluation'
            cd ./src
            echo -e '\n📂 Split dataset with "evaluation.py" script (random split).'
            python ./evaluation.py
            cd ..
            
            echo -e '\n💪 Training'
            python ./src/train.py --train_data ./data/data_training.csv
            
            echo -e '\n🔮 Predict'
            python ./src/predict.py --test_data ./data/data_test.csv | tee temp_output.txt
            
            # Extract LOSS value and store it
            loss=$(grep "LOSS:" temp_output.txt | awk '{print $2}')
            loss_values[$i]=$loss
            
            echo -e "\n✅ Cycle $i completed - LOSS: $loss"
            echo "========================================"
            sleep 1
        done

        # Find minimum LOSS value
        min_loss=${loss_values[1]}
        for loss in "${loss_values[@]}"; do
            if (( $(echo "$loss < $min_loss" | bc -l) )); then
                min_loss=$loss
            fi
        done
        
        # Display summary of all cycles
        echo -e "\n\n"
        echo "========================================"
        echo -e "📊 Summary of evaluation cycles:"
        echo "========================================"
        for i in $(seq 1 $CYCLES); do
            if (( $(echo "${loss_values[$i]} == $min_loss" | bc -l) )); then
                echo -e "Cycle $i \tLOSS: ${GREEN}${loss_values[$i]}${NC} (minimum)"
            else
                echo -e "Cycle $i \tLOSS: ${loss_values[$i]}"
            fi
        done
        
        # Clean up temporary file
        rm temp_output.txt
        
    else
        echo "❌ File not found"
    fi
}

clean_project() {
    echo '🧹 Starting project cleanup...'
    
    # Array of files and directories to clean
    declare -a paths=(
        "./data/data_training.csv"
        "./data/data_validation.csv"
        "./data/data_test.csv"
        "./data/predictions.csv"
    )
    
    # Directories to clean while preserving .gitkeep
    declare -a dirs=(
        "./src/logs"
        "./src/models"
        "./src/plots"
        "./src/reports"
        "./src/results"
    )
    
    # Counter for deleted items
    deleted=0
    
    # Iterate through each path
    for path in "${paths[@]}"; do
        if [ -e "$path" ]; then
            echo "Removing: $path"
            rm -rf "$path"
            
            # Check if removal was successful
            if [ $? -eq 0 ]; then
                ((deleted++))
            else
                echo "❌ Error: Failed to remove $path"
                return 1
            fi
        else
            echo "Skipping: $path (not found)"
        fi
    done
    
    # Clean directories while preserving .gitkeep
    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
            echo "Cleaning directory: $dir"
            # Find and remove all files except .gitkeep
            find "$dir" -type f ! -name '.gitkeep' -delete
            find "$dir" -type d ! -path "$dir" -delete
            
            # Create .gitkeep if it doesn't exist
            touch "$dir/.gitkeep"
            
            ((deleted++))
        else
            echo "Skipping: $dir (not found)"
            mkdir -p "$dir"
            touch "$dir/.gitkeep"
        fi
    done

    echo "✨ Cleanup complete! Removed $deleted items"
    return 0
}

case "$1" in
    -init)
        create_venv
        activate_venv
        install_dependencies
        run_project
        ;;
    -grid)
        activate_venv
        install_dependencies
        grid_search
        ;;
    -run)
        activate_venv
        install_dependencies
        run_project
        ;;
    -eval)
        activate_venv
        install_dependencies
        eval_project
        ;;
    -clean)
        clean_project
        ;;
    -remove)
        remove_venv
        clean_project
        ;;
    *)
        echo "❌ Invalid argument: $1"
        echo "Usage: source $0 [-up]"
        echo "  -init     : Create venv, activate venv, install dependencies and run project"
        echo "  -grid     : Activate venv and run grid search"
        echo "  -run      : Activate venv and run project"
        echo "  -eval     : Activate venv and run evaluation"
        echo "  -clean    : Remove unnecessary files and directories"
        echo "  -remove   : Remove venv and unnecessary files and directories"
        return 1
        ;;
esac