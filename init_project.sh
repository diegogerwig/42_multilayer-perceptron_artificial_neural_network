#!/bin/bash

remove_venv() {
    echo '🧹 Removing venv'
    rm -rf ~/sgoinfre/mlp_venv
}

create_venv() {
    echo '🧹 Removing old venv'
    rm -rf ~/sgoinfre/mlp_venv
    echo '🔆 Creating new venv'
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

install_dependencies() {
    if [ -f "requirements.txt" ]; then
        echo '🔗 Installing dependencies'
        pip install -r requirements.txt
        echo "✅ Dependencies installed"
    else
        echo "❌ requirements.txt not found"
    fi
}

run_project() {
    if [ -f "./src/split.py" ]; then
        echo -e '\n🔎 EDA Exploratory Data Analysis'
        python ./src/EDA_exploratory_data_analysis.py --dataset ./data/data.csv

        echo -e '\n📂 Splitting dataset'
        python ./src/split.py 

        echo -e '\n💪 Training model'
        python ./src/train.py --early_stopping true

        echo -e '\n🔮 Making predictions'
        python ./src/predict.py 
    else
        echo "❌ File not found"
    fi
}

eval_project() {
    CYCLES=3

    if [ -f "./src/split.py" ]; then
        # Array to store LOSS values
        declare -a loss_values

        GREEN='\033[0;32m'
        NC='\033[0m' # No Color
        
        for i in $(seq 1 $CYCLES); do
            echo -e "\n\n🔄 Evaluation Cycle $i of $CYCLES"
            echo "========================================"
            
            echo -e '\n🎯 Starting Evaluation'
            echo -e '\n📂 Splitting dataset with "evaluation.py" script (random split)'
            python ./src/evaluation.py
            
            echo -e '\n💪 Training Model\n'
            python ./src/train.py --early_stopping true --skip-input

            echo -e '\n🔮 Making Predictions\n'
            python ./src/predict.py | tee temp_output.txt
            
            # Extract LOSS value and store it
            loss=$(grep "LOSS:" temp_output.txt | awk '{print $3}')
            loss_values[$i]=$loss
            
            echo -e "\n✅ Cycle $i completed - LOSS: $loss"
            echo "========================================"
            sleep 3
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
        echo -e "📊 Evaluation Summary:"
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
    
    # Define full paths to commands
    RM="/bin/rm"
    MKDIR="/bin/mkdir"
    TOUCH="/usr/bin/touch"
    FIND="/usr/bin/find"
    WC="/usr/bin/wc"
    TR="/usr/bin/tr"
    
    # Array of files and directories to clean
    declare -a paths=(
        "./data/processed/data_training.csv"
        "./data/processed/data_validation.csv"
        "./data/processed/data_test.csv"
        "./data/processed/predictions.csv"
    )
    
    # Directories to clean while preserving .gitkeep
    declare -a dirs=(
        "./models"
        "./plots"
        "./report"
    )
    
    # Counter for deleted items
    deleted=0
    
    # Iterate through each path
    for path in "${paths[@]}"; do
        if [ -e "$path" ]; then
            echo "Removing: $path"
            $RM -rf "$path"
            
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
            # Count files before cleaning (excluding .gitkeep)
            files_count=$($FIND "$dir" -type f ! -name '.gitkeep' | $WC -l | $TR -d '[:space:]')
            deleted=$((deleted + files_count))
            
            # Remove all contents except .gitkeep
            setopt NULL_GLOB
            $RM -rf "$dir"/.[!.]* "$dir"/*  2>/dev/null
            unsetopt NULL_GLOB
            $TOUCH "$dir/.gitkeep"
        else
            echo "Creating directory: $dir"
            $MKDIR -p "$dir"
            $TOUCH "$dir/.gitkeep"
        fi
    done

    echo "✨ Cleanup complete! Removed $deleted items"
    return 0
}

case "$1" in
    -init)
        remove_venv
        clean_project
        create_venv
        activate_venv
        install_dependencies
        run_project
        ;;
    -run)
        clean_project
        activate_venv
        install_dependencies
        run_project
        ;;
    -eval)
        clean_project
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
        echo "Usage: source $0 [-option]"
        echo "Options:"
        echo "  -init     : Create venv, activate venv, install dependencies and run project"
        echo "  -run      : Activate venv and run project"
        echo "  -eval     : Activate venv and run evaluation"
        echo "  -clean    : Remove unnecessary files and directories"
        echo "  -remove   : Remove venv and unnecessary files and directories"
        return 1
        ;;
esac