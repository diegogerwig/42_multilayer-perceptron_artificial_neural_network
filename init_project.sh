#!/bin/sh

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
        echo -e '⭐ Virtual environment ready\n'
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
    if [ -f "./src/grid_search.py" ]; then
        echo -e '\n🔍 Grid search'
        python ./src/grid_search.py --train_data ./data/data_training.csv
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
        # python ./src/train.py --train_data ./data/data_training.csv --test_data ./data/data_text.csv --layers 16 8 4 --learning_rate 0.001

        echo -e '\n🔮 Predict'
        python ./src/predict.py --test_data ./data/data_test.csv

    else
        echo "❌ File not found"
    fi
}

eval_project() {
    if [ -f "./src/split.py" ]; then
        echo -e '\n🎯 Evaluation'
        cd ./src

        echo -e '\n📂 Split dataset with "evaluation.py" script (random split).'
        python ./evaluation.py
        cd ..

        echo -e '\n💪 Trainig'
        python ./src/train.py --train_data ./data/data_training.csv
        
        echo -e '\n🔮 Predict'
        python ./src/predict.py --test_data ./data/data_test.csv

    else
        echo "❌ File not found"
    fi
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
        pip install scikeras
        pip install keras
        pip install scikit-learn
        pip install tensorflow
        grid_search
        ;;
    -up)
        activate_venv
        run_project
        ;;
    -eval)
        activate_venv
        eval_project
        ;;
    -clean)
        remove_venv
        ;;
    *)
        echo "❌ Invalid argument: $1"
        echo "Usage: source $0 [-up]"
        echo "  -init   : Create venv, activate venv, install dependencies and run project"
        echo "  -grid   : Activate venv and run grid search"
        echo "  -up     : Activate venv and run project"
        echo "  -eval   : Activate venv and run evaluation"
        echo "  -clean  : Remove venv"
        return 1
        ;;
esac