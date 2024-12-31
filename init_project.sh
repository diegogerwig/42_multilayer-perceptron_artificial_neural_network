#!/bin/sh

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
        pip install -r requirements.txt >/dev/null 2>&1
        echo "✅ Dependencies installed"
    else
        echo "❌ requirements.txt not found"
    fi
}

run_project() {
    if [ -f "./src/mlp.py" ]; then
        echo -e '\n📂 Split dataset'
        python ./src/split.py --dataset ./data/data.csv
    else
        echo "❌ mlp.py not found"
    fi
}

case "$1" in
    -up)
        activate_venv
        run_project
        ;;
    "")
        create_venv
        activate_venv
        install_dependencies
        run_project
        ;;
    *)
        echo "❌ Invalid argument: $1"
        echo "Usage: source $0 [-up]"
        echo "  no args : Full initialization"
        echo "  -up     : Just activate venv and run project"
        return 1
        ;;
esac