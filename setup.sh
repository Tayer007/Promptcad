#!/bin/bash
echo "=== PromptCAD Local Setup ==="

if ! command -v python3.10 &>/dev/null; then
    echo "Python 3.10 not found. Please install it first."
    exit 1
fi

echo "Creating virtual environment with Python 3.10..."
python3.10 -m venv cad_env

echo "Activating environment..."
source cad_env/bin/activate

echo "Installing PyTorch (CPU)..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

echo "Installing requirements..."
pip install -r requirements_full.txt

echo ""
echo "Setup complete. To run the app:"
echo "  source cad_env/bin/activate"
echo "  export HF_REPO=fourat25/mesh-to-cadquery-qwen"
echo "  python app.py"
