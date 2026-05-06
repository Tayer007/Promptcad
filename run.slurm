#!/bin/bash
#SBATCH --job-name=promptcad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=promptcad_%j.log

# Write the compute node hostname so you know where to tunnel
echo "Node: $(hostname)" > promptcad_node.txt
echo "Connect with: ssh -L 5000:$(hostname):5000 <your-hpc-login>" >> promptcad_node.txt
cat promptcad_node.txt

# Point HuggingFace cache to your HPC home so model is cached after first run
export HF_HOME=$HOME/.cache/huggingface

# Run the container with GPU support
singularity exec --nv \
    --bind $HOME/.cache/huggingface:/root/.cache/huggingface \
    promptcad.sif \
    python3 /app/app.py
