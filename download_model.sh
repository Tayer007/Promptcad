#!/bin/bash
# Run on the HPC login node (has internet access).
# Downloads the model into $HOME/.cache/huggingface so the compute node can use it.
B=$HOME/.cache/huggingface
mkdir -p $B
singularity exec \
  --bind $B:/root/.cache/huggingface \
  --env HF_HOME=/root/.cache/huggingface \
  --env HF_HUB_OFFLINE=0 \
  promptcad.sif /opt/venv/bin/python download_model.py
