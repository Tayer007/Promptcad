#!/bin/bash
B=$HOME/.cache/huggingface
UP=$HOME/promptcad/uploads
RE=$HOME/promptcad/results
mkdir -p $UP $RE
singularity exec --nv --contain \
  --bind /tmp \
  --bind $B:/root/.cache/huggingface \
  --bind $UP:/app/uploads \
  --bind $RE:/app/results \
  --bind $PWD/app.py:/app/app.py \
  --env HF_HOME=/root/.cache/huggingface \
  promptcad.sif /opt/venv/bin/python /app/app.py
