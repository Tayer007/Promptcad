FROM nvidia/cuda:12.8.2-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_REPO=fourat25/mesh-to-cadquery-qwen
ENV HF_HUB_OFFLINE=1
ENV PATH="/opt/venv/bin:$PATH"

# System dependencies (required for CadQuery/OpenCASCADE and trimesh)
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 libgomp1 \
    libxrender1 libxext6 libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv at /opt/venv (won't be overridden by HPC host mounts)
RUN python3 -m venv /opt/venv

WORKDIR /app

# Install Python packages into the venv
COPY requirements_full.txt .
RUN /opt/venv/bin/pip install --no-cache-dir \
    torch==2.5.1+cu124 torchaudio==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements_full.txt

# Copy application files
COPY app.py enhancer.py inference.py smart_scaler.py ./
COPY static/ ./static/

RUN mkdir -p uploads results

EXPOSE 5000

CMD ["/opt/venv/bin/python", "app.py"]
