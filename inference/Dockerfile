FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

USER root

WORKDIR /root

ENV DEBIAN_FRONTEND=noninteractive

# Set env for torch (compute capability)
ENV TORCH_CUDA_ARCH_LIST=9.0

# Install python & pip and Install libraries
COPY requirements.txt requirements.txt
RUN apt-get update && \
    apt-get install -y curl python3.10-distutils git && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip install --no-cache-dir -r requirements.txt

# Copy files that required for training
COPY main.py main.py
COPY models.py models.py
