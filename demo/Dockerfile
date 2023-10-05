FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

SHELL ["/bin/bash", "-c"]

USER root

WORKDIR /root

ENV DEBIAN_FRONTEND=noninteractive

# Set env for torch (compute capability)
ENV TORCH_CUDA_ARCH_LIST=9.0

# Set env for huggingface offline mode
ENV HF_DATASETS_OFFLINE=1

# Install packages
RUN apt-get update && \
    apt-get install -y pdsh vim openssh-server net-tools libnvidia-compute-515 && \
    mkdir -p /var/run/sshd

# Set for ssh
RUN mkdir .ssh
COPY key.pem .ssh/key.pem
COPY authorized_keys .ssh/authorized_keys

# Install python & pip and Install libraries
COPY requirements.txt requirements.txt
RUN apt-get install -y curl python3.10-dev python3.10-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip install --no-cache-dir -r requirements.txt

# Copy files that required for training
COPY run.sh run.sh
COPY train.py train.py
COPY config.yaml config.yaml
