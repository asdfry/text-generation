FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

USER root

WORKDIR /root

ENV DEBIAN_FRONTEND=noninteractive

# Set env for torch (compute capability)
ENV TORCH_CUDA_ARCH_LIST=9.0

# Set env for huggingface offline mode
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Set locale and Install packages
RUN apt-get update && \
    apt-get install -y pdsh vim openssh-server git collectl && \
    mkdir -p /var/run/sshd

# Set for ssh
RUN mkdir .ssh
COPY key.pem .ssh/key.pem
COPY authorized_keys .ssh/authorized_keys

# Install python & pip and Install libraries
COPY requirements.txt requirements.txt
RUN apt-get install -y curl python3.10-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip install --no-cache-dir -r requirements.txt

# Copy files that required for training
RUN mkdir logs
COPY tldr_news tldr_news
COPY create_config.py create_config.py
COPY train_multi.py train_multi.py
