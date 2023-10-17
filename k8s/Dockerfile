FROM python:3.10.6-slim

WORKDIR /root

RUN echo "alias ll='ls -al'" >> ~/.bashrc && \
    apt-get update && \
    apt-get install -y vim && \
    pip install kubernetes==26.1.0

COPY pods pods
COPY deploy.py deploy.py
COPY command.txt command.txt
COPY create_volume.py create_volume.py
