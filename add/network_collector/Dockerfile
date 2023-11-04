FROM python:3.10.6-slim

WORKDIR /root

COPY requirements.txt requirements.txt

RUN apt-get update && \
    apt-get install -y curl && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip install --no-cache-dir -r requirements.txt

RUN echo "alias ll='ls -al'" >> ~/.bashrc

COPY collector.py collector.py
COPY run.sh run.sh
