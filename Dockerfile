FROM nvcr.io/nvidia/tensorflow:20.10-tf2-py3
WORKDIR /app

# Install packages
RUN apt-get update && apt-get install -y sudo libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python3 has already installed
# We need to install necessary packages
RUN python -m pip install --upgrade pip \
    && pip install typer

#
