FROM tensorflow/tensorflow:2.4.1-gpu
WORKDIR /app

# Install packages
RUN apt-get update && apt-get install -y sudo libxml2 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python3 has already installed
# We need to install necessary packages
RUN python -m pip install --upgrade pip \
    && pip install numpy typer pydicom

#
