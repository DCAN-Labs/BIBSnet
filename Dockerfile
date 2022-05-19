ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.11-py3
FROM ${FROM_IMAGE_NAME} 

ENV nnUNet_raw_data_base="/opt/nnUNet/nnUNet_raw_data_base/"
ENV nnUNet_preprocessed="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models"

RUN mkdir -p /opt/nnUNet/nnUNet_raw_data_base/ /opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet /results
COPY trained_models/Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip /results
RUN cd /results && unzip -qq Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip
# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    apt-utils \
                    autoconf \
                    build-essential \
                    bzip2 \
                    ca-certificates \
                    curl \
                    gcc \
                    git \
                    gnupg \
                    libtool \
                    lsb-release \
                    pkg-config \
                    unzip \
                    wget \
                    xvfb \
		    zlib1g && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users bibsnet
WORKDIR /home/bibsnet
ENV HOME="/home/bibsnet" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# install nnUNet git repo
RUN cd .. && \
    git clone https://github.com/MIC-DKFZ/nnUNet.git && \
    cd nnUNet && \
    pip install -e .

# pull down version 1.1.0 of CABINET
RUN cd ../../ && \
    git clone https://github.com/DCAN-Labs/CABINET.git && \
    cd CABINET && \
    git checkout tags/1.1.0 && \
    pip install -r requirements.txt
ENV nnUNet_raw_data_base="/output"
ENV nnUNet_preprocessed="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models"

RUN mkdir -p /opt/nnUNet/nnUNet_raw_data_base/ /opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet
COPY trained_models/Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet
RUN cd /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet && unzip -qq Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip
COPY run.py /home/bibsnet/run.py
RUN cd /home/bibsnet/ && chmod 555 run.py

ENTRYPOINT ["/home/bibsnet/run.py"]
