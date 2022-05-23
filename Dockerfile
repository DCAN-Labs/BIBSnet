ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.11-py3
FROM ${FROM_IMAGE_NAME} 

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
RUN useradd -m -s /bin/bash -G users cabinet
WORKDIR /home/cabinet
ENV HOME="/home/cabinet" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# install nnUNet git repo
RUN cd /home/cabinet && \
    mkdir SW && \
    git clone https://github.com/MIC-DKFZ/nnUNet.git && \
    cd nnUNet && \
    pip install -e .

# pull down main branch of BIBSNet
RUN cd  /home/cabinet/SW && \
    git clone https://github.com/DCAN-Labs/BIBSnet.git

ENV nnUNet_raw_data_base="/output"
ENV nnUNet_preprocessed="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models"

RUN mkdir -p /opt/nnUNet/nnUNet_raw_data_base/ /opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet
COPY trained_models/Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet
RUN cd /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet && unzip -qq Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip
COPY run.py /home/cabinet/run.py && \
    src /home/cabinet/src && \
    bin /home/cabinet/bin && \
    param-file-defaults.json /home/cabinet/param-file-defaults.json && \
    param-file-template.json /home/cabinet/param-file-template.json && \
    requirements.txt  /home/cabinet/requirements.txt && \
    xcp_arguments.txt   xcp_arguments.txt 

RUN cd /home/cabinet/ && chmod 555 run.py

ENTRYPOINT ["/home/cabinet/run.py"]
