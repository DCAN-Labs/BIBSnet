FROM nvcr.io/nvidia/pytorch:21.11-py3

# Manually update the BIBSnet version when building
ENV BIBSNET_VERSION="3.2.0"

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

# FSL 6.0.5.1
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           bc \
           dc \
           file \
           libfontconfig1 \
           libfreetype6 \
           libgl1-mesa-dev \
           libgl1-mesa-dri \
           libglu1-mesa-dev \
           libgomp1 \
           libice6 \
           libxcursor1 \
           libxft2 \
           libxinerama1 \
           libxrandr2 \
           libxrender1 \
           libxt6 \
           sudo \
           wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo "Downloading FSL ..." \
    && mkdir -p /opt/fsl-6.0.5.1 \
    && curl -sSL "https://s3.msi.umn.edu/bibsnet-data/bibsnet-v3.2.0.tar.gz" \
    | tar -xzpf - fsl-6.0.5.1-centos7_64.tar.gz -O | tar -xzpC /opt/fsl-6.0.5.1 --no-same-owner --strip-components 1
    
ENV FSLDIR="/opt/fsl-6.0.5.1" \
    PATH="/opt/afni-latest:/opt/ants:/opt/fsl-6.0.5.1/bin:$PATH" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q" \
    LD_LIBRARY_PATH="/opt/fsl-6.0.5.1/lib:$LD_LIBRARY_PATH" \
    AFNI_IMSAVE_WARNINGS="NO" \
    AFNI_PLUGINPATH="/opt/afni-latest"

# Installing ANTs 2.3.3 (NeuroDocker build)
# Note: the URL says 2.3.4 but it is actually 2.3.3
RUN mkdir -p /opt/ants \
    && curl -sSL --retry 5 "https://dl.dropbox.com/s/gwf51ykkk5bifyj/ants-Linux-centos6_x86_64-v2.3.4.tar.gz" \
    | tar -xzC /opt/ants --strip-components 1

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users -u 1000 bibsnet
WORKDIR /home/bibsnet
ENV HOME="/home/bibsnet" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# install nnUNet git repo
RUN cd /home/bibsnet && \
    mkdir SW && \
    git clone https://github.com/MIC-DKFZ/nnUNet.git && \
    cd nnUNet && \
    git checkout -b v1.7.1 v1.7.1 && \
    pip install -e .

#ENV nnUNet_raw_data_base="/output"
ENV nnUNet_preprocessed="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed" \
    RESULTS_FOLDER="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models"

RUN mkdir -p /opt/nnUNet/nnUNet_raw_data_base/ /opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet /home/bibsnet/data
#COPY trained_models/Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet
RUN curl -sSL "https://s3.msi.umn.edu/bibsnet-data/bibsnet-v3.2.0.tar.gz" | tar -xzpf - Task552_uniform_distribution_synthseg.tar.gz -O | tar -xzpC /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --no-same-owner --strip-components 1 && \
    curl -sSL "https://s3.msi.umn.edu/bibsnet-data/bibsnet-v3.2.0.tar.gz" | tar -xzpf - Task514_BCP_ABCD_Neonates_SynthSeg_T1Only.tar.gz -O | tar -xzpC /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --no-same-owner --strip-components 1 &&\
    curl -sSL "https://s3.msi.umn.edu/bibsnet-data/bibsnet-v3.2.0.tar.gz" | tar -xzpf - Task515_BCP_ABCD_Neonates_SynthSeg_T2Only.tar.gz -O | tar -xzpC /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --no-same-owner --strip-components 1 && \
    curl -sSL "https://s3.msi.umn.edu/bibsnet-data/bibsnet-v3.2.0.tar.gz" | tar -xzpf - Task526_BIBSNet_Production_Model.tar.gz -O | tar -xzpC /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --no-same-owner --strip-components 1

COPY run.py /home/bibsnet/run.py
COPY src /home/bibsnet/src
COPY bin /home/bibsnet/bin
RUN curl -sSL "https://s3.msi.umn.edu/bibsnet-data/bibsnet-v3.2.0.tar.gz" | tar -xzpf - data.tar.gz -O | tar -xzpC /home/bibsnet/data --no-same-owner --strip-components 1

COPY requirements.txt  /home/bibsnet/requirements.txt

#Add bibsnet dir to path
ENV PATH="${PATH}:/home/bibsnet/"
RUN cp /home/bibsnet/run.py /home/bibsnet/bibsnet

RUN cd /home/bibsnet/ && pip install -r requirements.txt
RUN cd /home/bibsnet/ && chmod 555 -R run.py bin src bibsnet

ENTRYPOINT ["bibsnet"]
