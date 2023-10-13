FROM nvcr.io/nvidia/pytorch:21.11-py3

# Manually update the BIBSnet version when building
ENV BIBSNET_VERSION="3.0.0"

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
    && curl -fsSL --retry 5 "https://s3.msi.umn.edu/cabinet-fsl-install/fsl-6.0.5.1-centos7_64_reduced.tar.gz" \
    | tar -xz -C /opt/fsl-6.0.5.1 --no-same-owner  --strip-components 1 \
    && find /opt/fsl-6.0.5.1/data/standard -type f -not -name "MNI152_T1_2mm_brain.nii.gz" -delete
ENV FSLDIR="/opt/fsl-6.0.5.1" \
    PATH="/opt/fsl-6.0.5.1/bin:$PATH" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q" \
    LD_LIBRARY_PATH="/opt/fsl-6.0.5.1/lib:$LD_LIBRARY_PATH"

ENV PATH="/opt/afni-latest:$PATH" \
    AFNI_IMSAVE_WARNINGS="NO" \
    AFNI_PLUGINPATH="/opt/afni-latest"

# Installing ANTs 2.3.3 (NeuroDocker build)
# Note: the URL says 2.3.4 but it is actually 2.3.3
ENV ANTSPATH="/opt/ants" \
    PATH="/opt/ants:$PATH"
WORKDIR $ANTSPATH
RUN curl -sSL --retry 5 "https://dl.dropbox.com/s/gwf51ykkk5bifyj/ants-Linux-centos6_x86_64-v2.3.4.tar.gz" \
    | tar -xzC $ANTSPATH --strip-components 1

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
ENV nnUNet_preprocessed="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models"

RUN mkdir -p /opt/nnUNet/nnUNet_raw_data_base/ /opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet /home/bibsnet/data
#COPY trained_models/Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet
RUN curl -sSL "https://s3.msi.umn.edu/cabinet-data-targz/Task552_uniform_distribution_synthseg.tar.gz" | tar -xzC /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --no-same-owner --strip-components 1 && \
    curl -sSL "https://s3.msi.umn.edu/cabinet-data-targz/Task514_BCP_ABCD_Neonates_SynthSeg_T1Only.tar.gz" | tar -xzC /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --no-same-owner --strip-components 1 && \
    curl -sSL "https://s3.msi.umn.edu/cabinet-data-targz/Task515_BCP_ABCD_Neonates_SynthSeg_T2Only.tar.gz" | tar -xzC /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --no-same-owner --strip-components 1

RUN curl -sSL "https://dl.dropbox.com/s/gwf51ykkk5bifyj/ants-Linux-centos6_x86_64-v2.3.4.tar.gz" \
    | tar -xzC $ANTSPATH --strip-components 1

COPY run.py /home/bibsnet/run.py
COPY src /home/bibsnet/src
RUN chmod 777 -R /opt/fsl-6.0.5.1
RUN bash /home/bibsnet/src/fixpy.sh /opt/fsl-6.0.5.1
COPY bin /home/bibsnet/bin
RUN curl -sSL "https://s3.msi.umn.edu/cabinet-data-targz/data.tar.gz" | tar -xzC /home/bibsnet/data --no-same-owner --strip-components 1

COPY requirements.txt  /home/bibsnet/requirements.txt

#Add bibsnet dir to path
ENV PATH="${PATH}:/home/bibsnet/"
RUN cp /home/bibsnet/run.py /home/bibsnet/bibsnet

RUN cd /home/bibsnet/ && pip install -r requirements.txt
RUN cd /home/bibsnet/ && chmod 555 -R run.py bin src bibsnet data
RUN chmod -R a+r /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet/3d_fullres
RUN find /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet/3d_fullres -type f -name 'postprocessing.json' -exec chmod 666 {} \;

ENTRYPOINT ["bibsnet"]
