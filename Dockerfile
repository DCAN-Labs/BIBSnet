FROM nvcr.io/nvidia/pytorch:21.11-py3 as build_pytorch
FROM pennbbl/qsiprep-freesurfer:23.3.0 as build_freesurfer
FROM build_pytorch
# Manually update the BIBSnet version when building

ENV BIBSNET_VERSION="3.4.1"

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
		    zlib1g \
                    bc \
		    libgomp1 \
                    libxmu6 \
                    perl \
		    tcsh && \
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
    && wget -O bibsnet-v3.3.0.tar.gz "https://s3.msi.umn.edu/bibsnet-data/bibsnet-v3.3.0.tar.gz" \
    && tar -xzf bibsnet-v3.3.0.tar.gz fsl-6.0.5.1-centos7_64.tar.gz \
    && tar -xzf fsl-6.0.5.1-centos7_64.tar.gz -C /opt/fsl-6.0.5.1 --no-same-owner --strip-components 1 \
    && rm bibsnet-v3.3.0.tar.gz fsl-6.0.5.1-centos7_64.tar.gz

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

## Freesurfer
COPY --from=build_freesurfer /opt/freesurfer /opt/freesurfer
# Simulate SetUpFreeSurfer.sh
ENV FSL_DIR="/opt/fsl-6.0.5.1" \
    OS="Linux" \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT="nii.gz" \
    FREESURFER_HOME="/opt/freesurfer"
ENV SUBJECTS_DIR="$FREESURFER_HOME/subjects" \
    FUNCTIONALS_DIR="$FREESURFER_HOME/sessions" \
    MNI_DIR="$FREESURFER_HOME/mni" \
    LOCAL_DIR="$FREESURFER_HOME/local" \
    MINC_BIN_DIR="$FREESURFER_HOME/mni/bin" \
    MINC_LIB_DIR="$FREESURFER_HOME/mni/lib" \
    MNI_DATAPATH="$FREESURFER_HOME/mni/data"
ENV PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    MNI_PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    PATH="$FREESURFER_HOME/bin:$FSFAST_HOME/bin:$FREESURFER_HOME/tktools:$MINC_BIN_DIR:$PATH" \
    FREESURFER_DEPS="bc ca-certificates curl libgomp1 libxmu6 libxt6 tcsh perl"
RUN chmod a+rx /opt/freesurfer/bin/mri_synthseg /opt/freesurfer/bin/mri_synthstrip

# Installing ANTs 2.3.3 (NeuroDocker build)
# Note: the URL says 2.3.4 but it is actually 2.3.3
# TESTING: installing ANTS from tier2 tar.gz
RUN echo "Downloading ANTs ..." && \
    mkdir -p /opt/ants && \
    wget -O bibsnet-v3.3.0.tar.gz "https://s3.msi.umn.edu/bibsnet-data/bibsnet-v3.3.0.tar.gz" && \
    tar -xzf bibsnet-v3.3.0.tar.gz ants-Linux-centos6_x86_64-v2.3.4.tar.gz && \
    tar -xzf ants-Linux-centos6_x86_64-v2.3.4.tar.gz -C /opt/ants --no-same-owner --strip-components 1 && \
    rm bibsnet-v3.3.0.tar.gz ants-Linux-centos6_x86_64-v2.3.4.tar.gz

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

RUN wget -O bibsnet-v3.3.0.tar.gz "https://s3.msi.umn.edu/bibsnet-data/bibsnet-v3.3.0.tar.gz" && \
    tar -xzf bibsnet-v3.3.0.tar.gz Task540_BIBSnet_Production_T1T2_model.tar.gz && \
    tar -xzf Task540_BIBSnet_Production_T1T2_model.tar.gz -C /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --strip-components 1 && \
    rm bibsnet-v3.3.0.tar.gz Task540_BIBSnet_Production_T1T2_model.tar.gz

RUN wget -O bibsnet-v3.3.0.tar.gz "https://s3.msi.umn.edu/bibsnet-data/bibsnet-v3.3.0.tar.gz" && \
    tar -xzf bibsnet-v3.3.0.tar.gz Task541_BIBSnet_Production_T1only_model.tar.gz && \
    tar -xzf Task541_BIBSnet_Production_T1only_model.tar.gz -C /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --strip-components 1 && \
    rm bibsnet-v3.3.0.tar.gz Task541_BIBSnet_Production_T1only_model.tar.gz

RUN wget -O bibsnet-v3.3.0.tar.gz "https://s3.msi.umn.edu/bibsnet-data/bibsnet-v3.3.0.tar.gz" && \
    tar -xzf bibsnet-v3.3.0.tar.gz Task542_BIBSnet_Production_T2only_model.tar.gz && \
    tar -xzf Task542_BIBSnet_Production_T2only_model.tar.gz -C /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --strip-components 1 && \
    rm bibsnet-v3.3.0.tar.gz Task542_BIBSnet_Production_T2only_model.tar.gz

COPY run.py /home/bibsnet/run.py
COPY src /home/bibsnet/src
RUN wget -O bibsnet-v3.3.0.tar.gz "https://s3.msi.umn.edu/bibsnet-data/bibsnet-v3.3.0.tar.gz" && \
    tar -xzf bibsnet-v3.3.0.tar.gz data.tar.gz && \
    tar -xzf data.tar.gz -C /home/bibsnet/data --strip-components 1 && \
    rm bibsnet-v3.3.0.tar.gz data.tar.gz

RUN \
  cd /tmp && \
  wget https://github.com/freesurfer/surfa/archive/refs/tags/v0.6.0.tar.gz && \
  tar xvfz v0.6.0.tar.gz && \
  cd surfa-0.6.0 && \
  python setup.py install && \
  cd .. && \
  rm -rf surfa-0.6.0 v0.6.0.tar.gz

COPY requirements.txt  /home/bibsnet/requirements.txt

#Add bibsnet dir to path
ENV PATH="${PATH}:/home/bibsnet/"
RUN cp /home/bibsnet/run.py /home/bibsnet/bibsnet

#adding in debugging for nomodulefound pandas error
RUN cd /home/bibsnet/ && pip install -v -r requirements.txt
RUN pip list | grep pandas
RUN python -c "import pandas; print(pandas.__version__)"
RUN cd /home/bibsnet/ && chmod 555 -R run.py src bibsnet

ENTRYPOINT ["bibsnet"]
