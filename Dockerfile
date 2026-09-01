ARG DOCKER_PLATFORM=linux/amd64
FROM --platform=$DOCKER_PLATFORM pennbbl/qsiprep-freesurfer:23.3.0 AS build_freesurfer
FROM --platform=$DOCKER_PLATFORM python:3.8-slim-bullseye AS builder

ENV BIBSNET_VERSION_MAJOR="3"
ENV BIBSNET_VERSION_MINOR="7"
ENV BIBSNET_VERSION_PATCH="0"
ENV BIBSNET_VERSION="${BIBSNET_VERSION_MAJOR}.${BIBSNET_VERSION_MINOR}.${BIBSNET_VERSION_PATCH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        curl \
        git \
        wget \
        build-essential \
        libc6-dev \
        bc \
        gcc \
        libgomp1 \
        libxmu6 \
        libxt6 \
        perl \
        tcsh \
        && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch==2.4.1+cu124

WORKDIR /
RUN wget -O bibsnet-${BIBSNET_VERSION_MAJOR}.${BIBSNET_VERSION_MINOR}.tar.gz "https://s3.msi.umn.edu/bibsnet-data/bibsnet-${BIBSNET_VERSION_MAJOR}.${BIBSNET_VERSION_MINOR}.tar.gz" && \
    tar -xzf bibsnet-${BIBSNET_VERSION_MAJOR}.${BIBSNET_VERSION_MINOR}.tar.gz && \
    rm bibsnet-${BIBSNET_VERSION_MAJOR}.${BIBSNET_VERSION_MINOR}.tar.gz

COPY requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
    cd /tmp && \
    git clone https://github.com/DCAN-Labs/nnUNet.git nnUNet && \
    cd nnUNet && \
    git checkout -b 1.7.1-maintenance v1.7.1m7 && \
    /opt/venv/bin/pip install --no-cache-dir "SimpleITK==2.4.1" && \
    /opt/venv/bin/pip install --no-cache-dir . && \
    cd /tmp && \
    wget https://github.com/freesurfer/surfa/archive/refs/tags/v0.6.0.tar.gz && \
    tar xvfz v0.6.0.tar.gz && \
    cd surfa-0.6.0 && \
    /opt/venv/bin/pip install --no-cache-dir Cython && \
    /opt/venv/bin/pip install --no-cache-dir . && \
    cd .. && \
    rm -rf /tmp/nnUNet /tmp/surfa-0.6.0 v0.6.0.tar.gz

FROM builder AS build-fsl

WORKDIR /
RUN echo "2024.04.25" && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj
ENV MAMBA_ROOT_PREFIX="/opt/conda"

# Fetch the FSL environment
COPY fsl.yml fsl.yml
RUN micromamba create -y -f fsl.yml && \
    rm fsl.yml

# Installing ANTs 2.3.3 (NeuroDocker build)
FROM builder AS build-ants

RUN echo "Downloading ANTs ..." && \
    mkdir -p /opt/ants && \
    tar -xzf ants-Linux-centos6_x86_64-v2.3.4.tar.gz -C /opt/ants --no-same-owner --strip-components 1 && \
    rm ants-Linux-centos6_x86_64-v2.3.4.tar.gz

# We can prune everything except the N4BiasFieldCorrection and DenoiseImage binaries
RUN mv /opt/ants/N4BiasFieldCorrection /opt/ants/DenoiseImage /opt/ && \
    rm -rf /opt/ants/* && \
    mv /opt/N4BiasFieldCorrection /opt/DenoiseImage /opt/ants/

# From the builder image, we can now build the final image with all dependencies
FROM --platform=$DOCKER_PLATFORM python:3.8-slim-bullseye AS runtime

ENV BIBSNET_VERSION_MAJOR="3"
ENV BIBSNET_VERSION_MINOR="7"
ENV BIBSNET_VERSION_PATCH="0"
ENV BIBSNET_VERSION="${BIBSNET_VERSION_MAJOR}.${BIBSNET_VERSION_MINOR}.${BIBSNET_VERSION_PATCH}"

# Runtime libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bc \
        bzip2 \
        ca-certificates \
        curl \
        libgomp1 \
        libxmu6 \
        libxt6 \
        perl \
        procps \
        tcsh \
        wget \
        && rm -rf /var/lib/apt/lists/*

COPY --from=build-fsl /opt/conda/envs/fsl /opt/conda/envs/fsl
COPY --from=build-ants /opt/ants /opt/ants
COPY --from=build_freesurfer /opt/freesurfer /opt/freesurfer
COPY --from=builder /opt/venv /opt/venv
ENV FSFAST_HOME="/opt/freesurfer/fsfast"
ENV VIRTUAL_ENV="/opt/venv" \
    PATH="/opt/venv/bin:/opt/afni-latest:/opt/ants:/opt/conda/envs/fsl/bin:$PATH" \
    FSLDIR="/opt/conda/envs/fsl" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q" \
    LD_LIBRARY_PATH="/opt/conda/envs/fsl/lib:/usr/lib/x86_64-linux-gnu" \
    AFNI_IMSAVE_WARNINGS="NO" \
    AFNI_PLUGINPATH="/opt/afni-latest"
# Simulate SetUpFreeSurfer.sh
ENV FSL_DIR="/opt/conda/envs/fsl" \
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

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users -u 1000 bibsnet
WORKDIR /home/bibsnet
ENV HOME="/home/bibsnet" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

ENV nnUNet_preprocessed="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed" \
    RESULTS_FOLDER="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models"

RUN mkdir -p /opt/nnUNet/nnUNet_raw_data_base/ /opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet /home/bibsnet/data

WORKDIR /
COPY --from=builder /Task*.tar.gz /
RUN tar -xzf Task543_BIBSnet_Production_T1T2_model.tar.gz -C /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --strip-components 1 && \
    rm Task543_BIBSnet_Production_T1T2_model.tar.gz && \
    tar -xzf Task547_BIBSnet_Production_T1only_model.tar.gz -C /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --strip-components 1 && \
    rm Task547_BIBSnet_Production_T1only_model.tar.gz && \
    tar -xzf Task548_BIBSnet_Production_T2only_model.tar.gz -C /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --strip-components 1 && \
    rm Task548_BIBSnet_Production_T2only_model.tar.gz    
COPY run.py /home/bibsnet/run.py
COPY src /home/bibsnet/src
COPY data /home/bibsnet/data

# Add bibsnet dir to path
ENV PATH="/home/bibsnet/:${PATH}"
RUN cp /home/bibsnet/run.py /home/bibsnet/bibsnet

RUN chmod -R a+rx /home/bibsnet /opt/nnUNet
ENTRYPOINT ["bibsnet"]
