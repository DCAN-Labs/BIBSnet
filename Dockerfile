FROM nvcr.io/nvidia/pytorch:21.11-py3

# Manually update the CABINET version when building
ENV CABINET_VERSION="2.4.5"

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
    && curl -fsSL --retry 5 https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-6.0.5.1-centos7_64.tar.gz \
    | tar -xz -C /opt/fsl-6.0.5.1 --strip-components 1 \
    --exclude "fsl/config" \
    --exclude "fsl/data/first" \
    --exclude "fsl/data/mist" \
    --exclude "fsl/data/possum" \
    --exclude "fsl/data/standard/bianca" \
    --exclude "fsl/data/standard/tissuepriors" \
    --exclude "fsl/doc" \
    --exclude "fsl/etc/default_flobs.flobs" \
    --exclude "fsl/etc/js" \
    --exclude "fsl/etc/luts" \
    --exclude "fsl/etc/matlab" \
    --exclude "fsl/extras" \
    --exclude "fsl/include" \
    --exclude "fsl/refdoc" \
    --exclude "fsl/src" \
    --exclude "fsl/tcl" \
    --exclude "fsl/bin/FSLeyes" \
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
RUN curl -sSL "https://dl.dropbox.com/s/gwf51ykkk5bifyj/ants-Linux-centos6_x86_64-v2.3.4.tar.gz" \
    | tar -xzC $ANTSPATH --strip-components 1

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
    git checkout -b v1.7.1 v1.7.1 && \
    pip install -e .

# pull down main branch of BIBSNet
RUN cd  /home/cabinet/SW && \
    git clone https://github.com/DCAN-Labs/BIBSnet.git

#ENV nnUNet_raw_data_base="/output"
ENV nnUNet_preprocessed="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models"

RUN mkdir -p /opt/nnUNet/nnUNet_raw_data_base/ /opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet /home/cabinet/data
#COPY trained_models/Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet
RUN wget https://s3.msi.umn.edu/CABINET_data/Task552_uniform_distribution_synthseg.zip -O /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet/Task552_uniform_distribution_synthseg.zip && \
    wget https://s3.msi.umn.edu/CABINET_data/Task514_BCP_ABCD_Neonates_SynthSeg_T1Only.zip -O /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet/Task514_BCP_ABCD_Neonates_SynthSeg_T1Only.zip && \
    wget https://s3.msi.umn.edu/CABINET_data/Task515_BCP_ABCD_Neonates_SynthSeg_T2Only.zip -O /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet/Task515_BCP_ABCD_Neonates_SynthSeg_T2Only.zip
RUN cd /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet && \
    unzip -qq Task552_uniform_distribution_synthseg.zip && \
    rm Task552_uniform_distribution_synthseg.zip && \
    unzip -qq Task514_BCP_ABCD_Neonates_SynthSeg_T1Only.zip && \
    rm Task514_BCP_ABCD_Neonates_SynthSeg_T1Only.zip && \
    unzip -qq Task515_BCP_ABCD_Neonates_SynthSeg_T2Only.zip && \
    rm Task515_BCP_ABCD_Neonates_SynthSeg_T2Only.zip

COPY run.py /home/cabinet/run.py
COPY src /home/cabinet/src
RUN chmod 777 -R /opt/fsl-6.0.5.1
RUN bash /home/cabinet/src/fixpy.sh /opt/fsl-6.0.5.1
COPY bin /home/cabinet/bin
#COPY data /home/cabinet/data
RUN wget https://s3.msi.umn.edu/CABINET_data/data.zip -O /home/cabinet/data/temp_data.zip && cd /home/cabinet/data && unzip -qq temp_data.zip && rm temp_data.zip

COPY parameter-file-application.json /home/cabinet/parameter-file-application.json
COPY parameter-file-container.json /home/cabinet/parameter-file-container.json
COPY requirements.txt  /home/cabinet/requirements.txt

#Add cabinet dir to path
ENV PATH="${PATH}:/home/cabinet/"
RUN cp /home/cabinet/run.py /home/cabinet/cabinet

RUN cd /home/cabinet/ && pip install -r requirements.txt 
RUN cd /home/cabinet/ && chmod 555 -R run.py bin src parameter-file-application.json parameter-file-container.json cabinet data
RUN chmod 666 /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet/3d_fullres/Task*/nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json

ENTRYPOINT ["cabinet"]
