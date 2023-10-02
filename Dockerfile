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
    && curl -fsSL --retry 5 "https://s3.msi.umn.edu/cabinet-fsl-install/fsl-6.0.5.1-centos7_64.tar.gz" \
    | tar -xz -C /opt/fsl-6.0.5.1 --no-same-owner  --strip-components 1 \
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
    --exclude "fsl/bin/probtrackx2_gpu"  \
    --exclude "fsl/bin/mist"  \
    --exclude "fsl/bin/eddy_cuda8.0"  \
    --exclude "fsl/bin/eddy_cuda10.2"  \
    --exclude "fsl/bin/eddy_cuda9.1"  \
    --exclude "fsl/bin/melodic"  \
    --exclude "fsl/bin/eddy_openmp"  \
    --exclude "fsl/bin/fabber_asl"  \
    --exclude "fsl/bin/fabber_cest"  \
    --exclude "fsl/bin/probtrackx2"  \
    --exclude "fsl/bin/flameo"  \
    --exclude "fsl/bin/run_mesh_utils"  \
    --exclude "fsl/bin/fabber_dualecho"  \
    --exclude "fsl/bin/fabber_dsc"  \
    --exclude "fsl/bin/fabber_dce"  \
    --exclude "fsl/bin/fdt_matrix_merge"  \
    --exclude "fsl/bin/mvntool"  \
    --exclude "fsl/bin/fabber"  \
    --exclude "fsl/bin/fabber_dwi"  \
    --exclude "fsl/bin/fabber_pet"  \
    --exclude "fsl/bin/fabber_qbold"  \
    --exclude "fsl/bin/fabber_t1"  \
    --exclude "fsl/bin/surf_proj"  \
    --exclude "fsl/bin/surf2surf"  \
    --exclude "fsl/bin/probtrackx"  \
    --exclude "fsl/bin/film_gls"  \
    --exclude "fsl/bin/ftoz"  \
    --exclude "fsl/bin/ttoz"  \
    --exclude "fsl/bin/ttologp"  \
    --exclude "fsl/bin/contrast_mgr"  \
    --exclude "fsl/bin/qboot"  \
    --exclude "fsl/bin/cluster"  \
    --exclude "fsl/bin/xfibres"  \
    --exclude "fsl/bin/dtifit"  \
    --exclude "fsl/bin/vecreg"  \
    --exclude "fsl/binfsl_mvlm"  \
    --exclude "fsl/bin/new_invwarp"  \
    --exclude "fsl/bin/mm"  \
    --exclude "fsl/binfsl_regfilt"  \
    --exclude "fsl/bin/pvmfit"  \
    --exclude "fsl/binfsl_sbca"  \
    --exclude "fsl/binfslsurfacemaths"  \
    --exclude "fsl/binfsl_glm"  \
    --exclude "fsl/bin/swe"  \
    --exclude "fsl/binfsl_schurprod"  \
    --exclude "fsl/bin/first_utils"  \
    --exclude "fsl/bin/filmbabe"  \
    --exclude "fsl/bin/surf2volume"  \
    --exclude "fsl/bin/first"  \
    --exclude "fsl/bin/find_the_biggest"  \
    --exclude "fsl/bin/feat_model"  \
    --exclude "fsl/bin/tsplot"  \
    --exclude "fsl/bin/proj_thresh"  \
    --exclude "fsl/bin/midtrans"  \
    --exclude "fsl/binfsl_histogram"  \
    --exclude "fsl/bin/pulse"  \
    --exclude "fsl/bin/possum"  \
    --exclude "fsl/bin/asl_mfree"  \
    --exclude "fsl/bin/fugue"  \
    --exclude "fsl/bin/asl_file"  \
    --exclude "fsl/bin/fast"  \
    --exclude "fsl/bin/ccops"  \
    --exclude "fsl/bin/swap_subjectwise"  \
    --exclude "fsl/bin/swap_voxelwise"  \
    --exclude "fsl/bin/estimate_metric_distortion"  \
    --exclude "fsl/bin/prelude"  \
    --exclude "fsl/bin/betsurf"  \
    --exclude "fsl/bin/bet2"  \
    --exclude "fsl/bin/sigloss"  \
    --exclude "fsl/bin/pointflirt"  \
    --exclude "fsl/bin/signal2image"  \
    --exclude "fsl/bin/xfibres_gpu"  \
    --exclude "fsl/bin/pnm_evs"  \
    --exclude "fsl/bin/b0calc"  \
    --exclude "fsl/bin/merge_parts_gpu"  \
    --exclude "fsl/bin/slicer"  \
    --exclude "fsl/bin/dtigen"  \
    --exclude "fsl/bin/drawmesh"  \
    --exclude "fsl/bin/makerot"  \
    --exclude "fsl/bin/lesion_filling"  \
    --exclude "fsl/bin/distancemap"  \
    --exclude "fsl/bin/overlay"  \
    --exclude "fsl/bin/prewhiten"  \
    --exclude "fsl/bin/fdr"  \
    --exclude "fsl/bin/spharm_rm"  \
    --exclude "fsl/bin/tbss_skeleton"  \
    --exclude "fsl/binfslcc"  \
    --exclude "fsl/bin/slicetimer"  \
    --exclude "fsl/binfslsmoothfill"  \
    --exclude "fsl/bin/first_mult_bcorr"  \
    --exclude "fsl/bin/smoothest"  \
    --exclude "fsl/bin/calc_grad_perc_dev"  \
    --exclude "fsl/bin/avscale"  \
    --exclude "fsl/bin/make_dyadic_vectors"  \
    --exclude "fsl/binfslpspec"  \
    --exclude "fsl/bin/susan"  \
    --exclude "fsl/binfslfft"  \
    --exclude "fsl/bin/unconfound"  \
    --exclude "fsl/bin/rmsdiff"  \
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
RUN useradd -m -s /bin/bash -G users -u 1000 cabinet
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
RUN curl -sSL "https://s3.msi.umn.edu/cabinet-data-targz/Task552_uniform_distribution_synthseg.tar.gz" | tar -xzC /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --no-same-owner --strip-components 1 && \
    curl -sSL "https://s3.msi.umn.edu/cabinet-data-targz/Task514_BCP_ABCD_Neonates_SynthSeg_T1Only.tar.gz" | tar -xzC /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --no-same-owner --strip-components 1 && \
    curl -sSL "https://s3.msi.umn.edu/cabinet-data-targz/Task515_BCP_ABCD_Neonates_SynthSeg_T2Only.tar.gz" | tar -xzC /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --no-same-owner --strip-components 1

RUN curl -sSL "https://dl.dropbox.com/s/gwf51ykkk5bifyj/ants-Linux-centos6_x86_64-v2.3.4.tar.gz" \
    | tar -xzC $ANTSPATH --strip-components 1

COPY run.py /home/cabinet/run.py
COPY src /home/cabinet/src
RUN chmod 777 -R /opt/fsl-6.0.5.1
RUN bash /home/cabinet/src/fixpy.sh /opt/fsl-6.0.5.1
COPY bin /home/cabinet/bin
#COPY data /home/cabinet/data
RUN curl -sSL "https://s3.msi.umn.edu/cabinet-data-targz/data.tar.gz" | tar -xzC /home/cabinet/data --no-same-owner --strip-components 1

COPY parameter-file-application.json /home/cabinet/parameter-file-application.json
COPY parameter-file-container.json /home/cabinet/parameter-file-container.json
COPY requirements.txt  /home/cabinet/requirements.txt

#Add cabinet dir to path
ENV PATH="${PATH}:/home/cabinet/"
RUN cp /home/cabinet/run.py /home/cabinet/cabinet

RUN cd /home/cabinet/ && pip install -r requirements.txt 
RUN cd /home/cabinet/ && chmod 555 -R run.py bin src parameter-file-application.json parameter-file-container.json cabinet data
RUN find /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet/3d_fullres -type f -name 'postprocessing.json' -exec chmod 666 {} \;
RUN chmod -R a+r /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet/3d_fullres

ENTRYPOINT ["cabinet"]
