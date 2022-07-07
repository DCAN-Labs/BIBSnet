# CABINET : Connectome XCP-D (ABCD-XCP) Infant nnuNET

This software provides the utility of creating a [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) anatomical MRI segmentation and mask with a infant brain trained model for the purposes of circumventing JLF within [Nibabies](https://nibabies.readthedocs.io/en/latest/index.html). 

## Stages
1. PreBIBSnet
2. BIBSnet
3. PostBIBSnet
4. [Nibabies](https://nibabies.readthedocs.io/en/latest/index.html) (FUTURE)
5. [XCP-D](https://xcp-d.readthedocs.io/en/latest/) (FUTURE)

![CABINET - Stages for MRI Processing](https://user-images.githubusercontent.com/102316699/177638055-14894a92-9cb6-4a70-a649-71f61d47f3d6.png)

## Installation

Developers can benefit from running the application outside of the container for testing. For users interested in producing segmentations, we recommend one use the container instead

### Container

Container hosted here: 

#### Singularity

    singularity pull name_of_singularity_image.sif docker://path/to/docker/image #TODO FILL IN ACTUAL PATH

#### Docker

    docker pull path/to/docker/image

### Application

Set up CABINET environment: requirements.txt

Set up nnU-Net environment:

## Usage

The BIBSnet portion of CABINET needs a Volta (v), Ampere (a), or Turing (t) NVIDIA GPU.


    usage: run.py [-h] [-start {preBIBSnet,BIBSnet,postBIBSnet,nibabies,XCPD}]
                  [-end {preBIBSnet,BIBSnet,postBIBSnet,nibabies,XCPD}]
                  [--script-dir SCRIPT_DIR]
                  parameter_json

    positional arguments:
      parameter_json        Valid path to existing readable parameter .json file.
                            See README.md for more information on parameters.

    optional arguments:
      -h, --help            show this help message and exit
      -start {preBIBSnet,BIBSnet,postBIBSnet,nibabies,XCPD}, --starting-stage {preBIBSnet,BIBSnet,postBIBSnet,nibabies,XCPD}
      -end {preBIBSnet,BIBSnet,postBIBSnet,nibabies,XCPD}, --ending-stage {preBIBSnet,BIBSnet,postBIBSnet,nibabies,XCPD}
      --script-dir SCRIPT_DIR
                            Valid path to the existing parent directory of this
                            run.py script. Include this argument if and only if
                            you are running the script as a SLURM/SBATCH job.

#TODO EXPLAIN PARAM FILE FIELDS
#### Example paramater file fields: segmentation container

    {
    "common": {
        "age_months": change,
        "bids_dir": "path/to/bids/input",
        "fsl_bin_path": "/opt/fsl-6.0.5.1/bin/",
        "overwrite": true,
        "participant_label": "sub-ID#",
        "session": "ses-ID",
        "task_id": null,
        "verbose": true
    },

    "optional_out_dirs": {
        "derivatives": "path/to/output",
        "preBIBSnet": null,
        "BIBSnet": null,
        "postBIBSnet": null,
        "nibabies": null,
        "XCPD": null
    },

    "resource_management": {
        "mem_mb": null,
        "n_cpus": null,
        "nipype_plugin_file": null,
        "nthreads": null,
        "omp_nthreads": null,
        "resource_monitor": null
    },

    "preBIBSnet": {
        "brain_z_size": change,
        "averaged_dir": "averaged",
        "cropped_dir": "cropped",
        "resized_dir": "resized"
    },

    "BIBSnet": {
        "model": "3d_fullres",
        "nnUNet_predict_path": "/opt/conda/bin/nnUNet_predict",
        "code_dir": "/home/cabinet/SW/BIBSnet",
        "singularity_image_path": "/home/feczk001/gconan/placeholder.txt",
        "task": "512"
    },

### Container

The job typically takes about 45 minutes, 20 tasks, and one node with 40 gb of memory to run effectively. Less memory could result in holes in the segmentation produced by BIBSnet.

#### Docker

    docker run --rm -e DOCKER_VERSION_8395080871=20.10.6 -it \
    -v /path/to/input:/input \
    -v /path/to/output:/out \
    -v /path/to/param_file.json:param_file.json \
    docker_image:version /param_file.json -start preBIBSnet -end postBIBSnet


#### Singularity

    singularity run --nv --cleanenv \
    -B /path/to/input:/input \
    -B /path/to/output:/output \
    -B /path/to/param_file.json:/param_file.json \
    /home/faird/shared/code/internal/pipelines/cabinet_container/cabinet_1_3_2.sif \
    /param_file.json -end postBIBSnet


## 1. PreBIBSnet

Prepares the anatomical BIDS images for BIBSnet segmentation generation.

1. Renames T1 and T2 to fit nnU-Net naming (0000 and 0001 for T1 and T2, respectively)
2. Average multiple T1/T2 weighted images into one T1/T2 weighted image
3. Crops the neck and shoulders out of the images
4. Registers the T2 to the T1 either via traditional transform or ACPC-alignment. Optimal method gets chosen via eta-squared calculation


## 2. BIBSnet

Produces a segmentation from the optimal pair of T1 and T2 aligned images created by nnU-Net from a model trained on 0-8 month old infants.

## 3. PostBIBSnet

1. Generates a left-right mask and dilates it
2. Uses dilated left-right mask to correct segmentation chirality
3. Registers the segmentation back into native T1 space using transform produced via optimal registration in preBIBSnet
4. Generates a mask of the segmentation from the native T1 space segmentation
5. Renames the native T1 space segmentation and mask to BIDS standard naming conventions to feed into Nibabies


## 4. Nibabies (FUTURE)

Infant fmriprep image processing

## 5. XCP-D (FUTURE)

DCANBOLDProc and Executive Summary
