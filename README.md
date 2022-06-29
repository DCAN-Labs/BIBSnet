# CABINET

## Stages
1. PreBIBSnet
2. BIBSnet
3. PostBIBSnet
4. Nibabies (FUTURE)
5. ABCD-XCP (FUTURE)


## Installation

We highly recommend using the container to run CABINET. 

### Container

Container hosted here: 

singularity pull name_of_singularity_image.sif docker://path/to/docker/image #TODO FILL IN ACTUAL PATH

### Application

Set up CABINET environment: requirements.txt

Set up nnU-Net environment:

## Usage

### Container

#### Docker

#### Singularity

    singularity run --nv --cleanenv \
    -B $1:/input \
    -B $2:/output \
    -B $3:/param_file.json \
    /home/faird/shared/code/internal/pipelines/cabinet_container/cabinet_1_3_2.sif \
    /param_file.json -end postBIBSnet

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

## 1. PreBIBSnet

Crops the neck and shoulders out of infant images and resizes them.

## 2. BIBSnet

Produces a segmentation created by nnU-Net from a model trained on 0-8 month old infants.

## 3. PostBIBSnet

Checks the chirality of the segmentation and flips the regions if needed.

## 4. Nibabies

Infant fmriprep image processing

## 5. ABCD-XCP

DCANBOLDProc and Executive Summary
