# CABINET

This software provides the utility of creating a [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) anatomical MRI segmentation and mask with a infant brain trained model for the purposes of circumventing JLF within [Nibabies](https://nibabies.readthedocs.io/en/latest/index.html). 

## Stages

1. PreBIBSnet
2. BIBSnet
3. PostBIBSnet
4. [Nibabies](https://nibabies.readthedocs.io/en/latest/index.html) (FUTURE)
5. [XCP-D](https://xcp-d.readthedocs.io/en/latest/) (FUTURE)

![CABINET - Stages for MRI Processing](https://user-images.githubusercontent.com/102316699/177638055-14894a92-9cb6-4a70-a649-71f61d47f3d6.png)

## Dependencies 

docker or singularity (depending on what containerization software you will use for execution)

## Installation

Developers can benefit from running the application outside of the container for testing. For users interested in producing segmentations, we recommend one use the container instead

### Container

Container hosted here: 

#### Singularity

    singularity pull cabinet.sif docker://path/to/docker/image #TODO FILL IN ACTUAL PATH

#### Docker

    docker pull path/to/docker/image


## Usage

The BIBSnet portion of CABINET needs a Volta (v), Ampere (a), or Turing (t) NVIDIA GPU.

### Command-Line Arguments

```
    usage: run.py [-h] -jargs PARAMETER_JSON -participant PARTICIPANT_LABEL [-ses SESSION] [-age AGE_MONTHS] [-v]
                [--overwrite] [-start {prebibsnet,bibsnet,postbibsnet}] [-end {prebibsnet,bibsnet,postbibsnet}]
                [--script-dir SCRIPT_DIR]
                bids_dir output_dir {participant}

    positional arguments:
    bids_dir        Valid absolute path to
                    existing base study directory
                    containing BIDS-valid input
                    subject data directories.
                    Example: /path/to/bids/input/
    output_dir      Valid absolute path to
                    existing derivatives directory
                    to save each stage's outputs
                    by subject session into.
                    Example: /path/to/output/derivatives/
    {participant}   Processing level. Currently
                    the only choice is
                    'participant'. See BIDS-Apps
                    specification.

    optional arguments:
    -h, --help          show this help message
                        and exit
    -jargs PARAMETER_JSON, -params PARAMETER_JSON, --parameter-json PARAMETER_JSON
            Valid path to existing readable
            parameter .JSON file. See README.md
            and example parameter .JSON files for
            more information on parameters.
    -participant PARTICIPANT_LABEL, --subject PARTICIPANT_LABEL, -sub PARTICIPANT_LABEL, --participant-label PARTICIPANT_LABEL
            The participant's unique subject
            identifier, without 'sub-'prefix.
            Example: 'ABC12345'
    -ses SESSION, --session SESSION, --session-id SESSION
            The name of the session to processes
            participant data for. Example:
            baseline_year1
    -age AGE_MONTHS, -months AGE_MONTHS, --age-months AGE_MONTHS
            Positive integer, the participant's
            age in months. For example, -age 5
            would mean the participant is 5 months
            old. Include this argument unless the
            age in months is specified in the
            participants.tsv file inside the BIDS
            input directory.
    -v, --verbose
            Include this flag to print detailed
            information and every command being
            run by CABINET to stdout. Otherwise
            CABINET will only print warnings,
            errors, and minimal output.
    --overwrite, --overwrite-old
            Include this flag to overwrite any
            previous CABINET outputs in the
            derivatives sub-directories.
            Otherwise, by default CABINET will
            skip creating any CABINET output files
            that already exist in the
            sub-directories of derivatives.
    -start {prebibsnet,bibsnet,postbibsnet}, --starting-stage {prebibsnet,bibsnet,postbibsnet}
            Name of the stage to run first. By
            default, this will be the prebibsnet
            stage. Valid choices: prebibsnet,
            bibsnet, postbibsnet
    -end {prebibsnet,bibsnet,postbibsnet}, --ending-stage {prebibsnet,bibsnet,postbibsnet}
            Name of the stage to run last. By
            default, this will be the postbibsnet
            stage. Valid choices: prebibsnet,
            bibsnet, postbibsnet
    --script-dir SCRIPT_DIR
            Valid path to the existing parent
            directory of this run.py script.
            Include this argument if and only if
            you are running the script as a SLURM
            SBATCH job.
```

### Example paramater file fields: segmentation container

#### "common": parameters used by multiple stages within CABINET

- `"fsl_bin_path"`: string, a valid absolute path to existing `bin` directory in [FreeSurferLearner (FSL)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/). Example: `"/opt/fsl-6.0.5.1/bin/"`
- `"task_id"`: string, the name of the task performed by the participant to processes data for. This parameter can also be `null` for non-task data. Example: `nback`

#### "resource_management": parameters to determine resource use when running parallel scripts. These parameters are only needed for nibabies and XCPD.
        
#### "preBIBSnet": parameters used only for the preBIBSnet stage 
- `"brain_z_size"`: positive integer, the size of the participant's brain along the z axis in mm. Example: `110`

![brainzsize2](https://user-images.githubusercontent.com/102316699/184005162-0b1ebb76-3e5a-4bd3-b258-a686272e2ecc.png)

#### "BIBSnet": parameters used only for the BIBSnet stage
- `"model"`: string, the model to run. Example: `"3d_fullres"`
- `"nnUNet_predict_path"`: string, a valid path to nnUNet_predict executable file. Example: `"/opt/conda/bin/nnUNet_predict"`
- `"code_dir"`: string, a valid path to directory containing BIBSnet python wrapper `run.py`. Example: `"/home/cabinet/SW/BIBSnet"`
- `"singularity_image_path"`: string, a valid path to BIBSnet singularity image `.sif` file: Example: `"/home/cabinet/user/bibsnet.sif"`
- `"task"`: string naming the BIBSnet task performed by the participant to processes data for. Examples: `"512"`

#### "nibabies": [see here](https://nibabies.readthedocs.io/en/latest/index.html)

#### "XCPD": [see here](https://xcp-d.readthedocs.io/en/latest/)


### Container

The job typically takes about 45 minutes, 20 tasks, and one node with 40 gb of memory to run effectively. Less memory could result in holes in the segmentation produced by BIBSnet.

This has been primarily tested in Singularity. We are less able to provide technical support for Docker execution.

#### Docker

    docker run --rm -it \
    -v /path/to/input:/input \
    -v /path/to/output:/out \
    -v /path/to/param_file.json:param_file.json \
    docker_image:version /input /output participant -jargs /param_file.json \
    -end postbibsnet --participant_label subject_id -ses session_id -age age_months -v


#### Singularity

    singularity run --nv --cleanenv --no-home \
    -B /path/to/input:/input \
    -B /path/to/output:/output \
    -B /path/to/param_file.json:/param_file.json \
    /home/faird/shared/code/internal/pipelines/cabinet_container/cabinet_1_3_2.sif \
    /input /output participant -jargs /param_file.json -end postbibsnet \
    --participant_label subject_id -ses session_id -age age_months -v 


## 1. PreBIBSnet

Prepares the anatomical BIDS images for BIBSnet segmentation generation.

1. Renames T1 and T2 to fit nnU-Net naming (0000 and 0001 for T1 and T2, respectively)
2. Average multiple T1/T2 weighted images into one T1/T2 weighted image
3. Crops the neck and shoulders out of the images
4. Registers the T2 to the T1 either via traditional transform or ACPC-alignment. Optimal method gets chosen via eta-squared calculation


## 2. BIBSnet

Quickly and accurately segments an optimally-aligned T1 and T2 pair with a deep neural network trained via nnU-Net and SynthSeg with a large 0 to 8 month old infant MRI brain dataset.

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
