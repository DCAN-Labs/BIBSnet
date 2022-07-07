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

#### Example paramater file fields: segmentation container
##### "common": parameters used by multiple stages within CABINET

- `"age_months"`: positive integer, the participants age in months. For example, `5` would mean the participant is 5 months old
- `"bids_dir"`: string, valid absolute path to existing BIDS base study directory. Example: `"path/to/bids/input"`
- `"fsl_bin_path"`: string, valid absolute path to existing `bin` directory in [FreeSurferLearner (FSL)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/). Example: `"/opt/fsl-6.0.5.1/bin/"`
- `"overwrite"`: boolean, `true` to overwrite any previous CABINET outputs in the derivatives sub-directories, or `false` to skip creating any CABINET output files that already exist in the derivatives sub-directories 
- `"participant_label"`: string, the participant's unique subject identifier. Example: `"ABC12345"`
- `"session"`: string, the name of the session to processes particpant data for. This parameter may be combined with `"age_months"` in the future. Example: `"ses-ID"`
- `"task_id"`: string, the name of the task performed by the participant to processes data for. This parameter can also be `null` for non task data. Example: `nback`
- `"verbose"`: boolean, `true` to print detailed information and every command being run by CABINET. `false` to not print this information
   
##### "optional_out_dirs": path to and names of CABINET output directories 
- `"derivatives"`:  string, valid path to the directory to save the eventual CABINET outputs to. Example: `"/path/to/output/derivatives"`
- `"preBIBSnet"`: string, valid path to preBIBSnet output directory, or `null` to save preBIBSnet outputs into `preBIBSnet` sub-directory in `derivatives` directory. 
- `"BIBSnet"`: string, valid path to BIBSnet output directory, or `null` to save BIBSnet outputs into `BIBSnet` sub-directory in `derivatives` directory.,
- `"postBIBSnet"`: string, valid path to postBIBSnet output directory, or `null` to save postBIBSnet outputs into `postBIBSnet` sub-directory in `derivatives` directory.,
- `"nibabies"`: string, valid path to nibabies output directory, or `null` to save nibabies outputs into `nibabies` sub-directory in `derivatives` directory.,
- `"XCPD"`: string, valid path to XCPD output directory, or `null` to save XCPD outputs into `XCPD` sub-directory in `derivatives` directory.

##### "resource_management": parameters to determine resource use when running parallel scripts. These parameters are only needed for nibabies and XCPD.
        
##### "preBIBSnet": parameters used only for the preBIBSnet stage 
- `"brain_z_size"`: positive integer, the size of the participant's brain along the z axis in mm. Example; `110`
- `"averaged_dir"`: string, naming the directory to save averaged brain image into. Example: `"averaged"`
- `"cropped_dir"`: string, naming the directory to save cropped brain image into. Example: `"cropped"`
- `"resized_dir"`: string, naming the directory to save resized brain image into. Example: `"resized"`

##### "BIBSnet": parameters used only for the BIBSnet stage
- `"model"`: string, model to run. Example: `"3d_fullres"`
- `"nnUNet_predict_path"`: string, valid path to nnUNet_predict executable file. Example: `"/opt/conda/bin/nnUNet_predict"`
- `"code_dir"`: string, valid path to directory containing BIBSnet python wrapper `run.py`. Example: `"/home/cabinet/SW/BIBSnet"`
- `"singularity_image_path"`: string, valid path to BIBSnet singularity image `.sif` file: Example: `"/home/cabinet/user/bibsnet.sif"`
- `"task"`: string, the name of the BIBSnet task performed by the participant to processes data for. Examples: `"512"`

##### "nibabies": [see here](https://nibabies.readthedocs.io/en/latest/index.html)

##### "XCPD": [see here](https://xcp-d.readthedocs.io/en/latest/)


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
