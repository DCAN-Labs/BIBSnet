# CABINET

[![DOI](https://zenodo.org/badge/427489916.svg)](https://zenodo.org/badge/latestdoi/427489916)

This [BIDS App](https://bids-apps.neuroimaging.io/about/) provides the utility of creating a [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) anatomical MRI segmentation and mask with a infant brain trained model for the purposes of circumventing JLF within [Nibabies](https://nibabies.readthedocs.io/en/latest/index.html). 

## Stages

1. PreBIBSnet
2. BIBSnet
3. PostBIBSnet
4. [Nibabies](https://nibabies.readthedocs.io/en/latest/index.html) (FUTURE)
5. [XCP-D](https://xcp-d.readthedocs.io/en/latest/) (FUTURE)

<br />

![CABINET - Stages for MRI Processing](https://user-images.githubusercontent.com/102316699/195385888-77f627e1-1389-4f0c-991d-eeb0c9e663b8.png)

<br />

## Dependencies 

docker or singularity (depending on what containerization software you will use for execution)

<br />

## Installation

We strongly recommend that users interested in producing segmentations run the containerized version of `CABINET`. However, developers may sometimes benefit from running the application outside of the container for testing. 

<br />

### Container

Container hosted here: https://hub.docker.com/r/dcanumn/cabinet

#### Singularity

    singularity pull cabinet.sif docker://dcanumn/cabinet

#### Docker

    docker pull dcanumn/cabinet

<br />

## Usage

The BIBSnet portion of CABINET needs a Volta (v), Ampere (a), or Turing (t) NVIDIA GPU.

### Command-Line Arguments

```
usage: CABINET [-h] -jargs PARAMETER_JSON [-participant PARTICIPANT_LABEL]
               [-age AGE_MONTHS] [-end {prebibsnet,bibsnet,postbibsnet}]
               [-model MODEL] [--overwrite] [-ses SESSION]
               [-start {prebibsnet,bibsnet,postbibsnet}] [-v] [-z]
               [--script-dir SCRIPT_DIR]
               bids_dir output_dir {participant}

positional arguments:
  bids_dir              Valid absolute path to existing base study directory
                        containing BIDS-valid input subject data directories.
                        Example: /path/to/bids/input/
  output_dir            Valid absolute path to existing derivatives directory
                        to save each stage's outputs by subject session into.
                        Example: /path/to/output/derivatives/
  {participant}         Processing level. Currently the only choice is
                        'participant'. See BIDS-Apps specification.

optional arguments:
  -h, --help            show this help message and exit
  -jargs PARAMETER_JSON, -params PARAMETER_JSON, --parameter-json PARAMETER_JSON
                        Required. Valid path to existing readable parameter
                        .JSON file. See README.md and example parameter .JSON
                        files for more information on parameters.
  -participant PARTICIPANT_LABEL, --subject PARTICIPANT_LABEL, -sub PARTICIPANT_LABEL, --participant-label PARTICIPANT_LABEL
                        The participant's unique subject identifier, without
                        'sub-' prefix. Example: 'ABC12345'
  -age AGE_MONTHS, -months AGE_MONTHS, --age-months AGE_MONTHS
                        Positive integer, the participant's age in months. For
                        example, -age 5 would mean the participant is 5 months
                        old. Include this argument unless the age in months is
                        specified in the participants.tsv file inside the BIDS
                        input directory.
  -end {prebibsnet,bibsnet,postbibsnet}, --ending-stage {prebibsnet,bibsnet,postbibsnet}
                        Name of the stage to run last. By default, this will
                        be the postbibsnet stage. Valid choices: prebibsnet,
                        bibsnet, postbibsnet
  -model MODEL, --model-number MODEL, --bibsnet-model MODEL
                        Model/task number for BIBSnet. By default, this will
                        be inferred from CABINET/data/models.csv based
                        on which data (T1, T2, or both) exists in the --bids-
                        dir.
  --overwrite, --overwrite-old
                        Include this flag to overwrite any previous CABINET
                        outputs in the derivatives sub-directories. Otherwise,
                        by default CABINET will skip creating any CABINET
                        output files that already exist in the sub-directories
                        of derivatives.
  -ses SESSION, --session SESSION, --session-id SESSION
                        The name of the session to processes participant data
                        for. Example: baseline_year1
  -start {prebibsnet,bibsnet,postbibsnet}, --starting-stage {prebibsnet,bibsnet,postbibsnet}
                        Name of the stage to run first. By default, this will
                        be the prebibsnet stage. Valid choices: prebibsnet,
                        bibsnet, postbibsnet
  -v, --verbose         Include this flag to print detailed information and
                        every command being run by CABINET to stdout.
                        Otherwise CABINET will only print warnings, errors,
                        and minimal output.
  -z, --brain-z-size    Include this flag to infer participants' brain height
                        (z) using the participants.tsv brain_z_size column.
                        Otherwise, CABINET will estimate the brain height from
                        the participant age and averages of a large sample of
                        infant brain heights.
  --script-dir SCRIPT_DIR
                        Valid path to the existing parent directory of this
                        run.py script. Include this argument if and only if
                        you are running the script as a SLURM/SBATCH job.
```

<br />

### Parameter `.JSON` File

The repository contains two parameter files, one recommended to run CABINET inside its container and one recommended to run outside:

- Inside Container: `parameter-file-container.json`
- Outside Container: `parameter-file-application.json`

#### "common": parameters used by multiple stages within CABINET

- `"fsl_bin_path"`: string, a valid absolute path to existing `bin` directory in [FMRIB Software Library](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/). Example: `"/opt/fsl-6.0.5.1/bin/"`
- `"task_id"`: string, the name of the task performed by the participant to processes data for. This parameter can also be `null` for non-task data. Example: `nback`

#### "resource_management": parameters to determine resource use when running parallel scripts. These parameters are only needed for nibabies and XCPD.

#### "BIBSnet": parameters used only for the BIBSnet stage
- `"model"`: string, the model to run. Example: `"3d_fullres"`
- `"nnUNet_predict_path"`: string, a valid path to nnUNet_predict executable file. Example: `"/opt/conda/bin/nnUNet_predict"`
- `"code_dir"`: string, a valid path to directory containing BIBSnet python wrapper `run.py`. Example: `"/home/cabinet/SW/BIBSnet"`
- `"singularity_image_path"`: string, a valid path to BIBSnet singularity image `.sif` file: Example: `"/home/cabinet/user/bibsnet.sif"`

#### "nibabies": [see here](https://nibabies.readthedocs.io/en/latest/index.html)

#### "XCPD": [see here](https://xcp-d.readthedocs.io/en/latest/)

<br />

### Container

The job typically takes about 45 minutes, 20 tasks, and one node with 40 gb of memory to run effectively. Less memory could result in holes in the segmentation produced by BIBSnet.

This has been primarily tested in Singularity. We are less able to provide technical support for Docker execution.

#### Docker

    docker run --rm -it \
    -v /path/to/input:/input \
    -v /path/to/output:/out \
    -v /path/to/param_file.json:param_file.json \
    docker_image:version /input /output participant -jargs /param_file.json \
    -end postbibsnet -v

#### Singularity

    singularity run --nv --cleanenv --no-home \
    -B /path/to/input:/input \
    -B /path/to/output:/output \
    -B /path/to/param_file.json:/param_file.json \
    /home/faird/shared/code/internal/pipelines/cabinet_container/cabinet_1_3_2.sif \
    /input /output participant -jargs /param_file.json -end postbibsnet -v 

<br />

### Application

We do not recommend running `CABINET` outside of the container for the following reasons:
1. Installing nnU-Net can be complicated.
2. Running `CABINET` inside the container ensures you have the proper versions of all software.
3. It is hard to diagnose your errors if you are working in a different environment.

However, if you run `CABINET` outside of the container as an application, then you will need to do the following:
1. Download the `data` directory from the `https://s3.msi.umn.edu/CABINET_data/data.zip` URL, unzip it, and move it into your cloned `CABINET` repository directory here: `CABINET/data/`
2. Install [nnU-Net](https://github.com/MIC-DKFZ/nnUNet#installation)

<br />

## Multiple Participant Requirements

### `participants.tsv`

#### Format Specification Example

| participant_id | session | age |
|:-:|:-:|:-:|
| sub-123456 | ses-A | 1 |

NOTE: `sub-` and `ses-` prefixes are currently required for `participant_id` and `session` values.

#### Content

When running multiple subjects and/or sessions, the `participants.tsv` file in the `bids_dir` must include an `age` column. In that column, each row has one positive integer, the participant's age in months at that session.

If the user wants to specify the brain height (shown below) for each subject session, then the user must also include an additional `"brain_z_size"` column. That column also must have a positive integer for each row, which is the size of the participant's brain along the z-axis in millimeters. Without a `brain_z_size` column, `CABINET` will calculate the `brain_z_size` value based on a table with [BCP](https://babyconnectomeproject.org/) participants' average head radius per age. That table is called `age_to_avg_head_radius_BCP.csv` under the `data` directory.

<center><img src="https://user-images.githubusercontent.com/102316699/184005162-0b1ebb76-3e5a-4bd3-b258-a686272e2ecc.png" width=555em></center>

<br />


## 1. PreBIBSnet

Prepares the anatomical BIDS images for BIBSnet segmentation generation.

1. Renames T1 and T2 to fit nnU-Net naming (0000 and 0001 for T1 and T2, respectively)
2. Average multiple T1/T2 weighted images into one T1/T2 weighted image
3. Crops the neck and shoulders out of the images
4. Registers the T2 to the T1 either via traditional transform or ACPC-alignment. Optimal method gets chosen via eta-squared calculation

<br />

## 2. BIBSnet

Quickly and accurately segments an optimally-aligned T1 and T2 pair with a deep neural network trained via nnU-Net and SynthSeg with a large 0 to 8 month old infant MRI brain dataset.

<br />

## 3. PostBIBSnet

1. Generates a left-right mask and dilates it
2. Uses dilated left-right mask to correct segmentation chirality
3. Registers the segmentation back into native T1 space using transform produced via optimal registration in preBIBSnet
4. Generates a mask of the segmentation from the native T1 space segmentation
5. Renames the native T1 space segmentation and mask to BIDS standard naming conventions to feed into Nibabies
6. Creates a "precomputed" directory for input into Nibabies, containing the final segmentation and mask along with the `dataset_description.file`

<br />

## 4. Nibabies (FUTURE)

Infant fmriprep image processing

<br />

## 5. XCP-D (FUTURE)

DCANBOLDProc and Executive Summary
