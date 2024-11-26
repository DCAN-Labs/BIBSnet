# Installation

Running the BIBSNet container requires installation of either Singularity or Docker. The container is hosted on our dockerhub under [dcanumn/bibsnet](https://hub.docker.com/r/dcanumn/bibsnet). To create a local container to execute, use the relevant command depending on which platform you choose:

Singularity: `singularity pull bibsnet.sif docker://dcanumn/bibsnet:latest`

Docker: `docker pull dcanumn/bibsnet:latest`

<br />

# Inputs
Currently, BIBSNet uses ALL anatomical images present in the BIDS input directory. Therefore, any images that you would like to exclude (e.g. due to poor QC) must be removed from the input directory. Similarly, to use the T1w- or T2w-only model, you will need to remove all T2w or T1w image files, respectively.

<br>

# Usage
## Command-Line Arguments

```
usage: BIBSnet [-h] [-participant PARTICIPANT_LABEL]
               [-end {prebibsnet,bibsnet,postbibsnet}]
               [--fsl-bin-path FSL_BIN_PATH]
               [--overwrite]
               [-ses SESSION] [-start {prebibsnet,bibsnet,postbibsnet}]
               [-w WORK_DIR] [-z] [-v | -d]
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
  -participant PARTICIPANT_LABEL, --subject PARTICIPANT_LABEL, -sub PARTICIPANT_LABEL, --participant-label PARTICIPANT_LABEL
                        The participant's unique subject identifier, without
                        'sub-' prefix. Example: 'ABC12345'
  -end {prebibsnet,bibsnet,postbibsnet}, --ending-stage {prebibsnet,bibsnet,postbibsnet}
                        Name of the stage to run last. By default, this will
                        be the postbibsnet stage. Valid choices: prebibsnet,
                        bibsnet, postbibsnet
  --fsl-bin-path FSL_BIN_PATH
                        Valid path to fsl bin. Defaults to the path used by
                        the container: /opt/fsl-6.0.5.1/bin/
  --overwrite, --overwrite-old
                        Include this flag to overwrite any previous BIBSnet
                        outputs in the derivatives sub-directories. Otherwise,
                        by default BIBSnet will skip creating any BIBSnet
                        output files that already exist in the sub-directories
                        of derivatives.
  -ses SESSION, --session SESSION, --session-id SESSION
                        The name of the session to processes participant data
                        for. Example: baseline_year1
  -start {prebibsnet,bibsnet,postbibsnet}, --starting-stage {prebibsnet,bibsnet,postbibsnet}
                        Name of the stage to run first. By default, this will
                        be the prebibsnet stage. Valid choices: prebibsnet,
                        bibsnet, postbibsnet
  -w WORK_DIR, --work-dir WORK_DIR
                        Valid absolute path where intermediate results should
                        be stored. Example: /path/to/working/directory
  -v, --verbose         Include this flag to print detailed information and
                        every command being run by BIBSnet to stdout.
                        Otherwise BIBSnet will only print warnings, errors,
                        and minimal output.
  -d, --debug           Include this flag to print highly detailed information
                        to stdout. Use this to see subprocess log statements
                        such as those for FSL, nnUNet and ANTS. --verbose is
                        recommended for standard use.

```

## Example Usage
Example run command using Singularity:
```
singularity run --nv --cleanenv --no-home \
-B /path/to/input:/input \
-B /path/to/output:/output \
/home/faird/shared/code/internal/pipelines/bibsnet_container/bibsnet_3.0.0.sif \
/input /output participant -v 
```

Example run command using Docker:
```
docker run --rm -it \
-v /path/to/input:/input \
-v /path/to/output:/output \
docker_image:version /input /output participant -v
```

<br>

# Dependencies
### Container
BIBSNet utilizes nnU-Net for model training and inference, i.e. deploying the trained model to generate image segmentations for new data. We therefore recommend running [BIBSnet](https://github.com/DCAN-Labs/BIBSnet) on a GPU if possible (e.g. Volta (v), Ampere (a), Turing (t) NVIDIA) as the [nnU-Net installation instructions](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1?tab=readme-ov-file#installation) note that running inference requires a GPU with 4 GB of VRAM. When running BIBSnet using a GPU, the job typically requires about 45 minutes, 20 tasks, and one node with 40 GB of memory. However, we have also had success running BIBSNet on a CPU with 40 GB of RAM.

### Application
We do not recommend running `BIBSnet` outside of the container because installing nnU-Net can be complicated and containerization ensures reproducibility by providing standardized software versions. However, if you wish to run `BIBSnet` as an application for development or other purposes, then you will need to do the following:

1. Download the appropriate data release from `https://s3.msi.umn.edu/bibsnet-data/<DATA_RELEASE>.tar.gz`
2. Extract files from `data.tar.gz` and move them into your locally cloned `BIBSnet` repository under `BIBSnet/data/`
3. Install [nnU-Net](https://github.com/MIC-DKFZ/nnUNet#installation) and other dependencies listed [here](https://github.com/DCAN-Labs/BIBSnet/network/dependencies)

<br>

# Testing
To test BIBSNet to ensure it's working as expected:

Install `openneuro-py` on the command line (e.g. `pip install openneuro-py`) and use it to download test data from [OpenNeuro](https://openneuro.org/), e.g.: 
```
# Python
from pathlib import Path
import openneuro

Path("./ds004776").mkdir()
openneuro.download(dataset="ds004776", target_dir="./ds004776", include="sub-01")
```

Build the BIBSNet container on the command line (`singularity pull bibsnet.sif docker://dcanumn/bibsnet:latest`) and execute with the test data, e.g.:
```
# bash

mkdir derivatives

singularity run --nv --cleanenv --no-home \
	-B /path/to/ds004776:/input \
	-B /path/to/derivatives:/output \
	/path/to/bibsnet.sif \
	/input /output participant \
	-participant 01 \
```

Ensure that the application runs without error and that the expected outputs are produced (see [Outputs](outputs.md)). For good measure, visually inspect the segmentation generated. Here is a snapshot view of the brain segmentation generated by this test subject:

![](openneuro_testdata_aseg.png)
