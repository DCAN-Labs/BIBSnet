# Installation

We highly recommend running BIBSnet using a container service such as [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) or [Docker](https://www.docker.com/get-started/): please visit the links provided for installation instructions. 

The BIBSNet container is hosted on Docker Hub under [dcanumn/bibsnet](https://hub.docker.com/r/dcanumn/bibsnet). Once the container service is installed, create a local container to execute using the relevant command:

*Singularity*
```
singularity pull bibsnet.sif docker://dcanumn/bibsnet:latest
```

*Docker*
```
docker pull dcanumn/bibsnet:latest
```

To pull a specific version, replace `latest` (which is connected to the most recent version) with the version number, e.g: 
```
docker pull dcanumn/bibsnet:release-3.4.2
```


# Usage

## BIDS Input Data

BIBSNet expects at least one T1-weighted and/or one T2-weighted structural MRI image in valid [BIDS format](https://bids.neuroimaging.io/), e.g.:

```
assembly_bids/ 
|__ participants.json 
|__ sub-<label>/
|   |__ ses-<label>/
|       |__ anat/
|       |   |__ sub-<label>_ses-<label>_run-<label>_T1w.nii.gz 
|       |   |__ sub-<label>_ses-<label>_run-<label>_T1w.json
|       |   |__ sub-<label>_ses-<label>_run-<label>_T2w.nii.gz
|       |   |__ sub-<label>_ses-<label>_run-<label>_T2w.json
```

Note that, currently, BIBSNet uses ALL anatomical images present in the BIDS input directory. Therefore, any images that you would like to exclude (e.g. due to poor QC) must be removed from the input directory. Similarly, to use the T1w- or T2w-only model, you will need to remove all T2w or T1w image files, respectively.

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
