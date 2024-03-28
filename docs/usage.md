## Usage

[BIBSnet](https://github.com/DCAN-Labs/BIBSnet) can only be run with a GPU, and cannot currently be run with a CPU. BIBSnet needs a Volta (v), Ampere (a), or Turing (t) NVIDIA GPU.

### Command-Line Arguments

```
usage: BIBSnet [-h] [-participant PARTICIPANT_LABEL] [-age AGE_MONTHS]
               [-end {prebibsnet,bibsnet,postbibsnet}]
               [--fsl-bin-path FSL_BIN_PATH] [-jargs PARAMETER_JSON]
               [-model MODEL] [--nnUNet NNUNET]
               [--nnUNet-configuration {2d,3d_fullres,3d_lowres,3d_cascade_fullres}]
               [--overwrite] [-ses SESSION]
               [-start {prebibsnet,bibsnet,postbibsnet}] [-w WORK_DIR]
               [-z] [-v | -d]
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
  -age AGE_MONTHS, -months AGE_MONTHS, --age-months AGE_MONTHS
                        Positive integer, the participant's age in months. For
                        example, -age 5 would mean the participant is 5 months
                        old. Include this argument unless the age in months is
                        specified in each subject's sub-{}_sessions.tsv file
                        inside its BIDS input directory or inside the
                        participants.tsv file inside the BIDS directory at
                        thesubject-level.
  -end {prebibsnet,bibsnet,postbibsnet}, --ending-stage {prebibsnet,bibsnet,postbibsnet}
                        Name of the stage to run last. By default, this will
                        be the postbibsnet stage. Valid choices: prebibsnet,
                        bibsnet, postbibsnet
  --fsl-bin-path FSL_BIN_PATH
                        Valid path to fsl bin. Defaults to the path used by
                        the container: /opt/fsl-6.0.5.1/bin/
  -jargs PARAMETER_JSON, -params PARAMETER_JSON, --parameter-json PARAMETER_JSON
                        Parameter JSON is deprecated. All arguments formerly
                        in this file are now flags. This argument does
                        nothing. See https://bibsnet.readthedocs.io/ for
                        updated usage.
  -model MODEL, --model-number MODEL, --bibsnet-model MODEL
                        Model/task number for BIBSnet. By default, this will
                        be inferred from /home/bibsnet/data/models.csv based
                        on which data exists in the --bids-dir. BIBSnet will
                        run model 514 by default for T1w-only, model 515 for
                        T2w-only, and model 552 for both T1w and T2w.
  --nnUNet NNUNET, -n NNUNET
                        Valid path to existing executable file to run nnU-
                        Net_predict. By default, this script will assume that
                        nnU-Net_predict will be the path used by the
                        container: /opt/conda/bin/nnUNet_predict
  --nnUNet-configuration {2d,3d_fullres,3d_lowres,3d_cascade_fullres}
                        The nnUNet configuration to use.Defaults to 3d_fullres
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
  -z, --brain-z-size    Include this flag to infer participants' brain height
                        (z) using the sub-{}_sessions.tsv or participant.tsv
                        brain_z_size column.Otherwise, BIBSnet will estimate
                        the brain height from the participant age and averages
                        of a large sample of infant brain heights.
  -v, --verbose         Include this flag to print detailed information and
                        every command being run by BIBSnet to stdout.
                        Otherwise BIBSnet will only print warnings, errors,
                        and minimal output.
  -d, --debug           Include this flag to print highly detailed information
                        to stdout. Use this to see subprocess log statements
                        such as those for FSL, nnUNet and ANTS. --verbose is
                        recommended for standard use.

```

<br />

### Container

When running BIBSnet using a GPU, the job typically takes about 45 minutes, 20 tasks, and one node with 40 gb of memory to run effectively. Less memory could result in holes in the segmentation produced by BIBSnet.

This has been primarily tested in Singularity. We are less able to provide technical support for Docker execution.

#### Docker

    docker run --rm -it \
    -v /path/to/input:/input \
    -v /path/to/output:/out \
    docker_image:version /input /output participant -v

#### Singularity

    singularity run --nv --cleanenv --no-home \
    -B /path/to/input:/input \
    -B /path/to/output:/output \
    /home/faird/shared/code/internal/pipelines/bibsnet_container/bibsnet_3.0.0.sif \
    /input /output participant -v 

<br />

### Application

We do not recommend running `BIBSnet` outside of the container for the following reasons:

1. Installing nnU-Net can be complicated.
1. Running `BIBSnet` inside the container ensures you have the proper versions of all software.
1. It is hard to diagnose your errors if you are working in a different environment.

However, if you run `BIBSnet` outside of the container as an application, then you will need to do the following:

1. Download the `data` directory from the `https://s3.msi.umn.edu/CABINET_data/data.zip` URL, unzip it, and move it into your cloned `BIBSnet` repository directory here: `BIBSnet/data/`
1. Install [nnU-Net](https://github.com/MIC-DKFZ/nnUNet#installation)

<br />
