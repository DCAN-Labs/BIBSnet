## Usage

CABINET can be run with either a CPU or GPU. The BIBSnet portion of CABINET needs a Volta (v), Ampere (a), or Turing (t) NVIDIA GPU.

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
                        be inferred from CABINET/data/models.csv based on
                        which data exists in the --bids-dir. BIBSnet will run 
                        model 514 by default for T1w-only, model 515 for 
                        T2w-only, and model 5550 for both T1w and T2w.
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
- `"task_id"`: string, the name of the task performed by the participant to processes data for. This parameter can also be `null` for non-task data. Example: `nback` (note: this is not utilized by cabinet yet, please designate it as null)

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

When running CABINET using a GPU, the job typically takes about 45 minutes, 20 tasks, and one node with 40 gb of memory to run effectively. When running CABINET with a CPU, the job typically takes about 1.5 hours, 20 tasks, and a node with 80 gb of memory to run effectively. Less memory could result in holes in the segmentation produced by BIBSnet.

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