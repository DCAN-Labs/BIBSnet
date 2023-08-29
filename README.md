# CABINET

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8015338.svg)](https://doi.org/10.5281/zenodo.8015338)

This [BIDS App](https://bids-apps.neuroimaging.io/about/) provides the utility of creating a [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) anatomical MRI segmentation and mask with a infant brain trained model for the purposes of circumventing JLF within [Nibabies](https://nibabies.readthedocs.io/en/latest/index.html). 

<br />

![CABINET - Stages for MRI Processing (7)](https://user-images.githubusercontent.com/102316699/229221477-001245e4-5687-413e-a996-4ee722e0ffc8.png)

<hr>

## Getting Started

To use CBAW, users will need to create a parameters json that contains the information about which containers will be run. The order of the containers in the parameter json is the order that the container will run them. Any amount of containers can be added to the json. The json will have any necessary binds, arguments, and flags to run the container. Below is an example for how BIBSNet would be run. 

        {
            "cabinet": {
                "container_type": "singularity",
                "verbose": true,
                "handle_missing_host_paths": "make_directories"
            },

            "stages": [
                {
                    "name":"bibsnet",
                    "sif_filepath": "/path/to/container/cabinet_v2.4.3.sif",
                    "singularity_args": {
                        "--cleanenv": true,
                        "--nv": true
                    },
                    "binds": [
                        {
                            "host_path":"/path/to/input/",
                            "container_path":"/input"
                        },
                        {
                            "host_path":"/path/to/output/derivatives/",
                            "container_path":"/output"
                        },
                        {
                            "host_path":"/path/to/workdir/",
                            "container_path":"/work"
                        }
                    ],
                    "positional_args": [
                        "/input", "/output", "participant"
                    ],
                    "flags": {
                        "--parameter-json":"/home/cabinet/parameter-file-container.json",
                        "-start":"prebibsnet",
                        "-end":"postbibsnet",
                        "-v": true,
                        "--participant-label": SUBJECT-ID,
                        "-w":"/work"
                    }
                }
            ]
        }

The parameter-jsons folder contains examples of parameter jsons that string together multiple containers. If you use CBAW for any containers that aren't in the parameter-jsons folder, please add your json as a resource for others to use. 

Once you have created your parameter.json, run `run.py` with your json as the single argument. CBAW will output the logging information for each step, as well as indicate when each stage begins and ends. 

For comprehensive information on CABINET, including installation and usage, visit <a href="https://cabinet.readthedocs.io/en/latest/" target="_blank">our documentation</a>.
