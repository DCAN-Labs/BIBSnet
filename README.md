# CBAW

CBAW is a way to bind multiple containers together and run them with a single command. This is done by composing a parameters json that contains any necessary inputs for whichever containers your heart desires. With one command, CBAW will then run these containers one after the other to produce the desired outputs. 

The parameter-jsons folder contains examples of parameter jsons that string together multiple containers. If you use CBAW for any containers that aren't in the parameter-jsons folder, please add your json as a resource for others to use.

## Setting up the parameters.json

To use CBAW, users will need to create a parameters json that contains the information about which containers will be run. The order of the containers in the parameter json is the order that the container will run them. Any amount of containers can be added to the json. The json will have any necessary binds, arguments, and flags to run the container. There are two required keys: `cabinet` and `stages`. If you would like to make sure your json is valid before trying to run all of the containers, run `validate-param-json.py` with your json as the input. Before trying to run all of the containers, the main `run.py` script will also validate the json. If the json is invalid, it will not try to run the containers.

### cabinet variables

`container_type`: Required; the type of containers you are running (`singularity` or `docker`). Right now we can only support singularity containers.

`verbose`: Optional; value of true or false; if not specified, defaults to false.

`handle_missing_host_paths`: Optional; how you want the container to behave if any given paths don't exist; value of `stop`, `make_directories`, or `allow`; defaults to allow.

* stop will cause the json to be invalidated if paths don't exist.

* make_directories will create the directory if it doesn't exist. **NOTE:** If you're binding a file (instead of a directory) and it doesn't find the file, it will create a directory instead (not recommended).

* allow will not do anything about the missing directories.

### stages variables 

`name`: Required; name of the container

`sif_filepath`: Required; path to sif file of container

`action`: Optional; value of `exec` or `run`; defaults to `run`

`singularity_args`: Optional; arguments for singularity run options 

`binds`: Optional; any necessary binds for the containers; if specified, need both the `host_path` and `container_path`

`positional_args`: Optional; any necessary arguments for the container

`flags`: Optional; any necessary flags to run the container

All optional variables without an explicit default will appear in the validated JSON as an empty data structure (when using the verbose output).

### Example JSON

Below is an example for how BIBSNet would be run. 

        {
            "cabinet": {
                "container_type": "singularity",
                "verbose": true,
                "handle_missing_host_paths": "make_directories"
            },

            "stages": [
                {
                    "name":"bibsnet",
                    "sif_filepath": "/path/to/container/bibsnet.sif",
                    "action": "run",
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
 

## Running 

Once you have created your parameter.json, run `run.py` with your json as the single argument. 

At the beginning of each stage, CBAW will print the name of the stage and at which time it started. At the end of the stage, it will print out how long that stage took to run and if it completed successfully or failed. Right now, even if a stage fails, it will continue to try to run the remaining containers. If you have the `verbose` option set to true, at the very beginning it will print the validated JSON and during each container it will print the singularity commands it is running. 

After it runs all of the containers, it will print out how long the entire process took to run and if every stage completed successfully or if some stages failed. Right now, it will not tell you which stages failed at the end, you will have to look at the error and output logs to determine why and where it failed. 

Below is an example output from running bibsnet and lolcow. 

    INFO 2023-09-08 16:22:18,049: Getting Arguments from arg file: /panfs/jay/groups/6/faird/tikal004/CBAW/parameter-jsons/bibsnet-nibabies-xcpd.json

    INFO 2023-09-08 16:22:18,052: Validating parameter JSON

    INFO 2023-09-08 16:22:18,082: Made directory /home/feczk001/shared/projects/segpipeline_testing/Barry_test/cbaw-test/derivatives/bibsnet/

    INFO 2023-09-08 16:22:18,092: Parameter JSON /panfs/jay/groups/6/faird/tikal004/CBAW/parameter-jsons/bibsnet-nibabies-xcpd.json is valid.
    Validated JSON: {'cabinet': {'container_type': 'singularity', 'verbose': True, 'handle_missing_host_paths': 'make_directories'}, 
    'stages': [{'name': 'bibsnet', 'sif_filepath': '/home/faird/shared/code/internal/pipelines/cabinet_container/cabinet_v2.4.3.sif', 'singularity_args': {'--cleanenv': True, '--nv': True}, 
    'binds': [{'host_path': '/home/faird/shared/data/BCP-MVP_BIBSNet/input/', 'container_path': '/input'}, {'host_path': '/home/feczk001/shared/projects/segpipeline_testing/Barry_test/cbaw-test/derivatives/', 'container_path': '/output'}, 
    {'host_path': '/home/feczk001/shared/projects/segpipeline_testing/Barry_test/cbaw-test/work/', 'container_path': '/work'}], 'positional_args': ['/input', '/output', 'participant'], 
    'flags': {'--parameter-json': '/home/cabinet/parameter-file-container.json', '-start': 'prebibsnet', '-end': 'postbibsnet', '-v': True, '--participant-label': '123456', '-w': '/work'}, 'action': 'run'}, 
    {'name': 'lolcow', 'sif_filepath': '/home/faird/tikal004/lolcow.sif', 'action': 'exec', 'positional_args': ['cowsay', '"Thank you for testing CABINET"'], 'singularity_args': {}, 'binds': [], 'flags': {}}]}

    INFO 2023-09-08 16:22:18,092: Identified stages to be run: ['bibsnet', 'nibabies', 'xcpd', 'lolcow']

    INFO 2023-09-08 16:22:18,092: Now running stage: bibsnet

    INFO 2023-09-08 16:22:18,093: run command for bibsnet:
    singularity run -B /home/faird/shared/data/BCP-MVP_BIBSNet/input/:/input -B /home/feczk001/shared/projects/segpipeline_testing/Barry_test/cbaw-test/derivatives/:/output 
    -B /home/feczk001/shared/projects/segpipeline_testing/Barry_test/cbaw-test/work/:/work --cleanenv --nv 
    /home/faird/shared/code/internal/pipelines/cabinet_container/cabinet_v2.4.3.sif /input /output participant --parameter-json /home/cabinet/parameter-file-container.json -start prebibsnet -end postbibsnet -v --participant-label 123456 -w /work

    INFO 2023-09-08 22:00:30,762: bibsnet failed. Time elapsed since bibsnet started: 1:05:52.760840

    INFO 2023-09-08 22:00:30,762: Now running stage: lolcow

    INFO 2023-09-08 22:00:30,762: run command for lolcow:
    singularity exec /home/faird/tikal004/lolcow.sif cowsay "Thank you for testing CABINET"

    _________________________________
    < "Thank you for testing CABINET" >
    ---------------------------------
            \   ^__^
             \  (oo)\_______
                (__)\       )\/\
                    ||----w |
                    ||     ||

    INFO 2023-09-08 22:00:31,781: lolcow finished. Time elapsed since lolcow started: 0:00:01.018425
    CABINET ran for this long but some stages were not successful: 5:38:13.735623

