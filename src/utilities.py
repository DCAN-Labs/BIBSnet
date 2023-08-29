#!/usr/bin/env python3
# coding: utf-8

"""
Common source for utility functions used by CABINET :)
Barry Tikalsky: tikal004@umn.edu
"""
# Import standard libraries
import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
import logging

# NOTE All functions below are in alphabetical order.

def exit_with_time_info(start_time, success):
    """
    Terminate the pipeline after displaying a message showing how long it ran
    :param start_time: datetime.datetime object of when the script started
    :param success: bool, whether all stages were successful
    """
    print("CABINET {}: {}"
          .format("took this long to run all stages successfully" if success else "ran for this long but some stages were not successful",
                  datetime.now() - start_time))
    sys.exit(0 if success else 1)

def extract_from_json(json_path):
    """
    :param json_path: String, a valid path to a real readable .json file
    :return: Dictionary, the contents of the file at json_path
    """
    with open(json_path, "r") as infile:
        return json.load(infile)

def get_args():
    """
    :return args: Namespace object containing the command line arguments
    """
    parser = argparse.ArgumentParser("CABINET")

    # Required positional arguments
    parser.add_argument(
        "parameter_json", type=valid_readable_json,
        help=("Required. Valid path to existing readable parameter .JSON "
              "file. See README.md and example parameter .JSON files for more "
              "information on parameters.")
    )
    args = parser.parse_args()
    return args

def get_binds(stage_args):
    '''
    :param to_bind: List of dicts, list of dicts with 'host_path' and 'container_path'
    :return binds: list of formatted binds for use in subprocess.check_call
    '''
    binds = []
    to_bind = stage_args['binds']
        
    for bind in to_bind:
        binds.append("-B")
        binds.append(f"{bind['host_path']}:{bind['container_path']}")
        
    return binds

def get_optional_args_in(a_dict):
    """
    :param a_dict: Dictionary with validated parameters,
                   all of which are used by this function
    :return: List of most a_dict optional arguments and their values
    """
    optional_args = []
    for arg in a_dict.keys():
        if a_dict[arg]:
            optional_args.append(arg)
            if isinstance(a_dict[arg], list):
                for el in a_dict[arg]:
                    optional_args.append(str(el))
            elif not isinstance(a_dict[arg], bool):
                optional_args.append(str(a_dict[arg]))
    return optional_args

def log_stage_finished(stage_name, event_time, logger, success):
    """
    Print and return a string showing how much time has passed since the
    current running script reached a certain part of its process
    :param stage_name: String, name of event that just finished
    :param event_time: datetime object representing when {stage_name} started
    :param sub_ses: List with either only the subject ID str or the session too
    :return: String with an easily human-readable message showing how much time
             has passed since {stage_start} when {stage_name} started.
    """
    successful = 'finished' if success else 'failed'
    logger.info("{0} {2}. "
                "Time elapsed since {0} started: {1}"
                .format(stage_name, datetime.now() - event_time, successful))
    

def make_logger():
    """
    Make logger to log status updates, warnings, and other important info
    :return: logging.Logger able to print info to stdout and problems to stderr
    """  # TODO Incorporate pprint to make printed JSONs/dicts more readable
    fmt = "\n%(levelname)s %(asctime)s: %(message)s"
    logging.basicConfig(stream=sys.stdout, format=fmt, level=logging.INFO)  
    logging.basicConfig(stream=sys.stderr, format=fmt, level=logging.ERROR)
    logging.basicConfig(stream=sys.stderr, format=fmt, level=logging.WARNING)
    return logging.getLogger(os.path.basename(sys.argv[0]))


def run_all_stages(j_args, logger):
    """
    Run stages sequentially, starting and ending at stages specified by user
    :param all_stages: List of functions in order where each runs one stage
    :param sub_ses_IDs: List of dicts mapping "age_months", "subject",
                        "session", etc. to unique values per subject session
    :param ubiquitous_j_args: Dictionary of all args needed by each stage
    :param logger: logging.Logger object to show messages and raise warnings
    """
    # ...run all stages that the user said to run
    success = True
    for stage in j_args['stages']:
        stage_start = datetime.now()
        if j_args["cabinet"]["verbose"]:
            logger.info("Now running stage: {}\n"
                        .format(stage['name']))
        stage_success = run_stage(stage, j_args, logger)
        log_stage_finished(stage['name'], stage_start, logger, stage_success)
        success = success and stage_success
    
    return success


def run_stage(stage, j_args, logger):
    '''
    Gathers arguments form parameter file, constructs container run command and runs it.
    :param stage: String, name of the stage to run
    :param j_args: Dictionary, copy of j_args
    :param logger: logging.Logger object to show messages and raise warnings
    '''
    if j_args['cabinet']['container_type'] == 'singularity':
        binds = get_binds(stage)
        singularity_args = get_optional_args_in(stage['singularity_args'])
        container_path = stage['sif_filepath']
        flag_stage_args = get_optional_args_in(stage['flags'])
        action = stage['action']
        stage_name = stage['name']
        positional_stage_args = stage['positional_args']

        cmd = ["singularity", action, *binds, *singularity_args, container_path, *positional_stage_args, *flag_stage_args]

        if j_args["cabinet"]["verbose"]:
            logger.info(f"run command for {stage_name}:\n{' '.join(cmd)}\n")

        try:
            subprocess.check_call(cmd)
            return True

        except Exception:
            logger.exception(f"Error running {stage_name}")
            return False


def valid_readable_file(path):
    """
    Throw exception unless parameter is a valid readable filepath string. Use
    this, not argparse.FileType("r") which leaves an open file handle.
    :param path: Parameter to check if it represents a valid filepath
    :return: String representing a valid filepath
    """
    return validate(path, lambda x: os.access(x, os.R_OK),
                    os.path.abspath, "Cannot read file at '{}'")


def valid_readable_json(path):
    """
    :param path: Parameter to check if it represents a valid .json file path
    :return: String representing a valid .json file path
    """
    return validate(path, lambda _: os.path.splitext(path)[-1] == ".json",
                    valid_readable_file,
                    "'{}' is not a path to a readable .json file")

def validate(to_validate, is_real, make_valid, err_msg, prepare=None):
    """
    Parent/base function used by different type validation functions. Raises an
    argparse.ArgumentTypeError if the input object is somehow invalid.
    :param to_validate: String to check if it represents a valid object 
    :param is_real: Function which returns true iff to_validate is real
    :param make_valid: Function which returns a fully validated object
    :param err_msg: String to show to user to tell them what is invalid
    :param prepare: Function to run before validation
    :return: to_validate, but fully validated
    """
    try:
        if prepare:
            prepare(to_validate)
        assert is_real(to_validate)
        return make_valid(to_validate)
    except (OSError, TypeError, AssertionError, ValueError,
            argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError(err_msg.format(to_validate))

def validate_parameter_json(j_args, json_path, logger):
    logger.info("Validating parameter JSON\n")

    is_valid = True

    # validate cabinet key
    if "cabinet" not in j_args.keys():
        logger.error("Missing key in parameter JSON: 'cabinet'")
        is_valid = False
    else:
        # set verbose
        if "verbose" in j_args['cabinet'].keys():
            if not isinstance(j_args['cabinet']['verbose'], bool):
                logger.error("Invalid value for cabinet verbosity, must be true or false.")
                is_valid = False
        else:
            j_args['cabinet']['verbose'] = False
        #set handle_missing_host_paths
        if "handle_missing_host_paths" in j_args["cabinet"].keys():
            options = [ # for any given bind, if the host path doesnt exist...
                'allow', # do nothing
                'make_directories', # make a directory there
                'stop'  # fail validation and do not execute cabinet.
            ]
            if j_args['cabinet']['handle_missing_host_paths'] not in options:
                logger.error(f"Invalid argument for handle_missing_host_paths. must be in f{options}")
                is_valid = False
        else:
            j_args['cabinet']['handle_missing_host_paths'] = "allow"
        # validate container_type
        if "container_type" not in j_args['cabinet']:
            logger.error("Missing key in parameter JSON: cabinet container_type")
            is_valid = False
        else:
            container_types = ["singularity"]
            if j_args['cabinet']['container_type'] not in container_types:
                logger.error(f"Invalid container type in parameter JSON.\ncabinet container_type must be in {container_types}")
                is_valid = False
            else:
                # validate stages key based on specified container type
                if j_args['cabinet']['container_type'] == "singularity":
                    if "stages" not in j_args.keys():
                        logger.error("Missing key in parameter JSON: 'stages'")
                        is_valid = False
                    else:
                        for stage_index, stage in enumerate(j_args['stages']):
                            stage_name = "Unnamed Stage"
                            if "name" not in stage.keys():
                                logger.error("Unnamed stage found. Please provide a name for all stages.")
                                is_valid = False
                            else:
                                stage_name = stage['name']
                            if "sif_filepath" not in stage.keys():
                                logger.error(f"Missing key 'sif_filepath' in stage {stage_name}")
                                is_valid = False                            
                            optional_args = { "singularity_args": {}, "binds": [], "positional_args": [], "flags": {}, "action": "run" }
                            for arg, default in optional_args.items():
                                if arg not in stage.keys():
                                    stage[arg] = default
                            if stage['action'] not in ['run', 'exec']:
                                logger.error(f"Invalid action '{stage['action']}' in {stage_name}, must be 'run' or 'exec'")
                            for binds in stage["binds"]:
                                if 'host_path' not in binds.keys() or 'container_path' not in binds.keys():
                                    logger.error(f"Invalid bind in {stage_name}. 'host_path' and 'container_path' are required for all binds.")
                                    is_valid = False
                                if not os.path.exists(binds['host_path']):
                                    if j_args["cabinet"]["handle_missing_host_paths"] == 'stop':
                                        logger.error(f"Host filepath for {stage_name} does not exist: {binds['host_path']}")
                                        is_valid = False
                                    elif j_args["cabinet"]["handle_missing_host_paths"] == 'make_directories':
                                        os.makedirs(binds["host_path"])
                                        logger.info(f"Made directory {binds['host_path']}")

                            j_args['stages'][stage_index] = stage

    if not is_valid:
        logger.error(f"Parameter JSON {json_path} is invalid. See https://cabinet.readthedocs.io/ for examples.")
        sys.exit()
    elif j_args['cabinet']['verbose']:
        logger.info(f"Parameter JSON {json_path} is valid.\nValidated JSON: {j_args}")

    return j_args

def validate_path(path, type, logger):
    """
    :param path: String, filepath
    :param type: String, 'file' or 'directory'
    :param logger: cabinet's logger object
    :return is_valid: bool, whether the path has passed validation
    """
    is_valid = True
    if type == "directory":
        if not os.is_dir(path):
            try:
                os.makedirs(path)
                logger.info(f"Created output directory: {path}")
            except:
                logger.error(f"Error creating non-existant directory: {path}")
                is_valid = False
    else:
        if not os.is_file(path):
            logger.error(f"File does not exist: {path}")
            is_valid = False

    return is_valid