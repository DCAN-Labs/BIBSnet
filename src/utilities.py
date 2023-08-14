#!/usr/bin/env python3
# coding: utf-8

"""
Common source for utility functions used by CABINET :)
Greg Conan: gconan@umn.edu
Created: 2021-11-12
Updated: 2023-01-26
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

def as_cli_arg(arg_str):
    """
    :param arg_str: String naming a stored argument taken from the command line
    :return: String which is the command-line argument form of arg_str
    """
    return "--" + arg_str.replace("_", "-")

def exit_with_time_info(start_time, exit_code=0):
    """
    Terminate the pipeline after displaying a message showing how long it ran
    :param start_time: datetime.datetime object of when the script started
    :param exit_code: exit code
    """
    print("CABINET {}: {}"
          .format("took this long to run all stages successfully" if exit_code == 0 else "ran for this long and then crashed",
                  datetime.now() - start_time))
    sys.exit(exit_code)

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

def get_binds(to_bind):
    '''
    :param to_bind: List of dicts, list of dicts with 'host_path' and 'container_path'
    :return binds: list of formatted binds for use in subprocess.check_call
    '''
    binds = []
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
    optional_args = list()
    for arg in a_dict.keys():
        if a_dict[arg]:
            optional_args.append(as_cli_arg(arg))
            if isinstance(a_dict[arg], list):
                for el in a_dict[arg]:
                    optional_args.append(str(el))
            elif not isinstance(a_dict[arg], bool):
                optional_args.append(str(a_dict[arg]))
    return optional_args

def log_stage_finished(stage_name, event_time, logger):
    """
    Print and return a string showing how much time has passed since the
    current running script reached a certain part of its process
    :param stage_name: String, name of event that just finished
    :param event_time: datetime object representing when {stage_name} started
    :param sub_ses: List with either only the subject ID str or the session too
    :return: String with an easily human-readable message showing how much time
             has passed since {stage_start} when {stage_name} started.
    """
    logger.info("{0} finished. "
                "Time elapsed since {0} started: {1}"
                .format(stage_name, datetime.now() - event_time))

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


def run_all_stages(all_stages, j_args, logger):
    """
    Run stages sequentially, starting and ending at stages specified by user
    :param all_stages: List of functions in order where each runs one stage
    :param sub_ses_IDs: List of dicts mapping "age_months", "subject",
                        "session", etc. to unique values per subject session
    :param ubiquitous_j_args: Dictionary of all args needed by each stage
    :param logger: logging.Logger object to show messages and raise warnings
    """
    if j_args["cabinet"]["verbose"]:
        logger.info("All parameters from input args and input .JSON file:\n{}"
                    .format(j_args))

    # ...run all stages that the user said to run
    for stage in all_stages:
        stage_start = datetime.now()
        if j_args["cabinet"]["verbose"]:
            logger.info("Now running stage: {}\n"
                        .format(stage))
        run_stage(stage, j_args, logger)
        log_stage_finished(stage, stage_start, logger)


def run_stage(stage, j_args, logger):
    '''
    :param stage: String, name of the stage to run
    :param j_args: Dictionary, copy of j_args
    :param logger: logging.Logger object to show messages and raise warnings
    '''
    if j_args['cabinet']['container_type'] == 'singularity':
        binds = get_binds(j_args['stages'][stage]['binds'])
        run_args = get_optional_args_in(j_args['stages'][stage]['run_args'])
        container = j_args['stages'][stage]['container_path']
        positional_stage_args =j_args['stages'][stage]['stage_args']['positional_args']
        flag_stage_args = get_optional_args_in(j_args['stages'][stage]['stage_args']['flags'])

        action = "exec" if  "exec" in j_args['stages'][stage].keys() else "run"

        cmd = ["singularity", action, *binds, *run_args, container, *positional_stage_args, *flag_stage_args]
        logger.info(f"run command for {stage}:\n{' '.join(cmd)}\n")
        try:
            subprocess.check_call(cmd)
        except Exception:
            logger.exception(f"Error running {stage}")
    else:
        pass

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
