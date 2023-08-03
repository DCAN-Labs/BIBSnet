#!/usr/bin/env python3
# coding: utf-8

"""
Connectome ABCD-XCP niBabies Imaging nnu-NET (CABINET)
Greg Conan: gconan@umn.edu
Created: 2021-11-12
Updated: 2023-01-25
"""
# Import standard libraries
import argparse
from datetime import datetime
import logging
import os
import sys


def find_myself(flg):
    """
    Ensure that this script can find its dependencies if parallel processing
    :param flg: String in sys.argv immediately preceding the path to return
    :return: String, path to the directory this Python script is in
    """
    try:
        parallel_flag_pos = -2 if sys.argv[-2] == flg else sys.argv.index(flg)
        sys.path.append(os.path.abspath(sys.argv[parallel_flag_pos + 1]))
    except (IndexError, ValueError):
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    return sys.path[-1]


# Global constants: Paths to this dir and to level 1 analysis script
SCRIPT_DIR_ARG = "--script-dir"
SCRIPT_DIR = find_myself(SCRIPT_DIR_ARG)
AGE_TO_HEAD_RADIUS_TABLE = os.path.join(SCRIPT_DIR, "data",
                                        "age_to_avg_head_radius_BCP.csv")
LR_REGISTR_PATH = os.path.join(SCRIPT_DIR, "bin", "LR_mask_registration.sh")
TYPES_JSON = os.path.join(SCRIPT_DIR, "src", "param-types.json")

# Custom local imports
from src.utilities import (
    exit_with_time_info, run_all_stages, valid_readable_json, extract_from_json
)


def main():
    start_time = datetime.now()  # Time how long the script takes
    logger = make_logger()  # Make object to log error/warning/status messages

    # Get and validate command-line arguments and parameters from .JSON file
    json_args = get_params_from_JSON()
    STAGES = json_args['stages'].keys()
    
    # Run every stage that the parameter file says to run
    run_all_stages(STAGES, json_args, logger)
    # TODO default to running all stages if not specified by the user

    # Show user how long the pipeline took and end the pipeline here
    exit_with_time_info(start_time)


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


def get_params_from_JSON():
    """
    :return: Dictionary containing all parameters from parameter .JSON file
    """
    parser = argparse.ArgumentParser("CABINET")
    # TODO will want to add positional 'input' and 'output' arguments and '--participant-label' and '--session-label' arguments. For the HBCD study, we won't to have to create a JSON per scanning session, but this will likely be fine for the pilot.

    # Required flag arguments
    parser.add_argument(
        "-jargs", "-params", "--parameter-json", dest="parameter_json",
        type=valid_readable_json, required=True,
        help=("Required. Valid path to existing readable parameter .JSON "
              "file. See README.md and example parameter .JSON files for more "
              "information on parameters.")
        # TODO: Add description of all nibabies and XCP-D parameters to the README?
        # TODO: In the README.md file, mention which arguments are required and which are optional (with defaults)
    )
    args = parser.parse_args()
    return extract_from_json(args['parameter_json'])

if __name__ == "__main__":
    main()
