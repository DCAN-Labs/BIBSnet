#!/usr/bin/env python3
# coding: utf-8

"""
Connectome ABCD-XCP niBabies Imaging nnu-NET (CABINET)
Greg Conan: gconan@umn.edu
Created: 2021-11-12
Updated: 2023-01-25
"""
# Import standard libraries
from datetime import datetime

# Custom local imports
from src.utilities import (
    exit_with_time_info,
    extract_from_json,
    get_args,
    make_logger,
    run_all_stages
)


def main():
    start_time = datetime.now()  # Time how long the script takes
    logger = make_logger()  # Make object to log error/warning/status messages

    # Get and validate command-line arguments and parameters from .JSON file
    args = get_args()
    logger.info(f"Getting Arguments from arg file: {args}")
    json_args = extract_from_json(args.parameter_json)
    STAGES = list(json_args['stages'].keys())
    logger.info(f"Identified stages to be run: {STAGES}")
    
    # Run every stage that the parameter file says to run
    run_all_stages(STAGES, json_args, logger)
    # TODO default to running all stages if not specified by the user

    # Show user how long the pipeline took and end the pipeline here
    exit_with_time_info(start_time)


if __name__ == "__main__":
    main()
