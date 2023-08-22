#!/usr/bin/env python3
# coding: utf-8

# Import standard libraries
from datetime import datetime

# Custom local imports
from src.utilities import (
    exit_with_time_info,
    extract_from_json,
    get_args,
    make_logger,
    run_all_stages,
    validate_parameter_json
)


def main():
    start_time = datetime.now()  # Time how long the script takes
    logger = make_logger()  # Make object to log error/warning/status messages

    # Get and validate command-line arguments and parameters from .JSON file
    args = get_args()
    json_path = args.parameter_json
    logger.info(f"Getting Arguments from arg file: {json_path}")
    json_args = extract_from_json(args.parameter_json)
    json_args = validate_parameter_json(json_args, json_path, logger)
    STAGES = [stage['name'] for stage in json_args['stages']]
    logger.info(f"Identified stages to be run: {STAGES}")
    
    # Run every stage that the parameter file says to run
    success = run_all_stages(json_args, logger)
    # TODO default to running all stages if not specified by the user

    # Show user how long the pipeline took and end the pipeline here
    exit_with_time_info(start_time, success)


if __name__ == "__main__":
    main()
