#!/usr/bin/env python3
# coding: utf-8

"""
BIBSnet
Greg Conan: gconan@umn.edu
Created: 2021-11-12
Barry Tikalsky: tikal004@umn.edu
Updated: 2023-10-04
Lucille A. Moore: lmoore@umn.edu
Updated: 2024-08-21
"""
# Import standard libraries
from datetime import datetime
import os

# Custom local imports
from src.get_args import get_params

from src.prebibsnet import run_preBIBSnet
from src.bibsnet import run_BIBSnet
from src.postbibsnet import run_postBIBSnet

from src.utilities import (
    exit_with_time_info,
    get_stage_name,
    run_all_stages
)

def main():
    start_time = datetime.now()  # Time how long the script takes

    # Get and validate command-line arguments
    STAGES = [run_preBIBSnet, run_BIBSnet, run_postBIBSnet]
    json_args, sub_ses_IDs = get_params([get_stage_name(stg) for stg in STAGES])

    # Set output dir environment variable for BIBSnet to user-defined output dir
    os.environ["nnUNet_raw_data_base"] = json_args["optional_out_dirs"]["derivatives"]
    
    # Run every stage that the parameter file says to run
    run_all_stages(STAGES, sub_ses_IDs, json_args["stage_names"]["start"],
                   json_args["stage_names"]["end"], json_args)
    # TODO default to running all stages if not specified by the user

    # Show user how long the pipeline took and end the pipeline here
    exit_with_time_info(start_time)


if __name__ == "__main__":
    main()
