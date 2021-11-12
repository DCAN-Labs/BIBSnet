#!/usr/bin/env python3
# coding: utf-8

"""
CABINET
Created: 2021-11-12
Updated: 2021-11-12
"""

# Import standard libraries
from datetime import datetime
from glob import glob
import math
import os
import pandas as pd
import random
import shutil
import subprocess
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


# Constants: Paths to this dir and level 1 analysis script
WRAPPER_LOC = '--wrapper-location'
SCRIPT_DIR = find_myself(WRAPPER_LOC)


# Custom local imports
from src.utilities import (
    add_slurm_args_to, exit_with_time_info, 
    get_pipeline_cli_argparser, validate_cli_args, 
)


def main():
    # Time how long the script takes and get command-line arguments from user 
    start = datetime.now()
    cli_args = get_cli_args()

    print(cli_args)  # TODO REMOVE LINE

    # Show user how long the pipeline took and end the pipeline here
    exit_with_time_info(start)


def get_cli_args(slurm=True):
    """
    :return: Dictionary containing all command-line arguments from user
    """
    parser = (add_slurm_args_to(get_pipeline_cli_argparser())
              if slurm else get_pipeline_cli_argparser())

    cli_args = validate_cli_args(vars(parser.parse_args()), parser)
    return cli_args


if __name__ == '__main__':
    main()