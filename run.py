#!/usr/bin/env python3
# coding: utf-8

"""
CABINET
Created: 2021-11-12
Updated: 2021-11-18
"""

# Import standard libraries
import argparse
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
    exit_with_time_info, extract_from_json, valid_readable_json
)


def main():
    # Time how long the script takes and get command-line arguments from user 
    start = datetime.now()
    cli_args = get_cli_args()

    print(cli_args)  # TODO REMOVE LINE

    # Run nnU-Net
    cli_args = crop_and_resize_images(cli_args)  # Somebody else already writing this (Paul?)
    if cli_args["age_months"] <= 8:
        cli_args = copy_images_to_nnUNet_dir(cli_args)  # TODO
        segmentation = un_nnUNet_predict(cli_args)
        segmentation = ensure_chirality_flipped(segmentation)  # Paul(?) already writing this

    # Put just the mask and the segmentation (and nothing else) into a directory to run nibabies
        mask = make_mask(cli_args, segmentation)  # Luci has a script for this, but it's clunky so we'll just use it as a blueprint
        run_nibabies(cli_args, mask, segmentation)
    else:
        run_nibabies(cli_args)

    # Show user how long the pipeline took and end the pipeline here
    exit_with_time_info(start)


def get_cli_args():
    """
    :return: Dictionary containing all command-line arguments from user
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "parameter_json", type=valid_readable_json,
        help=("Valid path to readable parameter .json file. See README.md "
              "for more information on parameters")
        # TODO: Add description of every parameter to the README, and maybe also to this --help message
        # TODO: Add nnU-Net parameters (once they're decided on)
        # TODO: Maaaybe read in each parameter from the .json using argparse if possible? stackoverflow.com/a/61003775
    )
    return extract_from_json(parser.parse_args().parameter_json)


if __name__ == '__main__':
    main()