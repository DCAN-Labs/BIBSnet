#!/usr/bin/env python3
# coding: utf-8

"""
Wrapper to run nnU-Net_predict trained on BCP subjects
Greg Conan: gconan@umn.edu
Created: 2022-02-08
Updated: 2022-10-24
"""
# Import standard libraries
import argparse
from fnmatch import fnmatch
from glob import glob
import os
import pandas as pd
import subprocess
import sys

from src.logger import LOGGER

from src.validate import (
    valid_readable_dir,
    valid_output_dir,
    valid_readable_file,
    valid_whole_number
)

from src.utilities import (
    get_subj_ID_and_session
)


def run_BIBSnet(j_args):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :return: j_args, unchanged
    """    # TODO Test BIBSnet functionality once it's containerized
    sub_ses = get_subj_ID_and_session(j_args)
    dir_BIBS = os.path.join(j_args["optional_out_dirs"]["bibsnet"],
                            *sub_ses, "{}put")
    
    # TODO Change overwrite=False to skip=True in param files because it's more intuitive 
    # Skip BIBSnet if overwrite=False and outputs already exist
    if (not j_args["common"]["overwrite"]) and glob(os.path.join(
        dir_BIBS.format("out"), "*"
    )):
        LOGGER.info("Skipping BIBSnet because outputs already exist at the "
                    "BIBSnet output path below, and --overwrite is off.\n{}"
                    .format(dir_BIBS.format("out")))

    else:  # Run BIBSnet
        try:  # Run BIBSnet
            inputs_BIBSnet = {"model": j_args["bibsnet"]["model"],
                              "nnUNet": j_args["bibsnet"]["nnUNet_predict_path"],
                              "input": dir_BIBS.format("in"),
                              "output": dir_BIBS.format("out"),
                              "task": "{:03d}".format(j_args["ID"]["model"])} #   j_args["bibsnet"]["task"])}
            os.makedirs(inputs_BIBSnet["output"], exist_ok=True)
            if j_args["common"]["verbose"]:
                LOGGER.info("Now running BIBSnet with these parameters:\n{}\n"
                            .format(inputs_BIBSnet))
            run_nnUNet_predict(inputs_BIBSnet)

        except subprocess.CalledProcessError as e:
            # BIBSnet will crash even after correctly creating a segmentation,
            # so only crash CABINET if that segmentation is not made.
            outfpath = os.path.join(dir_BIBS.format("out"),
                                    "{}_optimal_resized.nii.gz"
                                    .format("_".join(sub_ses)))
            if not os.path.exists(outfpath):
                LOGGER.error("BIBSnet failed to create this segmentation "
                             "file...\n{}\n...from these inputs:\n{}"
                             .format(outfpath, inputs_BIBSnet))
                sys.exit(e)

        # Remove unneeded empty directories
        for unneeded_dir_name in ("nnUNet_cropped_data", "nnUNet_raw_data"):
            unneeded_dir_path = os.path.join(
                j_args["optional_out_dirs"]["derivatives"], unneeded_dir_name
            )
            LOGGER.info("Deleting unnecessary empty directory at {}"
                        .format(unneeded_dir_path))
            if os.path.isdir(unneeded_dir_path):
                os.removedirs(unneeded_dir_path)
    
        # TODO hardcoded below call to run_nnUNet_predict. Will likely want to 
        # change and integrate into j_args
        #run_nnUNet_predict({"model": "3d_fullres",
        #                    "nnUNet": "/opt/conda/bin/nnUNet_predict",
        #                    "input": dir_BIBS.format("in"),
        #                    "output": dir_BIBS.format("out"),
        #                    "task": "550"})
        LOGGER.info("BIBSnet has completed")
    return j_args


def run_nnUNet_predict(cli_args):
    """
    Run nnU-Net_predict in a subshell using subprocess
    :param cli_args: Dictionary containing all command-line input arguments
    :return: N/A
    """
    subprocess.call((cli_args["nnUNet"], "-i",
                     cli_args["input"], "-o", cli_args["output"], "-t",
                     str(cli_args["task"]), "-m", cli_args["model"]))
    
    # Only raise an error if there are no output segmentation file(s)
    if not glob(os.path.join(cli_args["output"], "*.nii.gz")):
        # TODO This statement should change if we add a new model
        sys.exit("Error: Output segmentation file not created at the path "
                 "below during nnUNet_predict run.\n{}\n\nFor your input files "
                 "at the path below, check their filenames and visually "
                 "inspect them if needed.\n{}\n\n"
                 .format(cli_args["output"], cli_args["input"]))


def get_cli_args():
    """ 
    :return: Dictionary containing all validated command-line input arguments
    """
    script_dir = os.path.dirname(__file__)
    default_model = "3d_fullres"
    default_nnUNet_path = os.path.join(script_dir, "nnUNet_predict")
    default_task_ID = 512
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=valid_readable_dir, required=True,
        help=("Valid path to existing input directory with 1 T1w file and/or "
              "1 T2w file. T1w files should end with '_0000.nii.gz'. "
              "T2w files only should for T1w-and-T2w model(s). For T2w-only "
              "model(s), T2w files should end with '_0001.nii.gz'.")
    )
    parser.add_argument(
        "--output", "-o", type=valid_output_dir, required=True,
        help=("Valid path to a directory to save BIBSnet output files into. "
              "If this directory or its parent directory/ies does not exist, "
              "then they will automatically be created.")
    )
    parser.add_argument(
        "--nnUNet", "-n", type=valid_readable_file, default=default_nnUNet_path,
        help=("Valid path to existing executable file to run nnU-Net_predict. "
              "By default, this script will assume that nnU-Net_predict will "
              "be in the same directory as this script: {}".format(script_dir))
    )
    parser.add_argument(
        "--task", "-t", type=valid_whole_number, default=default_task_ID,
        help=("Task ID, which should be a 3-digit positive integer starting "
              "with 5 (e.g. 512). The default task ID is {}."
              .format(default_task_ID))
    )
    parser.add_argument(  # TODO Does this even need to be an argument, or will it always be the default?
        "--model", "-m", default=default_model,
        help=("Name of the nnUNet model to run. By default, it will run '{}'."
              .format(default_model))
    )
    return validate_cli_args(vars(parser.parse_args()), script_dir, parser)


def validate_cli_args(cli_args, script_dir, parser):
    """
    Verify that at least 1 T1w and/or 1 T2w file (depending on the task ID)
    exists in the --input directory
    :param cli_args: Dictionary containing all command-line input arguments
    :param script_dir: String, valid path to existing dir containing run.py
    :param parser: argparse.ArgumentParser to raise error if anything's invalid
    :return: cli_args, but with all input arguments validated
    """
    # Get info about which task ID(s) need T1s and which need T2s from .csv
    try:
        models_csv_path = os.path.join(script_dir, "models.csv")  # TODO Should we make this file path an input argument?
        tasks = pd.read_csv(models_csv_path, index_col=0)
        specified_task = tasks.loc[cli_args["task"]]

    # Verify that the specified --task number is a valid task ID
    except OSError:
        parser.error("{} not found. This file is needed to determine nnUNet "
                     "requirements for BIBSnet task {}."
                     .format(models_csv_path, cli_args["task"]))
    except KeyError:
        parser.error("BIBSnet task {0} is not in {1} so its requirements are "
                     "unknown. Add a task {0} row in that .csv or try one of "
                     "these tasks: {2}"
                     .format(cli_args["task"], models_csv_path, 
                             tasks.index.values.tolist()))

    # Validate that BIBSnet has all T1w/T2w input file(s) needed for --task
    err_msg = ("BIBSnet task {} requires image file(s) at the path(s) below, "
               "and at least 1 is missing. Either save the image file(s) "
               "there or try a different task.\n{}")
    img_glob_path = os.path.join(cli_args["input"], "*_000{}.nii.gz")
    how_many_T_expected = 0
    for t1or2 in (1, 2):
        # TODO Should this verify that ONLY one T1w file and/or ONLY one T2w file exists?
        if specified_task.get("T{}w".format(t1or2)):
            how_many_T_expected += 1
    img_files = glob(img_glob_path.format("?"))
    if how_many_T_expected == 2 and len(img_files) < 2:
        parser.error(err_msg.format(cli_args["task"], "\n".join((
            img_glob_path.format(0), img_glob_path.format(1)
        ))))
    elif how_many_T_expected == 1 and (
            len(img_files) < 1 or not fnmatch(img_files[0],
                                              img_glob_path.format(0))
        ):
        parser.error(err_msg.format(cli_args["task"], img_glob_path.format(0)))
        
    return cli_args
