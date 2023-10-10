#!/usr/bin/env python3
# coding: utf-8

"""
Common source for utility functions used by BIBSnet.
Contains functions used by multiple stages, only used in run.py, or called by other utility functions.
Greg Conan: gconan@umn.edu
Created: 2021-11-12
Barry Tikalsky: tikal004@umn.edu
Updated: 2023-10-04
"""
# Import standard libraries
from datetime import datetime
from glob import glob
import numpy as np
import os
import subprocess
import sys

from src.logger import LOGGER

SCRIPT_DIR = os.path.dirname(os.path.dirname(__file__))


# NOTE All functions below are in alphabetical order.

def dict_has(a_dict, a_key):
    """
    :param a_dict: Dictionary (any)
    :param a_key: Object (any)
    :return: True if and only if a_key is mapped to something truthy in a_dict
    """
    return a_key in a_dict and a_dict[a_key]


def ensure_prefixed(label, prefix):
    """ 
    :param label: String to check whether it starts with prefix
    :param prefix: String that should be a substring at the beginning of label
    :return: label, but guaranteed to start with prefix
    """
    return label if label[:len(prefix)] == prefix else prefix + label


def exit_with_time_info(start_time, exit_code=0):
    """
    Terminate the pipeline after displaying a message showing how long it ran
    :param start_time: datetime.datetime object of when the script started
    :param exit_code: exit code
    """
    print("The pipeline for this subject took this long to run {}: {}"
          .format("successfully" if exit_code == 0 else "and then crashed",
                  datetime.now() - start_time))
    sys.exit(exit_code)


def get_optimal_resized_paths(sub_ses, j_args):  # bibsnet_out_dir):
    """
    :param sub_ses: List with either only the subject ID str or the session too
    :param j_args: Dict mapping (A) "optional_out_dirs" to a dict mapping 
                   "bibsnet" to the bibsnet derivatives dir path, and 
                   (B) "ID" to a dict mapping "has_T1w" and "has_T2w" to bools
    :return: Dict mapping "T1w" and "T2w" to their respective optimal (chosen 
             by the cost function) resized (by prebibsnet) image file paths
    """
    input_dir_BIBSnet = os.path.join(j_args["optional_out_dirs"]["bibsnet"],
                                     *sub_ses, "input")
    return {f"T{t}w": os.path.join(input_dir_BIBSnet, 
                "{}_optimal_resized_000{}.nii.gz".format(
                    "_".join(sub_ses),
                    get_preBIBS_final_digit_T(t, j_args["ID"])
                )
            ) for t in only_Ts_needed_for_bibsnet_model(j_args["ID"])}
  

def get_preBIBS_final_digit_T(t, sub_ses_ID):
    """
    :param t: Int, either 1 or 2 (to signify T1w or T2w respectively)
    :param sub_ses_ID: Dictionary mapping subject-session-specific input
                       parameters' names (as strings) to their values for
                       this subject session; the same as j_args[ID]
    :return: Int, the last digit of the preBIBSnet final image filename: 0 or 1
    """
    return (t - 1 if sub_ses_ID["has_T1w"]  
            and sub_ses_ID["has_T2w"] else 0)


def get_stage_name(stage_fn):
    """ 
    :param stage_fn: Function to run one stage of BIBSnet. Its name must start
                     with "run_", e.g. "run_preBIBSnet"
    :return: String naming the BIBSnet stage to run
    """
    return stage_fn.__name__[4:].lower()


def get_subj_ID_and_session(j_args):
    """
    :param j_args: Dictionary containing all args
    :return: List of 2 strings (subject ID and session from parameter file,
             with their correct "sub-" and "ses-" prefixes) if the parameter
             file has a session, otherwise just with the prefixed subject ID
    """ 
    sub = ensure_prefixed(j_args["ID"]["subject"], "sub-")
    return [sub, ensure_prefixed(j_args["ID"]["session"], "ses-")
            ] if dict_has(j_args["ID"], "session") else [sub]


def get_age_closest_to(subject_age, all_ages):
    """
    :param subject_age: Int, participant's actual age in months
    :param all_ages: List of ints, each a potential participant age in months
    :return: Int, the age in all_ages which is closest to subject_age
    """
    return all_ages[np.argmin(np.abs(np.array(all_ages)-subject_age))]


def log_stage_finished(stage_name, event_time, sub_ses):
    """
    Print and return a string showing how much time has passed since the
    current running script reached a certain part of its process
    :param stage_name: String, name of event that just finished
    :param event_time: datetime object representing when {stage_name} started
    :param sub_ses: List with either only the subject ID str or the session too
    :return: String with an easily human-readable message showing how much time
             has passed since {stage_start} when {stage_name} started.
    """
    LOGGER.info("{0} finished on subject {1}. "
                "Time elapsed since {0} started: {2}"
                .format(stage_name, " session ".join(sub_ses),
                        datetime.now() - event_time))


def only_Ts_needed_for_bibsnet_model(sub_ses_ID):
    """
    :param sub_ses_ID: Dictionary mapping subject-session-specific input
                       parameters' names (as strings) to their values for
                       this subject session; the same as j_args[ID]
    :yield: Int, each T value (1 and/or 2) which inputs exist for
    """
    for t in (1, 2):
        if sub_ses_ID[f"has_T{t}w"]:
            yield t


def run_FSL_sh_script(j_args, fsl_fn_name, *fsl_args):
    """
    Run any FSL function in a Bash subprocess, unless its outputs exist and the
    parameter file said not to overwrite outputs
    :param j_args: Dictionary containing all args
    :param fsl_fn_name: String naming the FSL function which is an
                        executable file in j_args[common][fsl_bin_path]
    """
    # TODO Run FSL commands using the Python fsl.ImageMaths /etc functions instead of subprocess

    # FSL command to (maybe) run in a subprocess
    to_run = [os.path.join(j_args["common"]["fsl_bin_path"], fsl_fn_name)
              ] + [str(f) for f in fsl_args]

    # If the output image(s) exist(s) and j_args[common][overwrite] is False,
    # then skip the entire FSL command and tell the user
    outputs = list()
    skip_cmd = False
    if not j_args["common"]["overwrite"]:
        for i in range(len(to_run)):
            if to_run[i].strip('-') in ("o", "omat", "out", "m"):  # -m to skip robustFOV
                outputs.append(to_run[i + 1])
        if outputs and all([os.path.exists(output) for output in outputs]):
            skip_cmd = True
    if skip_cmd:
        if j_args["common"]["verbose"]:
            LOGGER.info("Skipping FSL {} command because its output image(s) "
                        "listed below exist(s) and overwrite=False.\n{}"
                        .format(fsl_fn_name, "\n".join(outputs)))

    # Otherwise, just run the FSL command
    else:
        if j_args["common"]["verbose"]:
            LOGGER.info("Now running FSL command:\n{}"
                        .format(" ".join(to_run)))
        subprocess.check_call(to_run)

    # pdb.set_trace()  # TODO Add "debug" flag?


def run_all_stages(all_stages, sub_ses_IDs, start, end,
                   ubiquitous_j_args):
    """
    Run stages sequentially, starting and ending at stages specified by user
    :param all_stages: List of functions in order where each runs one stage
    :param sub_ses_IDs: List of dicts mapping "age_months", "subject",
                        "session", etc. to unique values per subject session
    :param start: String naming the first stage the user wants to run
    :param end: String naming the last stage the user wants to run
    :param ubiquitous_j_args: Dictionary of all args needed by each stage
    """
    if ubiquitous_j_args["common"]["verbose"]:
        LOGGER.info("All parameters from input args:\n{}"
                    .format(ubiquitous_j_args))

    # For every session of every subject...
    running = False
    for dict_with_IDs in sub_ses_IDs:

        # ...make a j_args copy with its subject ID, session ID, and age 
        sub_ses_j_args = ubiquitous_j_args.copy()
        sub_ses_j_args["ID"] = dict_with_IDs
        sub_ses = get_subj_ID_and_session(sub_ses_j_args)
        sub_ses_j_args["optimal_resized"] = get_optimal_resized_paths(
            sub_ses, sub_ses_j_args # ubiquitous_j_args["optional_out_dirs"]["bibsnet"]
        )

        # ...check that all required input files exist for the stages to run
        verify_BIBSnet_inputs_exist(sub_ses, sub_ses_j_args)

        # ...run all stages that the user said to run
        for stage in all_stages:
            name = get_stage_name(stage)
            if name == start:
                running = True
            if running:
                stage_start = datetime.now()
                if sub_ses_j_args["common"]["verbose"]:
                    LOGGER.info("Now running {} stage on:\n{}"
                                .format(name, sub_ses_j_args["ID"]))
                sub_ses_j_args = stage(sub_ses_j_args)
                log_stage_finished(name, stage_start, sub_ses)
            if name == end:
                running = False


def split_2_exts(a_path):
    """
    :param path: String, a file path with two extensions (like ".dscalar.nii")
    :return: Tuple of 2 strings, the extensionless path and the 2 extensions
    """
    base, ext2 = os.path.splitext(a_path)
    base, ext1 = os.path.splitext(base)
    return base, ext1 + ext2


def verify_BIBSnet_inputs_exist(sub_ses, j_args):
    """
    Given a stage, verify that all of the necessary inputs for that stage exist 
    :param a_stage: String naming a stage
    :param sub_ses: List with either only the subject ID str or the session too
    :param j_args: Dictionary containing all args
    """
    # Define globbable paths to prereq files for the script to check
    out_BIBSnet_seg = os.path.join(j_args["optional_out_dirs"]["bibsnet"],
                                   *sub_ses, "output", "*.nii.gz")
    all_T_suffixes = ["0000"]
    if j_args["ID"]["has_T1w"] and j_args["ID"]["has_T2w"]:
        all_T_suffixes.append("0001") # Only check for _0001 file for T1-and-T2
    subject_heads = [os.path.join(
            j_args["optional_out_dirs"]["bibsnet"], *sub_ses, "input",
            "*{}*_{}.nii.gz".format("_".join(sub_ses), suffix_T) 
        ) for suffix_T in all_T_suffixes] 
    out_paths_BIBSnet = [os.path.join(j_args["optional_out_dirs"]["bibsnet"],
                                      "*{}*.nii.gz".format(x))
                         for x in ("aseg", "mask")]

    # Map each stage's name to its required input files
    stage_prerequisites = {"prebibsnet": list(),
                           "bibsnet": list(j_args["optimal_resized"].values()),
                           "postbibsnet": [out_BIBSnet_seg, *subject_heads]}

    # For each stage that will be run, verify that its prereq input files exist
    all_stages = [s for s in stage_prerequisites.keys()]
    start_ix = all_stages.index(j_args["stage_names"]["start"]) 
    for stage in all_stages[:start_ix+1]:
        missing_files = list()
        for globbable in stage_prerequisites[stage]:
            if not glob(globbable):
                missing_files.append(globbable)
        if missing_files:
            LOGGER.error("The file(s) below are needed to run the {} stage, "
                        "but they do not exist.\n{}\n"
                        .format(stage, "\n".join(missing_files)))
            sys.exit(1)

    LOGGER.info("All required input files exist.")

