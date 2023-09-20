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
from glob import glob
import logging
import math
from nipype.interfaces import fsl
import os
import pandas as pd
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


# Global constants: Paths to this dir and to level 1 analysis script
SCRIPT_DIR_ARG = "--script-dir"
SCRIPT_DIR = find_myself(SCRIPT_DIR_ARG)
AGE_TO_HEAD_RADIUS_TABLE = os.path.join(SCRIPT_DIR, "data",
                                        "age_to_avg_head_radius_BCP.csv")
LR_REGISTR_PATH = os.path.join(SCRIPT_DIR, "bin", "LR_mask_registration.sh")
TYPES_JSON = os.path.join(SCRIPT_DIR, "src", "param-types.json")

# Custom local imports
from src.utilities import (
    apply_final_prebibsnet_xfms, as_cli_attr, as_cli_arg, correct_chirality, 
    create_anatomical_averages, crop_image, dilate_LR_mask, ensure_prefixed,
    exit_with_time_info, extract_from_json, generate_sidecar_json,
    get_age_closest_to, get_and_make_preBIBSnet_work_dirs, get_optional_args_in,
    get_preBIBS_final_img_fpath_T, get_stage_name, get_subj_ID_and_session,
    get_template_age_closest_to, make_given_or_default_dir,
    only_Ts_needed_for_bibsnet_model, register_preBIBSnet_imgs_ACPC, 
    register_preBIBSnet_imgs_non_ACPC, remove_extra_clusters_from_mask, reverse_regn_revert_to_native,
    run_FSL_sh_script, run_all_stages, valid_output_dir, valid_readable_json,
    validate_parameter_types, valid_readable_dir,
    valid_subj_ses_ID, valid_whole_number
)


def main():
    start_time = datetime.now()  # Time how long the script takes
    logger = make_logger()  # Make object to log error/warning/status messages

    # Get and validate command-line arguments and parameters from .JSON file
    STAGES = [run_preBIBSnet, run_BIBSnet, run_postBIBSnet, run_nibabies,
              run_XCPD]
    json_args, sub_ses_IDs = get_params_from_JSON([get_stage_name(stg) for stg
                                                   in STAGES], logger)  # TODO Un-capitalize "BIBS" everywhere (except import BIBSnet.run?)

    # Set output dir environment variable for BIBSnet to user-defined output dir
    os.environ["nnUNet_raw_data_base"] = json_args["optional_out_dirs"]["derivatives"]
    
    # Run every stage that the parameter file says to run
    run_all_stages(STAGES, sub_ses_IDs, json_args["stage_names"]["start"],
                   json_args["stage_names"]["end"], json_args, logger)
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


def get_params_from_JSON(stage_names, logger):
    """
    :param stage_names: List of strings; each names a stage to run
    :return: Dictionary containing all parameters from parameter .JSON file
    """
    default_end_stage = "postbibsnet"  # TODO Change to stage_names[-1] once nibabies and XCPD run from CABINET
    msg_stage = ("Name of the stage to run {}. By default, this will be "
                 "the {} stage. Valid choices: {}")
    parser = argparse.ArgumentParser("CABINET")
    # TODO will want to add positional 'input' and 'output' arguments and '--participant-label' and '--session-label' arguments. For the HBCD study, we won't to have to create a JSON per scanning session, but this will likely be fine for the pilot.

    # BIDS-App required positional args, validated later in j_args
    parser.add_argument(
        "bids_dir", type=valid_readable_dir,
        help=("Valid absolute path to existing base study directory "
              "containing BIDS-valid input subject data directories. "
              "Example: /path/to/bids/input/")  # TODO Keep as j_args[common][bids_dir]
    )
    parser.add_argument(
        "output_dir", type=valid_readable_dir,  # TODO Does this dir have to already exist?
        help=("Valid absolute path to existing derivatives directory to save "
              "each stage's outputs by subject session into. Example: "
              "/path/to/output/derivatives/")
    )
    parser.add_argument(
        "analysis_level", choices=["participant"],  # TODO Will we ever need to add group-level analysis functionality? Currently this argument does absolutely nothing
        help=("Processing level. Currently the only choice is 'participant'. "
              "See BIDS-Apps specification.")
    )

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
    parser.add_argument(
        "-participant", "--subject", "-sub", "--participant-label",
        dest="participant_label", type=valid_subj_ses_ID,
        help=("The participant's unique subject identifier, without 'sub-' "
              "prefix. Example: 'ABC12345'")  # TODO Make CABINET able to accept with OR without 'sub-' prefix
    )

    # Optional flag arguments
    parser.add_argument(
        "-age", "-months", "--age-months", type=valid_whole_number,
        help=("Positive integer, the participant's age in months. For "
              "example, -age 5 would mean the participant is 5 months old. "
              "Include this argument unless the age in months is specified in "
              "each subject's sub-{}_sessions.tsv file inside its BIDS input directory "
              "or inside the participants.tsv file inside the BIDS directory at the" 
              "subject-level.")
    )
    parser.add_argument(
        "-end", "--ending-stage", dest="end",
        choices=stage_names[:3], default=default_end_stage,  # TODO change to choices=stage_names,
        help=msg_stage.format("last", default_end_stage, ", ".join(stage_names[:3]))
    )
    parser.add_argument(
        "-model", "--model-number", "--bibsnet-model",
        type=valid_whole_number, dest="model",
        help=("Model/task number for BIBSnet. By default, this will be "
              "inferred from {} based on which data exists in the "
              "--bids-dir. BIBSnet will run model 514 by default for T1w-"
              "only, model 515 for T2w-only, and model 552 for both T1w and "
              "T2w.".format(os.path.join(SCRIPT_DIR, "data", "models.csv")))
    )
    parser.add_argument(
        "--overwrite", "--overwrite-old",  # TODO Change this to "-skip"
        dest="overwrite", action="store_true",
        help=("Include this flag to overwrite any previous CABINET outputs "
              "in the derivatives sub-directories. Otherwise, by default "
              "CABINET will skip creating any CABINET output files that "
              "already exist in the sub-directories of derivatives.")
    )
    parser.add_argument(
        "-ses", "--session", "--session-id", type=valid_subj_ses_ID,
        help=("The name of the session to processes participant data for. "
              "Example: baseline_year1")
    )
    parser.add_argument(
        "-start", "--starting-stage", dest="start",
        choices=stage_names[:3], default=stage_names[0],   # TODO Change default to start where we left off by checking which stages' prerequisites and outputs already exist
        help=msg_stage.format("first", stage_names[0], ", ".join(stage_names[:3]))  # TODO Change to include all stage names; right now it just includes the segmentation pipeline
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help=("Include this flag to print detailed information and every "
              "command being run by CABINET to stdout. Otherwise CABINET "
              "will only print warnings, errors, and minimal output.")
    )
    parser.add_argument(
        "-w", "--work-dir", type=valid_output_dir, dest="work_dir",
        default=os.path.join("/", "tmp", "cabinet"),
        help=("Valid absolute path where intermediate results should be stored."
              "Example: /path/to/working/directory")
    )
    parser.add_argument(
        "-z", "--brain-z-size", action="store_true",
        help=("Include this flag to infer participants' brain height (z) "
              "using the sub-{}_sessions.tsv or participant.tsv brain_z_size column." 
              "Otherwise, CABINET will estimate the brain height from the participant "
              "age and averages of a large sample of infant brain heights.")  # TODO rephrase
    )
    parser.add_argument(
        SCRIPT_DIR_ARG, dest=as_cli_attr(SCRIPT_DIR_ARG),
        type=valid_readable_dir,
        help=("Valid path to the existing parent directory of this run.py "
              "script. Include this argument if and only if you are running "
              "the script as a SLURM/SBATCH job.")
    )
    return validate_cli_args(vars(parser.parse_args()), stage_names,
                             parser, logger)


def validate_cli_args(cli_args, stage_names, parser, logger):
    """
    :param cli_args: Dictionary containing all command-line input arguments
    :param stage_names: List of strings naming stages to run (in order)
    :param logger: logging.Logger object to show messages and raise warnings
    :param parser: argparse.ArgumentParser to raise error if anything's invalid
    :return: Tuple of 2 objects:
        1. Dictionary of validated parameters from parameter .JSON file
        2. List of dicts which each map "subject" to the subject ID string,
           "age_months" to the age in months (int) during the session, & maybe
           also "session" to the session ID string. Each will be j_args[IDs]
    """
    # Get command-line input arguments and use them to get .JSON parameters
    j_args = extract_from_json(cli_args["parameter_json"])
    script_dir_attr = as_cli_attr(SCRIPT_DIR_ARG)
    j_args["meta"] = {script_dir_attr: SCRIPT_DIR,
                      "slurm": bool(cli_args[script_dir_attr])}

    # Crash immediately if the end is given as a stage that happens before start
    if (stage_names.index(cli_args["start"])
            > stage_names.index(cli_args["end"])):
        parser.error("Error: {} stage must happen before {} stage."
                     .format(cli_args["start"], cli_args["end"]))

    # Add command-line arguments to j_args
    j_args["stage_names"] = {"start": cli_args["start"],
                             "end": cli_args["end"]}  # TODO Maybe save the stage_names list in here too to replace optional_out_dirs use cases?
    for arg_to_add in ("bids_dir", "overwrite", "verbose", "work_dir"):
        j_args["common"][arg_to_add] = cli_args[arg_to_add]

    # TODO Remove all references to the optional_out_dirs arguments, and change
    #      j_args[optional_out_dirs][derivatives] to instead be j_args[common][output_dir]
    j_args["optional_out_dirs"] = {stagename: None for stagename in stage_names} 
    j_args["optional_out_dirs"]["derivatives"] = cli_args["output_dir"]

    # TODO Automatically assign j_args[common][fsl_bin_path] and 
    #      j_args[bibsnet][nnUNet_predict_path] if in the Singularity container

    # TODO Check whether the script is being run from a container...
    # - Docker: stackoverflow.com/a/25518538
    # - Singularity: stackoverflow.com/a/71554776
    # ...and if it is, then assign default values to these j_args, overwriting user's values:
    # j_args[common][fsl_bin_path]
    # j_args[BIBSnet][nnUNet_predict_path]
    # j_args[BIBSnet][code_dir]
    # Also, check whether the Docker/Singularity environment variables show up in os.environ for us to use here
    # print(vars(os.environ))

    # Define (and create) default paths in derivatives directory structure for 
    # each stage of each session of each subject
    sub_ses_IDs = get_all_sub_ses_IDs(j_args, cli_args["participant_label"],
                                      cli_args["session"])  # TODO Add brain_z_size into j_args[ID]
    
    # TODO Iff the user specifies a session, then let them specify an age
    default_derivs_dir = os.path.join(j_args["common"]["bids_dir"], "derivatives")
    for ix in range(len(sub_ses_IDs)):
        # Create a list with the subject ID and (if it exists) the session ID
        sub_ses = [sub_ses_IDs[ix]["subject"]]
        if sub_ses_IDs[ix].get("session"): 
            sub_ses.append(sub_ses_IDs[ix]["session"])

        j_args = ensure_j_args_has_bids_subdirs(
            j_args, stage_names, sub_ses, default_derivs_dir
        )

        # Verify that subject-session BIDS/input directory path exists
        sub_ses_dir = os.path.join(j_args["common"]["bids_dir"], *sub_ses)
        if not os.path.exists(sub_ses_dir):
            parser.error("No subject/session directory found at {}\nCheck "
                         "that your participant_label and session are correct."
                         .format(sub_ses_dir))
    
        # User only needs sessions.tsv if they didn't specify age_months
        if not j_args["common"].get("age_months"): 
            sub_ses_IDs[ix]["age_months"] = read_from_tsv(
                j_args, logger, "age", *sub_ses
            )
        
        # Infer brain_z_size for this sub_ses using sessions.tsv if the 
        # user said to (by using --brain-z-size flag), otherwise infer it 
        # using age_months and the age-to-head-radius table .csv file
        sub_ses_IDs[ix]["brain_z_size"] = read_from_tsv(
                j_args, logger, "brain_z_size", *sub_ses
            ) if cli_args["brain_z_size"] else get_brain_z_size(
                sub_ses_IDs[ix]["age_months"], j_args, logger
            )

        # TODO Check if this adds more sub_ses-dependent stuff to j_args
        # Verify that every parameter in the parameter .JSON file is a valid input
        j_args = validate_parameter_types(
            j_args, extract_from_json(TYPES_JSON),
            cli_args["parameter_json"], parser, stage_names
        )

        # Check whether this sub ses has T1w and/or T2w input data
        data_path_BIDS_T = dict()  # Paths to expected input data to check
        for t in (1, 2):
            data_path_BIDS_T[t] = os.path.join(j_args["common"]["bids_dir"],
                                               *sub_ses, "anat",
                                               f"*T{t}w.nii.gz")
            sub_ses_IDs[ix][f"has_T{t}w"] = bool(glob(data_path_BIDS_T[t]))

        models_df = get_df_with_valid_bibsnet_models(sub_ses_IDs[ix])
        sub_ses_IDs[ix]["model"] = validate_model_num(
            cli_args, data_path_BIDS_T, models_df, sub_ses_IDs[ix], parser
        )

        # Create BIBSnet in/out directories
        dir_BIBSnet = dict()
        for io in ("in", "out"):
            dir_BIBSnet[io] = os.path.join(j_args["optional_out_dirs"]["bibsnet"],
                                           *sub_ses, f"{io}put")
            os.makedirs(dir_BIBSnet[io], exist_ok=True)

    if j_args["common"]["verbose"]:
        logger.info(" ".join(sys.argv[:]))  # Print all

    # 2. roi2full for preBIBSnet and postBIBSnet transformation
    # j_args["xfm"]["roi2full"] =   # TODO
    return j_args, sub_ses_IDs


def get_df_with_valid_bibsnet_models(sub_ses_ID):
    """
    :param sub_ses_ID: Dictionary mapping subject-session-specific input
                       parameters' names (as strings) to their values for
                       this subject session; the same as j_args[ID]
    :return: pandas.DataFrame of all bibsnet models viable for the input data
    """
    # Read in models.csv info mapping model num to which T(s) it has
    models_df = pd.read_csv(os.path.join(SCRIPT_DIR, "data", "models.csv"))

    # Exclude any models which require (T1w or T2w) data the user lacks
    for t in only_Ts_needed_for_bibsnet_model(sub_ses_ID):
        models_df = select_model_with_data_for_T(
            t, models_df, sub_ses_ID[f"has_T{t}w"]
        )
    return models_df


def validate_model_num(cli_args, data_path_BIDS_T, models_df, sub_ses_ID, parser):
    """
    :param cli_args: Dictionary containing all command-line input arguments
    :param data_path_BIDS_T: Dictionary mapping 1 and 2 to the (incomplete)
                             paths to expected T1w and T2w data respectively
    :param models_df: pd.DataFrame of all bibsnet models viable for input data
    :param sub_ses_ID: Dict mapping (string) names to values for sub- &
                       ses-specific input parameters; same as j_args[ID]
    :param parser: argparse.ArgumentParser to raise error if anything's invalid
    :return: Int, validated bibsnet model number
    """
    model = cli_args["model"]  # Model number (if given from command line)

    # Exclude any models which require (T1w or T2w) data the user lacks
    for t in (1, 2):

        # If user gave a model number but not the data the model needs,
        # then crash with an informative error message
        if model and (model not in models_df["model_num"]):
            parser.error("CABINET needs T{}w data at the path below " 
                            "to run model {}, but none was found.\n{}\n"
                            .format(t, model, data_path_BIDS_T[t]))

    if not model:  # Get default model number if user did not give one
        models_df = models_df[models_df["is_default"]]
        if len(models_df) > 1:
            for t in (1, 2):
                models_df = select_model_with_data_for_T(
                    t, models_df, sub_ses_ID[f"has_T{t}w"]
                )
        model = models_df.squeeze()["model_num"]
            
    return model


def select_model_with_data_for_T(t, models_df, has_T):
    """
    :param t: Int, either 1 or 2 (to signify T1w or T2w respectively)
    :param models_df: pandas.DataFrame with columns called "T1w" and "T2w"
                      with bool values describing which T(s) a model needs
    :param has_T: bool, True if T{t}w data exists for this subject/ses
    :return: pandas.DataFrame, all models_df rows with data for this sub/ses/t
    """
    has_T_row = models_df[f"T{t}w"]
    return models_df.loc[has_T_row if has_T else ~has_T_row]


def get_brain_z_size(age_months, j_args, logger, buffer=5):
    """ 
    Infer a participant's brain z-size from their age and from the average
    brain diameters table at the AGE_TO_HEAD_RADIUS_TABLE path
    :param age_months: Int, participant's age in months
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :param buffer: Int, extra space (in mm), defaults to 5
    :return: Int, the brain z-size (height) in millimeters
    """
    MM_PER_IN = 25.4  # Conversion factor: inches to millimeters

    # Other columns' names in the age-to-head-radius table
    age_months_col = "Candidate_Age(mo.)"
    head_r_col = "Head_Radius(in.)"
    head_diam_mm = "head_diameter_mm"

    # Get table mapping each age in months to average head radius
    age2headradius = pd.read_csv(AGE_TO_HEAD_RADIUS_TABLE)

    # Get BCP age (in months) closest to the subject's age
    closest_age = get_age_closest_to(age_months, age2headradius[age_months_col])
    if j_args["common"]["verbose"]:
        logger.info(f"Subject age in months: {age_months}\nClosest BCP age in "
                    f"months in age-to-head-radius table: {closest_age}")

    # Get average head radii in millimeters by age from table
    age2headradius[head_diam_mm] = age2headradius[head_r_col
                                                  ] * MM_PER_IN * 2
    row = age2headradius[age2headradius[age_months_col] == closest_age]
    
    # Return the average brain z-size for the participant's age
    return math.ceil(row.get(head_diam_mm)) + buffer


def get_all_sub_ses_IDs(j_args, subj_or_none, ses_or_none):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param subj_or_none: String (the subject ID) or a falsey value
    :param ses_or_none: String (the session name) or a falsey value
    :return: List of dicts; each dict maps "subject" to its subject ID string
             and may also map "session" to its session ID string
    """
    sub_ses_IDs = list()

    # Find all subject-session directories in tree under bids_dir
    sub_match = "sub-{}".format(subj_or_none.split("sub-")[-1] if
                                subj_or_none else "*")
    ses_match = "ses-{}".format(ses_or_none.split("ses-")[-1] if
                                ses_or_none else "*")
    sub_ses_dirs = glob(os.path.join(j_args["common"]["bids_dir"],
                        sub_match, ses_match))

    # If there are subjects and sessions, then add each pair to a list
    if sub_ses_dirs:
        for sub_ses_dirpath in sub_ses_dirs:
            sub_dirpath, session = os.path.split(sub_ses_dirpath)
            _, subject = os.path.split(sub_dirpath)
            sub_ses_IDs.append({"subject": subject, "session": session})

    # Otherwise, make a list that only has subject IDs
    else:
        for sub_dirpath in glob(os.path.join(j_args["common"]["bids_dir"],
                                sub_match)):
            sub_ses_IDs.append({"subject": os.path.basename(sub_dirpath)})

    return sub_ses_IDs


def ensure_j_args_has_bids_subdirs(j_args, derivs, sub_ses, default_parent):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param derivs: Unpacked list of strings. Each names 1 part of a path under
                   j_args[common][bids_dir]. The last string is mapped by
                   j_args[optional_out_dirs] to the subdir path.
    :param sub_ses: List with either only the subject ID str or the session too
    :param default_parent: The default parent directory where all output
                   directories will be placed.
    :return: j_args, but with the (now-existing) subdirectory path
    """

    j_args["optional_out_dirs"] = make_given_or_default_dir(
        j_args["optional_out_dirs"], "derivatives", default_parent
    )
    work_dir_list = ["prebibsnet", "bibsnet", "postbibsnet"]
    for deriv in derivs:
        subdir_path = os.path.join(j_args["common"]["work_dir"], deriv) if deriv in work_dir_list else os.path.join(
                j_args["optional_out_dirs"]["derivatives"], deriv)
        j_args["optional_out_dirs"] = make_given_or_default_dir(
            j_args["optional_out_dirs"], deriv, subdir_path
        )
        os.makedirs(os.path.join(j_args["optional_out_dirs"][deriv], *sub_ses),
                    exist_ok=True)  # Make all subject-session output dirs
    return j_args


def read_from_tsv(j_args, logger, col_name, *sub_ses):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :param col_name: String naming the column of sessions.tsv to return
                     a value from (for this subject or subject-session)
    :param sub_ses: Tuple containing subject and session labels. 
    :return: Int, either the subject's age (in months) or the subject's
             brain_z_size (depending on col_name) as listed in sessions.tsv
    """

    session_tsv_path = os.path.join(j_args["common"]["bids_dir"], sub_ses[0],
                     "{}_sessions.tsv".format(sub_ses[0]))
    participant_tsv_path = os.path.join(j_args["common"]["bids_dir"],
                     "participants.tsv")

    ID_col = "session_id" if os.path.exists(session_tsv_path) else "participant_id"

    tsv_path = session_tsv_path if ID_col == "session_id" else participant_tsv_path

    tsv_df = pd.read_csv(
        tsv_path, delim_whitespace=True, index_col=ID_col
    )
    # Check if column name exists in either tsv, grab the value if column name exists
    try:
        if col_name not in tsv_df.columns:
            raise ValueError("Did not find {} in {}".format(col_name, tsv_path))
        else:
            col_value = get_col_value_from_tsv(j_args, logger, tsv_df, ID_col, col_name, sub_ses)
    except ValueError as exception:
        logger.info(exception)
        if ID_col == "participant_id":
            pass 
        else:
            ID_col = "participant_id"
            tsv_path = participant_tsv_path
            tsv_df = pd.read_csv(
                tsv_path, delim_whitespace=True, index_col=ID_col
            )
            if col_name not in tsv_df.columns:
                raise ValueError("Did not find {} in {}".format(col_name, tsv_path))
            else:
                col_value = get_col_value_from_tsv(j_args, logger, tsv_df, ID_col, col_name, sub_ses)
        
    return col_value

def get_col_value_from_tsv(j_args, logger, tsv_df, ID_col, col_name, sub_ses):
    # Get and return the col_name value from sessions.tsv

    subj_row = tsv_df.loc[
        ensure_prefixed(sub_ses[1], "ses-") if ID_col == "session_id" else ensure_prefixed(sub_ses[0], "sub-")
    ]  # select where "participant_id" matches
    if j_args["common"]["verbose"]:
        logger.info(f"ID_col used to get details from tsv: {ID_col}")
        logger.info(f"Subject details from tsv row:\n{subj_row}")
    return int(subj_row[col_name])

def run_preBIBSnet(j_args, logger):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return: j_args, but with preBIBSnet working directory names added
    """
    completion_msg = "The anatomical images have been {} for use in BIBSnet"
    preBIBSnet_paths = get_and_make_preBIBSnet_work_dirs(j_args)
    sub_ses = get_subj_ID_and_session(j_args)

    # If there are multiple T1ws/T2ws, then average them
    create_anatomical_averages(preBIBSnet_paths["avg"], logger)  # TODO make averaging optional with later BIBSnet model?

    # Crop T1w and T2w images
    cropped = dict()
    crop2full = dict()
    for t in only_Ts_needed_for_bibsnet_model(j_args["ID"]):
        cropped[t] = preBIBSnet_paths[f"crop_T{t}w"]
        crop2full[t] = crop_image(preBIBSnet_paths["avg"][f"T{t}w_avg"],
                                  cropped[t], j_args, logger)
    logger.info(completion_msg.format("cropped"))

    # Resize T1w and T2w images if running a BIBSnet model using T1w and T2w
    # TODO Make ref_img an input parameter if someone wants a different reference image?
    # TODO Pipeline should verify that reference_img files exist before running
    reference_img = os.path.join(SCRIPT_DIR, "data", "MNI_templates",
                                 "INFANT_MNI_T{}_1mm.nii.gz") 
    id_mx = os.path.join(SCRIPT_DIR, "data", "identity_matrix.mat")
    # TODO Resolution is hardcoded; infer it or get it from the command-line
    resolution = "1"  
    if j_args["ID"]["has_T1w"] and j_args["ID"]["has_T2w"]:
        msg_xfm = "Arguments for {}ACPC image transformation:\n{}"

        # Non-ACPC
        regn_non_ACPC = register_preBIBSnet_imgs_non_ACPC(
            cropped, preBIBSnet_paths["resized"], reference_img, 
            id_mx, resolution, j_args, logger
        )
        if j_args["common"]["verbose"]:
            logger.info(msg_xfm.format("non-", regn_non_ACPC["vars"]))

        # ACPC
        regn_ACPC = register_preBIBSnet_imgs_ACPC(
            cropped, preBIBSnet_paths["resized"], regn_non_ACPC["vars"],
            crop2full, preBIBSnet_paths["avg"], j_args, logger
        )
        if j_args["common"]["verbose"]:
            logger.info(msg_xfm.format("", regn_ACPC["vars"]))

        transformed_images = apply_final_prebibsnet_xfms(
            regn_non_ACPC, regn_ACPC, preBIBSnet_paths["avg"], j_args, logger
        )
        logger.info(completion_msg.format("resized"))

    # If running a T1w-only or T2w-only BIBSnet model, skip registration/resizing
    else:
        # Define variables and paths needed for the final (only) xfm needed
        t1or2 = 1 if j_args["ID"]["has_T1w"] else 2
        outdir = os.path.join(preBIBSnet_paths["resized"], "xfms")
        os.makedirs(outdir, exist_ok=True)
        out_img = get_preBIBS_final_img_fpath_T(t1or2, outdir, j_args["ID"])
        crop2BIBS_mat = os.path.join(outdir,
                                     "crop2BIBS_T{}w_only.mat".format(t1or2))
        out_mat = os.path.join(outdir, "full_crop_T{}w_to_BIBS_template.mat"
                                       .format(t1or2))

        run_FSL_sh_script(  # Get xfm moving the T1 (or T2) into BIBS space
            j_args, logger, "flirt", "-in", cropped[t1or2],
            "-ref", reference_img.format(t1or2), "-applyisoxfm", resolution,
            "-init", id_mx, # TODO Should this be a matrix that does a transformation?
            "-omat", crop2BIBS_mat
        )

        # Invert crop2full to get full2crop
        # TODO Move this to right after making crop2full, then delete the 
        #      duplicated functionality in align_ACPC_1_image
        full2crop = os.path.join(
            os.path.dirname(preBIBSnet_paths["avg"][f"T{t}w_avg"]),
            f"full2crop_T{t}w_only.mat"
        )
        run_FSL_sh_script(j_args, logger, "convert_xfm", "-inverse",
                          crop2full[t], "-omat", full2crop) 

        # - Concatenate crop .mat to out_mat (in that order) and apply the
        #   concatenated .mat to the averaged image as the output
        # - Treat that concatenated output .mat as the output to pass
        #   along to postBIBSnet, and the image output to BIBSnet
        run_FSL_sh_script(  # Combine ACPC-alignment with robustFOV output
            j_args, logger, "convert_xfm", "-omat", out_mat,
            "-concat", full2crop, crop2BIBS_mat
        )
        run_FSL_sh_script(  # Apply concat xfm to crop and move into BIBS space
            j_args, logger, "applywarp", "--rel", "--interp=spline",
            "-i", preBIBSnet_paths["avg"][f"T{t}w_avg"],
            "-r", reference_img.format(t1or2),
            "--premat=" + out_mat, "-o", out_img
        )
        transformed_images = {f"T{t1or2}w": out_img,
                              f"T{t1or2}w_crop2BIBS_mat": out_mat}

    # TODO Copy this whole block to postBIBSnet, so it copies everything it needs first
    # Copy preBIBSnet outputs into BIBSnet input dir
    for t in only_Ts_needed_for_bibsnet_model(j_args["ID"]): 
        # Copy image files
        out_nii_fpath = j_args["optimal_resized"][f"T{t}w"]
        os.makedirs(os.path.dirname(out_nii_fpath), exist_ok=True)
        if j_args["common"]["overwrite"]:  # TODO Should --overwrite delete old image file(s)?
            os.remove(out_nii_fpath)
        if not os.path.exists(out_nii_fpath): 
            shutil.copy2(transformed_images[f"T{t}w"], out_nii_fpath)

        # Copy .mat into postbibsnet dir with the same name regardless of which
        # is chosen, so postBIBSnet can use the correct/chosen .mat file
        concat_mat = transformed_images[f"T{t}w_crop2BIBS_mat"]
        out_mat_fpath = os.path.join(  # TODO Pass this in (or out) from the beginning so we don't have to build the path twice (once here and once in postBIBSnet)
            j_args["optional_out_dirs"]["postbibsnet"],
            *sub_ses, "preBIBSnet_" + os.path.basename(concat_mat)
        )
        if not os.path.exists(out_mat_fpath):
            shutil.copy2(concat_mat, out_mat_fpath)
            if j_args["common"]["verbose"]:
                logger.info(f"Copying {concat_mat} to {out_mat_fpath}")
    logger.info("PreBIBSnet has completed")
    return j_args


def run_BIBSnet(j_args, logger):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
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
        logger.info("Skipping BIBSnet because outputs already exist at the "
                    "BIBSnet output path below, and --overwrite is off.\n{}"
                    .format(dir_BIBS.format("out")))

    else:  # Run BIBSnet
        # Import BIBSnet functionality from BIBSnet/run.py
        parent_BIBSnet = os.path.dirname(j_args["bibsnet"]["code_dir"])
        logger.info(f"Importing BIBSnet from {parent_BIBSnet}")
        sys.path.append(parent_BIBSnet)  #sys.path.append("/home/cabinet/SW/BIBSnet")
        from BIBSnet.run import run_nnUNet_predict
               
        # TODO test functionality of importing BIBSNet function via params json (j_args)

        try:  # Run BIBSnet
            inputs_BIBSnet = {"model": j_args["bibsnet"]["model"],
                              "nnUNet": j_args["bibsnet"]["nnUNet_predict_path"],
                              "input": dir_BIBS.format("in"),
                              "output": dir_BIBS.format("out"),
                              "task": "{:03d}".format(j_args["ID"]["model"])} #   j_args["bibsnet"]["task"])}
            os.makedirs(inputs_BIBSnet["output"], exist_ok=True)
            if j_args["common"]["verbose"]:
                logger.info("Now running BIBSnet with these parameters:\n{}\n"
                            .format(inputs_BIBSnet))
            run_nnUNet_predict(inputs_BIBSnet)

        except subprocess.CalledProcessError as e:
            # BIBSnet will crash even after correctly creating a segmentation,
            # so only crash CABINET if that segmentation is not made.
            outfpath = os.path.join(dir_BIBS.format("out"),
                                    "{}_optimal_resized.nii.gz"
                                    .format("_".join(sub_ses)))
            if not os.path.exists(outfpath):
                logger.error("BIBSnet failed to create this segmentation "
                             "file...\n{}\n...from these inputs:\n{}"
                             .format(outfpath, inputs_BIBSnet))
                sys.exit(e)

        # Remove unneeded empty directories
        for unneeded_dir_name in ("nnUNet_cropped_data", "nnUNet_raw_data"):
            unneeded_dir_path = os.path.join(
                j_args["optional_out_dirs"]["derivatives"], unneeded_dir_name
            )
            logger.info("Deleting unnecessary empty directory at {}"
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
        logger.info("BIBSnet has completed")
    return j_args


def run_postBIBSnet(j_args, logger):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return: j_args, unchanged
    """
    sub_ses = get_subj_ID_and_session(j_args)

    # Template selection values
    age_months = j_args["ID"]["age_months"]
    logger.info("Age of participant: {} months".format(age_months))

    # Get template closest to age
    tmpl_age = get_template_age_closest_to(
        age_months, os.path.join(SCRIPT_DIR, "data", "chirality_masks")
    )
    if j_args["common"]["verbose"]:
        logger.info("Closest template-age is {} months".format(tmpl_age))

    # For left/right registration, use T1 for T1-only and T2 for T2-only, but
    # for T1-and-T2 combined use T2 for <22 months otherwise T1 (img quality)
    if j_args["ID"]["has_T1w"] and j_args["ID"]["has_T2w"]:
        t1or2 = 2 if int(age_months) < 22 else 1  # NOTE 22 cutoff might change
    elif j_args["ID"]["has_T1w"]:
        t1or2 = 1
    else:  # if j_args["ID"]["has_T2w"]:
        t1or2 = 2

    # Run left/right registration script and chirality correction
    left_right_mask_nifti_fpath = run_left_right_registration(
        sub_ses, tmpl_age, t1or2, j_args, logger
    )
    logger.info("Left/right image registration completed")

    # Dilate the L/R mask and feed the dilated mask into chirality correction
    if j_args["common"]["verbose"]:
        logger.info("Now dilating left/right mask")
    dilated_LRmask_fpath = dilate_LR_mask(
        os.path.join(j_args["optional_out_dirs"]["postbibsnet"], *sub_ses),
        left_right_mask_nifti_fpath
    )
    logger.info("Finished dilating left/right segmentation mask")
    nifti_file_paths, chiral_out_dir, xfm_ref_img_dict = run_correct_chirality(dilated_LRmask_fpath,
                                                      j_args, logger)
    for t in only_Ts_needed_for_bibsnet_model(j_args["ID"]):
        nii_outfpath = reverse_regn_revert_to_native(
            nifti_file_paths, chiral_out_dir, xfm_ref_img_dict[t], t, j_args, logger
        )
        
        logger.info("The BIBSnet segmentation has had its chirality checked and "
                    "registered if needed. Now making aseg-derived mask.")

        # TODO Skip mask creation if outputs already exist and not j_args[common][overwrite]
        aseg_mask = make_asegderived_mask(j_args, chiral_out_dir, t, nii_outfpath)  # NOTE Mask must be in native T1 space too
        logger.info(f"A mask of the BIBSnet T{t} segmentation has been produced")

        # Make nibabies input dirs
        bibsnet_derivs_dir = os.path.join(j_args["optional_out_dirs"]["derivatives"], 
                                    "bibsnet")
        derivs_dir = os.path.join(bibsnet_derivs_dir, *sub_ses, "anat")
        os.makedirs(derivs_dir, exist_ok=True)
        copy_to_derivatives_dir(nii_outfpath, derivs_dir, sub_ses, t, "aseg_dseg")
        copy_to_derivatives_dir(aseg_mask, derivs_dir, sub_ses, t, "brain_mask")
        input_path = os.path.join(j_args["common"]["bids_dir"],
                                               *sub_ses, "anat",
                                               f"*T{t}w.nii.gz")
        reference_path = glob(input_path)[0]
        generate_sidecar_json(sub_ses, reference_path, derivs_dir, t, "aseg_dseg")
        generate_sidecar_json(sub_ses, reference_path, derivs_dir, t, "brain_mask")

    # Copy dataset_description.json into bibsnet_derivs_dir directory for use in nibabies
    new_data_desc_json = os.path.join(bibsnet_derivs_dir, "dataset_description.json")
    if j_args["common"]["overwrite"]:
        os.remove(new_data_desc_json)
    if not os.path.exists(new_data_desc_json):
        shutil.copy2(os.path.join(SCRIPT_DIR, "data",
                                  "dataset_description.json"), new_data_desc_json)
    if j_args["common"]["work_dir"] == os.path.join("/", "tmp", "cabinet"):
        shutil.rmtree(j_args["common"]["work_dir"])
        logger.info("Working Directory removed at {}."
                    "To keep the working directory in the future,"
                    "set a directory with the --work-dir flag.\n"
                    .format(j_args['common']['work_dir']))
    logger.info("PostBIBSnet has completed.")
    return j_args


def run_left_right_registration(sub_ses, age_months, t1or2, j_args, logger):
    """
    :param sub_ses: List with either only the subject ID str or the session too
    :param age_months: String or int, the subject's age [range] in months
    :param t1or2: Int, 1 to use T1w image for registration or 2 to use T2w
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return: String, path to newly created left/right registration output file
    """
    # Paths for left & right registration
    chiral_in_dir = os.path.join(SCRIPT_DIR, "data", "chirality_masks")
    tmpl_head = os.path.join(chiral_in_dir, "{}mo_T{}w_acpc_dc_restore.nii.gz")
    tmpl_mask = os.path.join(chiral_in_dir, "{}mo_template_LRmask.nii.gz")

    # Grab the first resized T?w from preBIBSnet to use for L/R registration
    last_digit = (t1or2 - 1 if j_args["ID"]["has_T1w"]  
                  and j_args["ID"]["has_T2w"] else 0)
    first_subject_head = glob(os.path.join(
        j_args["optional_out_dirs"]["bibsnet"], *sub_ses, "input",
        "*{}*_000{}.nii.gz".format("_".join(sub_ses), last_digit)
    ))[0]

    # Make postBIBSnet output directory for this subject/session
    outdir_LR_reg = os.path.join(j_args["optional_out_dirs"]["postbibsnet"],
                                 *sub_ses)
    os.makedirs(outdir_LR_reg, exist_ok=True)

    # Left/right registration output file path (this function's return value)
    left_right_mask_nifti_fpath = os.path.join(outdir_LR_reg, "LRmask.nii.gz")

    # Run left & right registration
    msg = "{} left/right registration on {}"
    if (j_args["common"]["overwrite"] or not
            os.path.exists(left_right_mask_nifti_fpath)):
        try:
            # In bin/LR_mask_registration.sh, the last 4 vars in cmd_LR_reg are
            # named SubjectHead, TemplateHead, TemplateMask, and OutputMaskFile
            cmd_LR_reg = (LR_REGISTR_PATH, first_subject_head,
                          tmpl_head.format(age_months, t1or2),
                          tmpl_mask.format(age_months),
                          left_right_mask_nifti_fpath)
            if j_args["common"]["verbose"]:
                logger.info(msg.format("Now running", "\n".join(
                    (first_subject_head, " ".join(cmd_LR_reg))
                )))
            subprocess.check_call(cmd_LR_reg)

        # Tell the user if ANTS crashes due to a memory error
        except subprocess.CalledProcessError as e:
            if e.returncode == 143:
                logger.error(msg.format("ANTS", first_subject_head)
                             + " failed because it ran without enough memory."
                             " Try running it again, but with more memory.\n")
            sys.exit(e)
    else:
        logger.info(msg.format("Skipping",  "{} because output already exists at {}".format(
            first_subject_head, left_right_mask_nifti_fpath
        )))
    logger.info(msg.format("Finished", first_subject_head))  # TODO Only print this message if not skipped (and do the same for all other stages)
    return left_right_mask_nifti_fpath


def run_correct_chirality(l_r_mask_nifti_fpath, j_args, logger):
    """
    :param l_r_mask_nifti_fpath: String, valid path to existing left/right
                                 registration output mask file
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return nii_fpaths: Dictionary output of correct_chirality
    :return chiral_out_dir: String file path to output directory
    :return chiral_ref_img_fpaths_dict: Dictionary containing T1w and T2w file paths
    """
    sub_ses = get_subj_ID_and_session(j_args)

    # Define paths to dirs/files used in chirality correction script
    chiral_out_dir = os.path.join(j_args["optional_out_dirs"]["postbibsnet"],
                                  *sub_ses, "chirality_correction")  # subj_ID, session, 
    os.makedirs(chiral_out_dir, exist_ok=True)
    segment_lookup_table_path = os.path.join(SCRIPT_DIR, "data", "look_up_tables",
                                             "FreeSurferColorLUT.txt")
    
    # Get BIBSnet output file, and if there are multiple, then raise an error
    out_BIBSnet_seg = os.path.join(j_args["optional_out_dirs"]["bibsnet"],
                                   *sub_ses, "output", "*.nii.gz")
    seg_BIBSnet_outfiles = glob(out_BIBSnet_seg)
    if len(seg_BIBSnet_outfiles) != 1:
        logger.error(f"There must be exactly one BIBSnet segmentation file: "
                     "{}\nResume at postBIBSnet stage once this is fixed."
                     .format(out_BIBSnet_seg))
        sys.exit()

    # Select an arbitrary T1w image path to use to get T1w space
    # (unless in T2w-only mode, in which case use an arbitrary T2w image)
    chiral_ref_img_fpaths_dict = {}
    for t in only_Ts_needed_for_bibsnet_model(j_args["ID"]):
        chiral_ref_img_fpaths = glob(os.path.join(
            j_args["common"]["bids_dir"], *sub_ses, "anat", f"*_T{t}w.nii.gz"
        ))
        chiral_ref_img_fpaths.sort()
        chiral_ref_img_fpaths_dict[t] = chiral_ref_img_fpaths[0]
    
    # Run chirality correction script and return the image to native space
    msg = "{} running chirality correction on " + seg_BIBSnet_outfiles[0]
    logger.info(msg.format("Now"))
    nii_fpaths = correct_chirality(
        seg_BIBSnet_outfiles[0], segment_lookup_table_path,
        l_r_mask_nifti_fpath, chiral_out_dir
    )
    logger.info(msg.format("Finished"))

    return nii_fpaths, chiral_out_dir, chiral_ref_img_fpaths_dict


def make_asegderived_mask(j_args, aseg_dir, t, nii_outfpath):
    """
    Create mask file(s) derived from aseg file(s) in aseg_dir
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param aseg_dir: String, valid path to existing directory with output files
                     from chirality correction
    :param t: 1 or 2, whether running on T1 or T2
    :param nii_outfpath: String, valid path to existing anat file
    :return: List of strings; each is a valid path to an aseg mask file
    """
    # binarize, fillh, and erode aseg to make mask:
    output_mask_fpath = os.path.join(
        aseg_dir, f"{nii_outfpath.split('.nii.gz')[0]}_T{t}_mask.nii.gz"
    )
    if (j_args["common"]["overwrite"] or not
            os.path.exists(output_mask_fpath)):
        maths = fsl.ImageMaths(in_file=nii_outfpath,
                               op_string=("-bin -dilM -dilM -dilM -dilM "
                                          "-fillh -ero -ero -ero -ero"),
                               out_file=output_mask_fpath)
        maths.run()

    remove_extra_clusters_from_mask(output_mask_fpath, path_to_aseg=nii_outfpath)

    return output_mask_fpath


def copy_to_derivatives_dir(file_to_copy, derivs_dir, sub_ses, space, new_fname_pt):
    """
    Copy file_to_copy into derivs_dir and rename it with the other 2 arguments
    :param file_to_copy: String, path to existing file to copy to derivs_dir
    :param derivs_dir: String, path to existing directory to copy file into
    :param sub_ses: List with either only the subject ID str or the session too
    :param space: 1 or 2, the space which the mask/aseg is in
    :param new_fname_pt: String to add to the end of the new filename
    """
    shutil.copy2(file_to_copy, os.path.join(derivs_dir, (
        "{}_space-T{}w_desc-{}.nii.gz".format("_".join(sub_ses), space, new_fname_pt)
    )))


def run_nibabies(j_args, logger):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return: j_args, unchanged
    """
    # Get all XCP-D parameters, excluding any whose value is null/None
    nibabies_args = get_optional_args_in(j_args["nibabies"])  # TODO

    # Get nibabies options from parameter file and turn them into flags
    nibabies_args.append("--age-months")
    nibabies_args.append(j_args["common"]["age_months"])
    for nibabies_arg in ["cifti_output", "work_dir"]:
        nibabies_args.append(as_cli_arg(nibabies_arg))
        nibabies_args.append(j_args["nibabies"][nibabies_arg])
        # TODO Ensure that all common args required by nibabies are added
        # TODO If an optional_real_dirpath is null/None, don't even include the flag

    # Check whether aseg and mask files were produced by BIBSnet
    glob_path = os.path.join(j_args["optional_out_dirs"]["bibsnet"],
                             "*{}*.nii.gz")
    aseg_glob = glob(glob_path.format("aseg"))
    mask_glob = glob(glob_path.format("mask"))
    if aseg_glob and mask_glob:
        derivs = ["--derivatives", j_args["optional_out_dirs"]["bibsnet"]]
    else:
        derivs = list()
        # TODO If j_args[nibabies][derivatives] has a path, use that instead
    
    # Run nibabies
    print(" ".join(str(x) for x in [j_args["nibabies"]["singularity_image_path"],  # subprocess.check_call  # TODO
                           j_args["common"]["bids_dir"],
                           j_args["optional_out_dirs"]["nibabies"],
                           "participant", *derivs, *nibabies_args]))
    logger.info("Nibabies has completed")
    
    return j_args


def run_XCPD(j_args, logger):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return: j_args, unchanged
    """
    # Get all XCP-D parameters, excluding any whose value is null/None
    xcpd_args = get_optional_args_in(j_args["XCPD"])
    subprocess.check_call([ #    # TODO Ensure that all "common" args required by XCPD are added
        "singularity", "run", "--cleanenv",
        "-B", j_args["optional_out_dirs"]["nibabies"] + ":/data:ro",
        "-B", j_args["optional_out_dirs"]["XCPD"] + ":/out",
        "-B", j_args["XCPD"]["work_dir"] + ":/work",
        "/home/faird/shared/code/external/pipelines/ABCD-XCP/xcp-d_unstable03112022a.sif",  # TODO Make this an import and/or a parameter
        "/data", "/out", "--participant-label",  # "-w", "/work", 
        j_args["ID"]["subject"], *xcpd_args
    ])
    logger.info("XCP-D has completed")
    return j_args


if __name__ == "__main__":
    main()
