#!/usr/bin/env python3
# coding: utf-8

"""
Connectome ABCD-XCP niBabies Imaging nnu-NET (CABINET)
Greg Conan: gconan@umn.edu
Created: 2021-11-12
Updated: 2022-08-02
"""
# Import standard libraries
import argparse
from datetime import datetime
from glob import glob
import logging
from nipype.interfaces import fsl
import os
import pandas as pd
import pdb
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
LR_REGISTR_PATH = os.path.join(SCRIPT_DIR, "bin", "LR_mask_registration.sh")
TYPES_JSON = os.path.join(SCRIPT_DIR, "src", "param-types.json")

# Custom local imports
from src.utilities import (
    as_cli_attr, as_cli_arg, correct_chirality, create_anatomical_average,
    crop_image, dict_has, dilate_LR_mask, exit_with_time_info,
    extract_from_json, get_and_make_preBIBSnet_work_dirs, get_optional_args_in,
    get_stage_name, get_subj_ID_and_session, get_template_age_closest_to,
    make_given_or_default_dir, resize_images, run_all_stages,
    valid_readable_json, validate_parameter_types, valid_readable_dir,
    valid_subj_ses_ID, valid_whole_number
)


def main():
    start_time = datetime.now()  # Time how long the script takes
    logger = make_logger()  # Make object to log error/warning/status messages

    # Get and validate command-line arguments and parameters from .JSON file
    STAGES = [run_preBIBSnet, run_BIBSnet, run_postBIBSnet, run_nibabies,
              run_XCPD]
    json_args = get_params_from_JSON([get_stage_name(stg) for stg in STAGES],
                                     logger)  # TODO Un-capitalize "BIBS" everywhere (except import BIBSnet.run?)
    if json_args["common"]["verbose"]:
        logger.info("Parameters from input .JSON file:\n{}"
                    .format(json_args))

    # Run every stage that the parameter file says to run
    run_all_stages(STAGES, json_args["stage_names"]["start"],
                   json_args["stage_names"]["end"], json_args, logger)
    # TODO default to running all stages if not specified by the user
    # TODO add error if end is given as a stage that happens before start

    # Show user how long the pipeline took and end the pipeline here
    exit_with_time_info(start_time)


def make_logger():
    """
    Make logger to log status updates, warnings, and other important info
    :return: logging.Logger
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
    parser = argparse.ArgumentParser()
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
              "/path/to/output/derivatives/")  # TODO Remove optional_out_dirs from parameter file and just keep output_dir as j_args[common][output_dir]
    )
    parser.add_argument(
        "analysis_level", choices=["participant"],  # TODO Will we ever need to add group-level analysis functionality? Currently this argument does absolutely nothing
        help=("Processing level. Currently the only choice is 'participant'."
              "See BIDS-Apps specification.")
    )

    # Required flag arguments
    parser.add_argument(
        "-jargs", "-params", "--parameter-json", dest="parameter_json",
        type=valid_readable_json, required=True,
        help=("Valid path to existing readable parameter .JSON file. See "
              "README.md and example parameter .JSON files for more "
              "information on parameters.")
        # TODO: Add description of all nibabies and XCP-D parameters to the README?
        # TODO: In the README.md file, mention which arguments are required and which are optional (with defaults)
    )
    parser.add_argument(
        "-participant", "--subject", "-sub", "--participant-label",
        required=True, dest="participant_label", type=valid_subj_ses_ID,
        help=("The participant's unique subject identifier, without 'sub-'"
              "prefix. Example: 'ABC12345'")  # TODO Make CABINET able to accept with OR without 'sub-' prefix
    )

    # Optional flag arguments
    parser.add_argument(
        "-ses", "--session", "--session-id", type=valid_subj_ses_ID,
        help=("The name of the session to processes participant data for. "
              "Example: baseline_year1")
    )
    parser.add_argument(
        "-age", "-months", "--age-months", type=valid_whole_number,
        help=("Positive integer, the participant's age in months. For "
              "example, -age 5 would mean the participant is 5 months old."
              "Include this argument unless the age in months is specified in"
              "the participants.tsv file inside the BIDS input directory.")
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help=("Include this flag to print detailed information and every "
              "command being run by CABINET to stdout. Otherwise CABINET "
              "will only print warnings, errors, and minimal output.")
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
        "-start", "--starting-stage", dest="start",
        choices=stage_names[:3], default=stage_names[0],   # TODO Change default to start where we left off by checking which stages' prerequisites and outputs already exist
        help=msg_stage.format("first", stage_names[0], ", ".join(stage_names[:3]))  # TODO Change to include all stage names; right now it just includes the segmentation pipeline
    )
    parser.add_argument(
        "-end", "--ending-stage", dest="end",
        choices=stage_names[:3], default=default_end_stage,
        help=msg_stage.format("last", default_end_stage, ", ".join(stage_names[:3]))
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
    :return: Dictionary of validated parameters from parameter .JSON file
    """
    # Get command-line input arguments and use them to get .JSON parameters
    j_args = extract_from_json(cli_args["parameter_json"])
    script_dir_attr = as_cli_attr(SCRIPT_DIR_ARG)
    j_args["meta"] = {script_dir_attr: SCRIPT_DIR,
                      "slurm": bool(cli_args[script_dir_attr])}

    # Add command-line arguments to j_args
    j_args["stage_names"] = {"start": cli_args["start"],
                             "end": cli_args["end"]}  # TODO Maybe save the stage_names list in here too to replace optional_out_dirs use cases?
    for arg_to_add in ("age_months", "bids_dir", "participant_label",
                       "overwrite", "session", "verbose"):
        j_args["common"][arg_to_add] = cli_args[arg_to_add]

    # TODO Remove all references to the optional_out_dirs arguments, and change
    #      j_args[optional_out_dirs][derivatives] to instead be j_args[common][output_dir]
    j_args["optional_out_dirs"] = {stagename: None for stagename in stage_names} 
    j_args["optional_out_dirs"]["derivatives"] = cli_args["output_dir"]

    # TODO Automatically assign j_args[common][fsl_bin_path] and 
    #      j_args[bibsnet][nnUNet_predict_path] if in the Singularity container

    sub_ses = get_subj_ID_and_session(j_args)

    # TODO Check whether the script is being run from a container...
    # - Docker: stackoverflow.com/a/25518538
    # - Singularity: stackoverflow.com/a/71554776
    # ...and if it is, then assign default values to these j_args, overwriting user's values:
    # j_args[common][fsl_bin_path]
    # j_args[BIBSnet][nnUNet_predict_path]
    # j_args[BIBSnet][code_dir]
    # Also, check whether the Docker/Singularity environment variables show up in os.environ for us to use here
    # print(vars(os.environ))

    # Define (and create) default paths in derivatives directory structure for each stage
    default_derivs_dir = os.path.join(j_args["common"]["bids_dir"], "derivatives")
    j_args = ensure_j_args_has_bids_subdirs(j_args, stage_names, sub_ses,
                                            default_derivs_dir)

    # Verify that every parameter in the parameter .JSON file is a valid input
    j_args = validate_parameter_types(j_args, extract_from_json(TYPES_JSON),
                                      cli_args["parameter_json"], parser,
                                      stage_names)
    
    # Verify that subject-session directory path exists
    sub_ses_dir = os.path.join(j_args["common"]["bids_dir"], *sub_ses)
    if not os.path.exists(sub_ses_dir):
        parser.error("No subject/session directory found at {}\nCheck that "
                     "your participant_label and session are correct."
                     .format(sub_ses_dir))

    # Using dict_has instead of easier ensure_dict_has so that the user only
    # needs a participants.tsv file if they didn't specify age_months
    if not dict_has(j_args["common"], "age_months"):
        j_args["common"]["age_months"] = read_age_from_participants_tsv(j_args,
                                                                        logger)
    # TODO Figure out which column in the participants.tsv file has age_months

    # Create BIBSnet in/out directories
    dir_BIBSnet = dict()
    for io in ("in", "out"):
        dir_BIBSnet[io] = os.path.join(j_args["optional_out_dirs"]["bibsnet"],
                                       *sub_ses, "{}put".format(io))
        os.makedirs(dir_BIBSnet[io], exist_ok=True)

    # Save paths to files used by multiple stages:

    # 1. Symlinks to resized images chosen by the preBIBSnet cost function
    j_args["optimal_resized"] = {"T{}w".format(t): os.path.join(
                                     dir_BIBSnet["in"], "{}_optimal_resized_000{}.nii.gz"
                                                        .format("_".join(sub_ses), t-1)
                                 ) for t in (1, 2)}

    # Check that all required input files exist for the stages to run
    verify_CABINET_inputs_exist(sub_ses, j_args, parser)
    logger.info("All required input files exist.")
    if j_args["common"]["verbose"]:
        logger.info(" ".join(sys.argv[:]))  # Print all

    # 2. roi2full for preBIBSnet and postBIBSnet transformation
    # j_args["xfm"]["roi2full"] =   # TODO

    return j_args


def ensure_j_args_has_bids_subdirs(j_args, derivs, sub_ses, default_parent):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param subdirnames: Unpacked list of strings. Each names 1 part of a path
                        under j_args[common][bids_dir]. The last string is
                        mapped by j_args[optional_out_dirs] to the subdir path.
    :return: j_args, but with the (now-existing) subdirectory path
    """
    j_args["optional_out_dirs"] = make_given_or_default_dir(
        j_args["optional_out_dirs"], "derivatives", default_parent
    )
    for deriv in derivs:
        subdir_path = os.path.join(j_args["optional_out_dirs"]["derivatives"],
                                   deriv)
        j_args["optional_out_dirs"] = make_given_or_default_dir(
            j_args["optional_out_dirs"], deriv, subdir_path
        )
        os.makedirs(os.path.join(j_args["optional_out_dirs"][deriv], *sub_ses),
                    exist_ok=True)  # Make all subject-session output dirs
    return j_args


def verify_CABINET_inputs_exist(sub_ses, j_args, parser):
    """
    Given a stage, verify that all of the necessary inputs for that stage exist 
    :param a_stage: String naming a stage
    :param sub_ses: List with either only the subject ID str or the session too
    :param j_args: Dictionary containing all args from parameter .JSON file
    """
    # Define globbable paths to prereq files for the script to check
    out_BIBSnet_seg = os.path.join(j_args["optional_out_dirs"]["bibsnet"],
                                   *sub_ses, "output", "*.nii.gz")
    subject_heads = [os.path.join(
            j_args["optional_out_dirs"]["bibsnet"], *sub_ses, "input",
            "*{}*_000{}.nii.gz".format("_".join(sub_ses), t1or2 - 1)
        ) for t1or2 in (1, 2)]
    out_paths_BIBSnet = [os.path.join(j_args["optional_out_dirs"]["bibsnet"],
                                      "*{}*.nii.gz".format(x))
                         for x in ("aseg", "mask")]

    # Map each stage's name to its required input files
    stage_prerequisites = {"prebibsnet": list(),
                           "bibsnet": [fpath for fpath in
                                       j_args["optimal_resized"].values()],
                           "postbibsnet": [out_BIBSnet_seg, *subject_heads],
                           "nibabies": out_paths_BIBSnet,
                           "xcpd": []}

    # For each stage that will be run, verify that its prereq input files exist
    all_stages = [s for s in stage_prerequisites.keys()]

    # required_files = stage_prerequisites[j_args["stage_names"]["start"]]
    start_ix = all_stages.index(j_args["stage_names"]["start"]) 
    for stage in all_stages[:start_ix+1]:

        # if stage == j_args["stage_names"]["start"]:
        # if will_run_stage(stage, j_args["stage_names"]["start"], j_args["stage_names"]["end"], all_stages):

        missing_files = list()
        for globbable in stage_prerequisites[stage]:
            if not glob(globbable):
                missing_files.append(globbable)
        if missing_files:
            parser.error("The file(s) below are needed to run the {} stage, "
                        "but they do not exist.\n{}\n"
                        .format(stage, "\n".join(missing_files)))


def read_age_from_participants_tsv(j_args, logger):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :return: Int, the subject's age (in months) listed in participants.tsv
    """
    columns = {x: "str" for x in ("age", "session", "participant_id")}

    # Read in participants.tsv
    part_tsv_df = pd.read_csv(
        os.path.join(j_args["common"]["bids_dir"],
                     "participants.tsv"), sep="\t", dtype=columns
    )

    # Column names of participants.tsv                         
    age_months_col = "age" # TODO is there a way to ensure the age column is given in months using the participants.json (the participants.tsv's sidecar)
    sub_ID_col = "participant_id"
    ses_ID_col = "session"

    # Get and return the age_months value from participants.tsv
    subj_row = part_tsv_df[  # TODO Run ensure_prefixed on the sub_ID_col?
        part_tsv_df[sub_ID_col] == j_args["common"]["participant_label"]
    ]  # select where "participant_id" matches
    subj_row = subj_row[  # TODO Run ensure_prefixed on the ses_ID_col?
        subj_row[ses_ID_col] == j_args["common"]["session"]
    ]  # select where "session" matches
    if j_args["common"]["verbose"]:
        logger.info("Subject details from participants.tsv row:\n{}"
                    .format(subj_row))
    return int(subj_row[age_months_col])


def run_preBIBSnet(j_args, logger):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return: j_args, but with preBIBSnet working directory names added
    """
    # sub_ses = get_subj_ID_and_session(j_args)
    completion_msg = "The anatomical images have been {} for use in BIBSnet"
    preBIBSnet_paths = get_and_make_preBIBSnet_work_dirs(j_args)

    # If there are multiple T1ws/T2ws, then average them
    create_anatomical_average(preBIBSnet_paths["avg"])  # TODO make averaging optional with later BIBSnet model?

    # Crop T1w and T2w images
    cropped = dict()
    crop2full = dict()
    for t in (1, 2):
        cropped[t] = preBIBSnet_paths["crop_T{}w".format(t)]
        crop2full[t] = crop_image(preBIBSnet_paths["avg"]["T{}w_avg".format(t)],
                                  cropped[t], j_args, logger)
    logger.info(completion_msg.format("cropped"))

    # Resize T1w and T2w images 
    # TODO Make ref_img an input parameter if someone wants a different reference image?
    reference_img = os.path.join(SCRIPT_DIR, "data", "MNI_templates",
                                 "INFANT_MNI_T{}_1mm.nii.gz") # TODO Pipeline should verify that these exist before running
    id_mx = os.path.join(SCRIPT_DIR, "data", "identity_matrix.mat")

    transformed_images = resize_images(
        cropped, preBIBSnet_paths["resized"], reference_img, 
        id_mx, crop2full, preBIBSnet_paths["avg"], j_args, logger
    )
    logger.info(completion_msg.format("resized"))
    
    # TODO Move this whole block to postBIBSnet, so it copies everything it needs first
    # Make a symlink in BIBSnet input dir to transformed_images[T1w]
    for t in (1, 2):
        tw = "T{}w".format(t)
        out_nii_fpath = j_args["optimal_resized"][tw]
        os.makedirs(os.path.dirname(out_nii_fpath), exist_ok=True)
        if not os.path.exists(out_nii_fpath):  # j_args["common"]["overwrite"] or 
            shutil.copy2(transformed_images[tw], out_nii_fpath)
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
    if j_args["common"]["overwrite"] or not glob(os.path.join(
        dir_BIBS.format("out"), "*"
    )):
        # Import BIBSnet functionality from BIBSnet/run.py
        parent_BIBSnet = os.path.dirname(j_args["bibsnet"]["code_dir"])
        logger.info("Importing BIBSnet from {}".format(parent_BIBSnet))
        sys.path.append(parent_BIBSnet)
        from BIBSnet.run import run_nnUNet_predict
               
        # TODO test functionality of importing BIBSNet function via params json (j_args)
        #parent_BIBSnet = os.path.dirname(j_args["bibsnet"]["code_dir"])
        #logger.info("Importing BIBSnet from {}".format(parent_BIBSnet))
        #sys.path.append("/home/cabinet/SW/BIBSnet")
        #from BIBSnet.run import run_nnUNet_predict

        try:  # Run BIBSnet
            inputs_BIBSnet = {"model": j_args["bibsnet"]["model"],
                              "nnUNet": j_args["bibsnet"]["nnUNet_predict_path"],
                              "input": dir_BIBS.format("in"),
                              "output": dir_BIBS.format("out"),
                              "task": str(j_args["bibsnet"]["task"])}
            os.makedirs(inputs_BIBSnet["output"], exist_ok=True)
            run_nnUNet_predict(inputs_BIBSnet)

        except subprocess.CalledProcessError as e:
            # BIBSnet will crash even after correctly creating a segmentation,
            # so only crash CABINET if that segmentation is not made.
            outfpath = os.path.join(dir_BIBS.format("out"),
                                    "{}_{}_optimal_resized.nii.gz".format(*sub_ses))
            if not os.path.exists(outfpath):
                logger.error("BIBSnet failed to create this segmentation file...\n{}\n...from these inputs:\n{}".format(outfpath, inputs_BIBSnet))
                sys.exit(e)

        # Remove unneeded empty directories
        for unneeded_dir_name in ("nnUNet_cropped_image", "nnUNet_raw_data"):
            unneeded_dir_path = os.path.join(j_args["optional_out_dirs"]["derivatives"], unneeded_dir_name)
            logger.info("Deleting unnecessary empty directory at {}"
                        .format(unneeded_dir_path))
            if os.path.isdir(unneeded_dir_path):
                os.removedirs(unneeded_dir_path)
    
        # TODO hardcoded below call to run_nnUNet_predict. Will likely want to change and integrate into j_args
        #run_nnUNet_predict({"model": "3d_fullres",
        #                    "nnUNet": "/opt/conda/bin/nnUNet_predict",
        #                    "input": dir_BIBS.format("in"),
        #                    "output": dir_BIBS.format("out"),
        #                    "task": str(512)})
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
    age_months = j_args["common"]["age_months"]
    logger.info("Age of participant: {} months".format(age_months))

    # Get template closest to age
    tmpl_age = get_template_age_closest_to(
        age_months, os.path.join(SCRIPT_DIR, "data", "chirality_masks")
    )
    if j_args["common"]["verbose"]:
        logger.info("Closest template-age is {} months".format(tmpl_age))
    # if age_months > 33: age_months = "34-38"

    # Run left/right registration script and chirality correction
    left_right_mask_nifti_fpath = run_left_right_registration(
        j_args, sub_ses, tmpl_age, 2 if int(age_months) < 22 else 1, logger  # NOTE 22 cutoff might change
    )
    logger.info("Left/right image registration completed")

    # Dilate the L/R mask and feed the dilated mask into chirality correction
    dilated_LRmask_fpath = dilate_LR_mask(
        os.path.join(j_args["optional_out_dirs"]["postbibsnet"], *sub_ses),
        left_right_mask_nifti_fpath
    )
    logger.info("Finished dilating left/right segmentation mask")
    nii_outfpath = run_chirality_correction(dilated_LRmask_fpath, # left_right_mask_nifti_fpath,  # TODO Rename to mention that this also does registration?
                                            j_args, logger)
    chiral_out_dir = os.path.dirname(nii_outfpath)
    logger.info("The BIBSnet segmentation has had its chirality checked and "
                "registered if needed. Now making aseg-derived mask.")

    # TODO Skip mask creation if outputs already exist and not j_args[common][overwrite]
    aseg_mask = make_asegderived_mask(j_args, chiral_out_dir, nii_outfpath)  # NOTE Mask must be in native T1 space too
    logger.info("A mask of the BIBSnet segmentation has been produced")

    # Make nibabies input dirs
    precomputed_dir = os.path.join(j_args["optional_out_dirs"]["derivatives"], 
                                   "precomputed")
    derivs_dir = os.path.join(precomputed_dir, *sub_ses, "anat")
    os.makedirs(derivs_dir, exist_ok=True)
    copy_to_derivatives_dir(nii_outfpath, derivs_dir, sub_ses, "aseg_dseg")
    """
    for eachfile in os.scandir(chiral_out_dir):
        if "native" in os.path.basename(eachfile):
            copy_to_derivatives_dir(eachfile, derivs_dir, sub_ses, "aseg_dseg")  # TODO Can these be symlinks?
    """
    copy_to_derivatives_dir(aseg_mask, derivs_dir, sub_ses, "brain_mask")

    # Copy dataset_description.json into precomputed directory for nibabies
    new_data_desc_json = os.path.join(precomputed_dir, "dataset_description.json")
    if j_args["common"]["overwrite"] or not os.path.exists(new_data_desc_json):
        shutil.copy2(os.path.join(SCRIPT_DIR, "data",
                                  "dataset_description.json"), new_data_desc_json)
        
    logger.info("PostBIBSnet has completed.")
    return j_args


def run_left_right_registration(j_args, sub_ses, age_months, t1or2, logger):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param sub_ses: List with either only the subject ID str or the session too
    :param age_months: String or int, the subject's age [range] in months
    :param t1or2: Int, 1 to use T1w image for registration or 2 to use T2w
    :return: String, path to newly created left/right registration output file
    """
    # Paths for left & right registration
    chiral_in_dir = os.path.join(SCRIPT_DIR, "data", "chirality_masks")
    tmpl_head = os.path.join(chiral_in_dir, "{}mo_T{}w_acpc_dc_restore.nii.gz")
    tmpl_mask = os.path.join(chiral_in_dir, "{}mo_template_LRmask.nii.gz") # "brainmasks", {}mo_template_brainmask.nii.gz")

    # Grab the first resized T?w from preBIBSnet to use for L/R registration
    first_subject_head = glob(os.path.join(
        j_args["optional_out_dirs"]["bibsnet"], *sub_ses, "input",
        "*{}*_000{}.nii.gz".format("_".join(sub_ses), t1or2 - 1)
    ))[0]

    # Make postBIBSnet output directory for this subject/session
    outdir_LR_reg = os.path.join(j_args["optional_out_dirs"]["postbibsnet"],
                                 *sub_ses)
    os.makedirs(outdir_LR_reg, exist_ok=True)

    # Left/right registration output file path (this function's return value)
    left_right_mask_nifti_fpath = os.path.join(outdir_LR_reg, "LRmask.nii.gz")

    # Run left & right registration  # NOTE: Script ran successfully until here 2022-03-08
    msg = "{} left/right registration on {}"
    if (j_args["common"]["overwrite"] or not
            os.path.exists(left_right_mask_nifti_fpath)):
        # logger.info(msg.format("Running", first_subject_head))
        try:
            # SubjectHead TemplateHead TemplateMask OutputMaskFile
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
    


def run_chirality_correction(l_r_mask_nifti_fpath, j_args, logger):
    """
    :param l_r_mask_nifti_fpath: String, valid path to existing left/right
                                 registration output mask file
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return: String, valid path to existing directory containing newly created
             chirality correction outputs
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
        logger.error("There must be exactly one BIBSnet segmentation file: "
                     "{}\nResume at postBIBSnet stage once this is fixed."
                     .format(out_BIBSnet_seg))
        sys.exit()

    # Select an arbitrary T1w image path to use to get T1w space
    path_T1w = glob(os.path.join(j_args["common"]["bids_dir"],
                                 *sub_ses, "anat", "*_T1w.nii.gz"))[0]

    # Run chirality correction script
    nii_outfpath = correct_chirality(seg_BIBSnet_outfiles[0], segment_lookup_table_path,
                                     l_r_mask_nifti_fpath, chiral_out_dir, 
                                     path_T1w, j_args, logger)
    return nii_outfpath  # chiral_out_dir


def make_asegderived_mask(j_args, aseg_dir, nii_outfpath):
    """
    Create mask file(s) derived from aseg file(s) in aseg_dir
    :param aseg_dir: String, valid path to existing directory with output files
                     from chirality correction
    :return: List of strings; each is a valid path to an aseg mask file
    """
    # binarize, fillh, and erode aseg to make mask:
    output_mask_fpath = os.path.join(
        aseg_dir, "{}_mask.nii.gz".format(nii_outfpath.split(".nii.gz")[0])
    )
    if (j_args["common"]["overwrite"] or not
            os.path.exists(output_mask_fpath)):
        maths = fsl.ImageMaths(in_file=nii_outfpath,  # (anat file)
                                op_string=("-bin -dilM -dilM -dilM -dilM "
                                            "-fillh -ero -ero -ero -ero"),
                                out_file=output_mask_fpath)
        maths.run()
    return output_mask_fpath


def copy_to_derivatives_dir(file_to_copy, derivs_dir, sub_ses, new_fname_pt):
    """
    Copy file_to_copy into derivs_dir and rename it with the other 2 arguments
    :param file_to_copy: String, path to existing file to copy to derivs_dir
    :param derivs_dir: String, path to existing directory to copy file into
    :param sub_ses: List with either only the subject ID str or the session too
    :param new_fname_pt: String to add to the end of the new filename
    """
    shutil.copy2(file_to_copy, os.path.join(derivs_dir, (
        "{}_space-orig_desc-{}.nii.gz".format("_".join(sub_ses), new_fname_pt)
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
        """
        warn_user_of_conditions(
            ("Missing {{}} files in {}\nNow running nibabies with JLF but not "
             "BIBSnet.".format(j_args["optional_out_dirs"]["bibsnet"])),
            logger, mask=mask_glob, aseg=aseg_glob
        ) """
    
    # Run nibabies
    print()
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
        j_args["common"]["participant_label"], *xcpd_args
    ])
    logger.info("XCP-D has completed")
    return j_args


if __name__ == "__main__":
    main()