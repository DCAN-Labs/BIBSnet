#!/usr/bin/env python3
# coding: utf-8

"""
Connectome ABCD-XCP niBabies Imaging nnu-NET (CABINET)
Created: 2021-11-12
Updated: 2022-01-24
"""

# Import standard libraries
import argparse
from datetime import datetime
from glob import glob
from nipype.interfaces import fsl
import os
import pandas as pd
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
LR_REGISTR_PATH = os.path.join(SCRIPT_DIR, "bin", "LR_mask_registration.sh")

# Custom local imports
from src.utilities import (
    as_cli_arg, copy_and_rename_file, crop_images, ensure_dict_has,
    ensure_prefixed, exit_with_time_info, extract_from_json, get_subj_ses,
    resize_images, run_all_stages, valid_readable_json, warn_user_biconditional
)
from src.img_processing.correct_chirality import correct_chirality


def main():
    # Time how long the script takes and get command-line arguments from user 
    start_time = datetime.now()

    STAGES = {"preBIBSnet": run_preBIBSnet, "BIBSnet": run_BIBSnet,
              "postBIBSnet": run_postBIBSnet, "nibabies": run_nibabies,
              "XCPD": run_XCPD}
    json_args = get_params_from_JSON(list(STAGES.keys()))

    print(json_args)  # TODO REMOVE LINE

    run_all_stages(STAGES, json_args["stages"]["start"],
                   json_args["stages"]["end"], json_args) # TODO Ensure that each stage only needs j_args?

    # Show user how long the pipeline took and end the pipeline here
    exit_with_time_info(start_time)


def get_params_from_JSON(stages):
    """
    :param stages: List of strings; each names a stage to run
    :return: Dictionary containing all parameters from parameter .JSON file
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
    parser.add_argument(
        "-start", "--starting-stage", dest="start",
        choices=stages, default=stages[0]
    )
    parser.add_argument(
        "-end", "--ending-stage", dest="end",
        choices=stages, default=stages[-1]
    )
    cli_args = parser.parse_args()
    return validate_json_args(extract_from_json(cli_args.parameter_json),
                              cli_args.start, cli_args.end, stages, parser)


def validate_json_args(j_args, start, end, stages, parser):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param stages: List of strings; each names a stage to run
    """
    # TODO Validate types in j_args; e.g. validate that nibabies[age_months] is an int

    j_args["common"] = ensure_dict_has(
        j_args["common"], "age_months",
        read_age_from_participants_tsv(j_args)
        # TODO Figure out which column in the participants.tsv file has the age_months value
    )

    # Define (and create) default paths in derivatives directory structure
    j_args = ensure_j_args_has_bids_subdir(j_args, "derivatives")
    for deriv in stages:
        j_args = ensure_j_args_has_bids_subdir(j_args, "derivatives", deriv)  #  + "_output_dir")

    j_args["stages"] = {"start": start, "end": end} 
    return j_args


def ensure_j_args_has_bids_subdir(j_args, *subdirnames):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param subdirnames: Unpacked list of strings. Each names 1 part of a path
                        under j_args[common][bids_dir]. The last string is
                        mapped by j_args[optional_out_dirs] to the subdir path.
    :return: j_args, but with the (now-existing) subdirectory path
    """
    subdir_path = os.path.join(j_args["common"]["bids_dir"], *subdirnames)
    j_args["optional_out_dirs"] = ensure_dict_has(j_args["optional_out_dirs"],
                                                  subdirnames[-1], subdir_path)
    os.makedirs(j_args["optional_out_dirs"][subdirnames[-1]], exist_ok=True)
    return j_args


def read_age_from_participants_tsv(j_args):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :return: Int, the subject's age (in months) listed in participants.tsv
    """
    columns = {"age": "str", "participant_id": "str", "session": "str"}

    # Read in participants.tsv
    part_tsv_df = pd.read_csv(
        os.path.join(j_args["common"]["bids_dir"],
                     "participants.tsv"), sep="\t", dtype=columns
    )

    # Column names of participants.tsv                         
    age_months_col = "age" # TODO Get name of column with age_months value
    sub_ID_col = "participant_id" # TODO Figure out the subject ID column name (participant ID or subject ID)
    ses_ID_col = "session"

    """
    # Ensure that the subject and session IDs start with the right prefixes
    sub_ses = {"participant_label": "sub-", "session": "ses-"}
    sub_ses_labels = sub_ses.copy()
    for param_name, prefix in sub_ses.items():
        param = j_args["common"][param_name]
        sub_ses_labels[prefix] = (param[-len(prefix)] if param[:len(prefix)]
                                  == prefix else param)
    print(sub_ses_labels)
    """
    # Get and return the age_months value from participants.tsv
    subj_row = part_tsv_df[
        part_tsv_df[sub_ID_col] == j_args["common"]["participant_label"]
    ]
    subj_row = subj_row[
        subj_row[ses_ID_col] == j_args["common"]["session"]
    ] # select where "participant_id" and "session" match

    print(subj_row)
    return int(subj_row[age_months_col])


def run_preBIBSnet(j_args, logger):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    """
    subj_ID = ensure_prefixed(j_args["common"]["participant_label"], "sub-")
    session = ensure_prefixed(j_args["common"]["session"], "ses-")
    subject_dir = os.path.join(j_args["common"]["bids_dir"], subj_ID)

    # Make working directories to run pre-BIBSnet processing in
    subj_work_dir = os.path.join(
        j_args["optional_out_dirs"]["preBIBSnet"], subj_ID
    )
    work_dirs = {"parent": os.path.join(subj_work_dir, session)}
    for dirname in ("BIDS_data", "cropped", "resized"):
        work_dirs[dirname] = os.path.join(work_dirs["parent"], dirname)
    for eachdir in work_dirs.values():
        os.makedirs(eachdir, exist_ok=True)

    # Copy any file with T1 or T2 in its name from BIDS/anat dir to BIBSnet work dir
    for each_anat in (1, 2):
        for eachfile in glob(os.path.join(
            subject_dir, session, "anat", "*T{}w*.nii.gz".format(each_anat)
        )):
            new_fpath = os.path.join(work_dirs["BIDS_data"],
                                     os.path.basename(eachfile))
            copy_and_rename_file(eachfile, new_fpath)
            os.chmod(new_fpath, 0o775)
            # TODO What do we do if there's >1 T1w or >1 T2w? nnU-Net can't handle >1 rn.
            #      Average them (after crop/resize) like the pre-FreeSurfer infant pipeline does,
            #      per Fez and Luci, and do a rigid body registration while averaging
    
    # Crop and resize images
    crop_images(work_dirs["BIDS_data"], work_dirs["cropped"])
    os.chmod(work_dirs["cropped"], 0o775)
    ref_img = os.path.join(SCRIPT_DIR, 'data', 'test_subject_data', '1mo',
                           'sub-00006_T1w_acpc_dc_restore.nii.gz')
    id_mx = os.path.join(SCRIPT_DIR, 'data', 'identity_matrix.mat')
    resize_images(work_dirs["cropped"], work_dirs["resized"], ref_img, id_mx)

    """
    # Copy resized images into bids_dir
    # (replacing originals, which were copied into the preBIBSnet work dir)
    for image in os.scandir(work_dirs["resized"]):
        copy_and_rename_file(image.path, os.path.join(
            j_args["common"]["bids_dir"], subj_ID, session, "anat", image.name
        ))
    """

    # TODO rename files to nnU-Net conventions (0000, 0001)
    return j_args


def run_BIBSnet(j_args, logger):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    """
    # TODO Test BIBSnet functionality once it's containerized
    """
    verify_image_file_names(j_args)  # TODO Ensure that the T1s have _0000 at the end of their filenames and T2s have _0001

    if j_args["common"]["age_months"] <= 8:
        j_args = copy_images_to_BIBSnet_dir(j_args)           # TODO
        j_args["segmentation"] = run_BIBSnet_predict(j_args)  # TODO
    """
    return j_args


def run_postBIBSnet(j_args, logger):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    """
    subj_ID = ensure_prefixed(j_args["common"]["participant_label"], "sub-")
    session = ensure_prefixed(j_args["common"]["session"], "ses-")

    # Template selection values
    age_months = j_args["common"]["age_months"]
    print(age_months)
    t1or2 = 2 if int(age_months) < 22 else 1
    if age_months > 33:
        age_months = "34-38"

    # Paths for left & right registration
    chiral_in_dir = os.path.join(SCRIPT_DIR, "data", "chirality_masks")
    tmpl_head = os.path.join(chiral_in_dir, "{}mo_T{}w_acpc_dc_restore.nii.gz")
    tmpl_mask = os.path.join(chiral_in_dir, "{}mo_template_LRmask.nii.gz")
    subject_head_path = os.path.join(
        j_args["optional_out_dirs"]["BIBSnet"], subj_ID, session,
        "{}_acq-T1inT2".format(j_args["common"]["participant_label"]) # TODO Figure out / double-check the BIBSnet output file name for this participant
    )

    # Run left & right registration  # NOTE: Script run successfully to here 2022-01-20_13:47
    subprocess.check_call((LR_REGISTR_PATH, subject_head_path,
                           tmpl_head.format(age_months, t1or2),
                           tmpl_mask.format(age_months)))

    # Chirality correction
    chiral_out_dir = os.path.join(j_args["BIBSnet"]["output_dir"], "chirality_correction")
    os.makedirs(chiral_out_dir)
    segment_lookup_table = os.path.join(SCRIPT_DIR, "data", "look_up_tables",
                                        "FreeSurferColorLUT.txt")
    left_right_mask_nifti_file = "LRmask.nii.gz"  # TODO Make the --output path in LR_mask_registration.sh an absolute path and copy it here
    nifti_output_file_path = os.path.join(chiral_out_dir,
                                          os.path.basename(subject_head_path))
    correct_chirality(subject_head_path, segment_lookup_table,
                      left_right_mask_nifti_file, nifti_output_file_path)

    # TODO Bring aseg back into native T1 space

    masks = make_asegderived_mask(chiral_out_dir)  # NOTE Mask must be in native T1 space too
    
    # Make nibabies input dirs
    derivs_dir = os.path.join(j_args["optional_out_dirs"]["BIBSnet"],
                              j_args["common"]["participant_label"],
                              j_args["common"]["session"])
    os.makedirs(derivs_dir, exist_ok=True)
    sub_ses = get_subj_ses(j_args)
    for eachfile in os.scandir(chiral_out_dir):
        if "aseg" in chiral_out_dir.name:
            copy_to_derivatives_dir(eachfile, derivs_dir, sub_ses, "aseg_dseg")
    for each_mask in masks:  # There should only be 1, for the record
        copy_to_derivatives_dir(each_mask, derivs_dir, sub_ses, "brain_mask")
        
    # TODO Get dataset_description.json and put it in derivs_dir

    return j_args


def copy_to_derivatives_dir(file_to_copy, derivs_dir, sub_ses, new_fname_part):
    """
    Copy file_to_copy into derivs_dir and rename it with the other 2 arguments
    :param file_to_copy: String, path to existing file to copy to derivs_dir
    :param derivs_dir: String, path to existing directory to copy file into
    :param sub_ses: String, the subject and session connected by an underscore
    :param new_fname_part: String to add to the end of the new filename
    """
    copy_and_rename_file(file_to_copy, os.path.join(
        derivs_dir, sub_ses + "_space-orig_desc-{}.nii.gz".format(new_fname_part)
    ))


def make_asegderived_mask(aseg_dir):
    """
    Create mask file(s) derived from aseg file(s) in aseg_dir
    :param aseg_dir: String, valid path to existing directory with output files
                     from chirality correction
    :return: List of strings; each is a valid path to an aseg mask file
    """
    # Only make masks from chirality-corrected nifti files
    aseg = glob(os.path.join(aseg_dir, 'sub-*_aseg.nii.gz'))  
    aseg.sort()

    # binarize, fillh, and erode aseg to make mask:
    mask_out_files = list()
    for aseg_file in aseg:
        mask_out_files.append(os.path.join(
            aseg_dir, '{}_mask.nii.gz'.format(aseg_file.split('.nii.gz')[0])
        ))
        maths = fsl.ImageMaths(in_file=aseg_file,  # (anat file)
                               op_string=('-bin -dilM -dilM -dilM -dilM '
                                          '-fillh -ero -ero -ero -ero'),
                               out_file=mask_out_files[-1])
        maths.run()
    return mask_out_files


def run_nibabies(j_args, logger):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    """
    # Get nibabies options from parameter file and turn them into flags
    nibabies_args = [j_args["common"]["age_months"], ]
    for nibabies_arg in ["cifti_output", "work_dir"]:
        nibabies_args.append(as_cli_arg(nibabies_arg))
        nibabies_args.append(j_args["nibabies"][nibabies_arg])  # TODO Ensure that all common args required by nibabies are added

    # Check whether aseg and mask files were produced by BIBSnet
    glob_path = os.path.join(j_args["optional_out_dirs"]["BIBSnet"],
                             "*{}*.nii.gz")
    aseg_glob = glob(glob_path.format("aseg"))
    mask_glob = glob(glob_path.format("mask"))
    if aseg_glob and mask_glob:
        derivs = ["--derivatives", j_args["optional_out_dirs"]["BIBSnet"]]
    else:
        derivs = list()
        warn_user_biconditional(
            aseg_glob, mask_glob, "mask", "aseg", logger,
            ("Missing {{}} files in {}\nNow running nibabies with JLF and "
             "without BIBSnet.".format(j_args["optional_out_dirs"]["BIBSnet"]))
        )
    
    # Run nibabies
    subprocess.check_call([j_args["nibabies"]["script_path"],
                           j_args["common"]["bids_dir"],
                           j_args["optional_out_dirs"]["nibabies"],
                           "participant", *derivs, *nibabies_args])
    return j_args


def run_XCPD(j_args, logger):
    """
    [summary] 
    :param j_args: Dictionary containing all args from parameter .JSON file
    :return: j_args, unchanged
    """
    subprocess.check_call((   # TODO Ensure that all "common" and "XCPD" args required by XCPD are added
        "singularity", "run", "--cleanenv",
        "-B", j_args["optional_out_dirs"]["nibabies"] + ":/data:ro",
        "-B", j_args["optional_out_dirs"]["XCPD"] + ":/out",
        "-B", j_args["XCPD"]["work_dir"] + ":/work",
        "/home/faird/shared/code/external/pipelines/ABCD-XCP/xcp-abcd_unstable01102022.sif",  # TODO Make this an import and/or a parameter
        "/data", "/out", "--cifti", "-w", "/work", "--participant-label",
        j_args["common"]["participant_label"]
    ))
    return j_args


if __name__ == '__main__':
    main()
