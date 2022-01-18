#!/usr/bin/env python3
# coding: utf-8

"""
CABINET
Created: 2021-11-12
Updated: 2022-01-18
"""

# Import standard libraries
import argparse
from datetime import datetime
from glob import glob
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


# Constants: Paths to this dir and level 1 analysis script
WRAPPER_LOC = '--wrapper-location'
SCRIPT_DIR = find_myself(WRAPPER_LOC)
LR_REGISTR_PATH = os.path.join(SCRIPT_DIR, "bin", "LR_mask_registration.sh")

# Custom local imports
from src.utilities import (
    as_cli_arg, crop_images, ensure_dict_has, exit_with_time_info,
    extract_from_json, resize_images, run_all_stages, valid_readable_json
)
from src.img_processing.correct_chirality import correct_chirality


def main():
    # Time how long the script takes and get command-line arguments from user 
    start_time = datetime.now()
    STAGES = {"preBIBSnet": run_preBIBSnet, "BIBSnet": run_BIBSnet,
              "postBIBSnet": run_postBIBSnet, "nibabies": run_nibabies,
              "XCP": run_XCP}
    json_args = get_params_from_JSON(list(STAGES.keys()))

    print(json_args)  # TODO REMOVE LINE

    run_all_stages(STAGES, json_args["stages"]["start"],
                   json_args["stages"]["end"], json_args) # TODO Ensure that each stage only needs j_args?

    # Show user how long the pipeline took and end the pipeline here
    exit_with_time_info(start_time)


def get_params_from_JSON(stage_names):
    """
    :param stage_names: List of strings where each names a stage to run
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
        choices=stage_names, default=stage_names[0]
    )
    parser.add_argument(
        "-end", "--ending-stage", dest="end",
        choices=stage_names, default=stage_names[-1]
    )
    cli_args = parser.parse_args()
    return validate_json_args(extract_from_json(cli_args.parameter_json),
                              cli_args.start, cli_args.end, parser)


def validate_json_args(j_args, start, end, parser):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    """
    # TODO Validate types in j_args; e.g. validate that nibabies[age_months] is an int

    j_args["common"] = ensure_dict_has(
        j_args["common"], "age_months",
        read_age_from_participants_tsv(j_args)
        # TODO Figure out which column in the participants.tsv file has the age_months value
    )
    j_args["stages"] = {"start": start, "end": end}
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
        "participants.tsv"), sep="\t", dtype=columns)

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


def run_preBIBSnet(j_args):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    """
    subj_ID = "sub-{}".format(j_args["common"]["participant_label"])
    session = "ses-{}".format(j_args["common"]["session"])
    subject_dir = os.path.join(j_args["common"]["bids_dir"], subj_ID)

    # Make working directories to run pre-BIBSnet processing in
    subj_work_dir = os.path.join(
        j_args["common"]["BIBSnet_work_dir"], subj_ID
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
            shutil.copy2(eachfile, new_fpath)
    
    # Crop and resize images
    crop_images(work_dirs["BIDS_data"], work_dirs["cropped"])
    os.chmod(work_dirs["cropped"], 0o775)
    ref_img = os.path.join(SCRIPT_DIR, 'data', 'test_subject_data', '1mo',
                           'sub-00006_T1w_acpc_dc_restore.nii.gz')
    id_mx = os.path.join(SCRIPT_DIR, 'data', 'identity_matrix.mat')
    resize_images(work_dirs["cropped"], work_dirs["resized"], ref_img, id_mx)


def run_BIBSnet(j_args):
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


def run_postBIBSnet(j_args):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    """
    # Template selection values
    age_months = j_args["common"]["age_months"]
    t1or2 = 2 if age_months < 22 else 1
    if age_months > 33:
        age_months = "34-38"

    # Paths for left & right registration
    chiral_dir = os.path.join(SCRIPT_DIR, "data", "chirality_masks")
    template_head = os.path.join(chiral_dir, "{}mo_T{}w_acpc_dc_restore.nii.gz")
    template_mask = os.path.join(chiral_dir, "{}mo_template_LRmask.nii.gz")
    subject_head_path = os.path.join(
        j_args["BIBSnet"]["output_dir"],
        "{}_acq-T1inT2".format(j_args["common"]["participant_label"]) # TODO Figure out / double-check the BIBSnet output file name for this participant
    )

    # Run left & right registration
    subprocess.check_call((LR_REGISTR_PATH, subject_head_path,
                           template_head.format(age_months, t1or2),
                           template_mask.format(age_months)))

    # Chirality correction
    segment_lookup_table = os.path.join(SCRIPT_DIR, "data", "look_up_tables",
                                        "FreeSurferColorLUT.txt")
    left_right_mask_nifti_file = "LRmask.nii.gz"  # TODO Make the --output path in LR_mask_registration.sh an absolute path and copy it here
    nifti_output_file_path = os.path.basename(subject_head_path) + "-corrected"  # TODO Figure out the directory path for this one too
    correct_chirality(subject_head_path, segment_lookup_table,
                      left_right_mask_nifti_file, nifti_output_file_path)
    return j_args


def make_mask(j_args):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    """
    base_dir = j_args["nibabies"]["work_dir"]
    os.chdir(base_dir)
    
    aseg = glob('sub-*_aseg.nii.gz')
    aseg.sort()

    aseg_spec = dict()
    for specifier in ("temp", "sub", "ses"):
        aseg_spec[specifier] = list()
        for i in range(len(aseg)):
            aseg_spec[specifier].append(aseg[i])

    # Step 1: Dilate aseg to make mask
    op_strings = ['{}-ero'.format(xtra) for xtra in
                  ('', '-bin -dilM -dilM -dilM -dilM -fillh -ero -ero ')]
    for sub, ses in zip(aseg_spec['sub'], aseg_spec['ses']):
        anatfiles = ['{}_{}_aseg{}.nii.gz'.format(sub, ses, uniq)
                     for uniq in ('', '_mask_dil', '_mask')]
        for i in range(1):
            img_maths = fsl.ImageMaths(in_file=anatfiles[i],
                                       out_file=anatfiles[i + 1],
                                       op_string=op_strings[i])
            img_maths.run()


def make_mask_Luci_original(j_args):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    """
    base_dir = j_args["nibabies"]["work_dir"]
    os.chdir(base_dir)
    
    aseg = glob('sub-*_aseg.nii.gz')
    aseg.sort()

    temp = []
    for i in aseg:
        temp.append(i.split('_'))
    sub = []
    for i in range(len(temp)):
        sub.append(temp[i][0])
    ses = []
    for i in range(len(temp)):
        ses.append(temp[i][1])

    ##step 1: dilate aseg to make mask:
    for sub, ses in zip(sub, ses):
        anatfile = '{}_{}_aseg.nii.gz'.format(sub,ses)
        img_maths = fsl.ImageMaths(in_file=anatfile, op_string='-bin -dilM -dilM -dilM -dilM -fillh -ero -ero -ero',
                                   out_file='{}_{}_aseg_mask_dil.nii.gz'.format(sub,ses))
        img_maths.run()
    
        anatfile = '{}_{}_aseg_mask_dil.nii.gz'.format(sub, ses)
        img_maths = fsl.ImageMaths(in_file=anatfile, op_string='-ero',
                                   out_file='{}_{}_aseg_mask.nii.gz'.format(sub, ses))
        img_maths.run()


def run_nibabies(j_args):
    segmentation = j_args["segmentation"]

    # Put just the mask and the segmentation (and nothing else) into a directory to run nibabies
    if j_args["common"]["age_months"] <= 8:
        mask = make_mask(j_args, segmentation)  # Luci has a script for this - we'll just use it as a blueprint
        run_nibabies_command(j_args, mask, segmentation)
    else:
        run_nibabies_command(j_args)
    return j_args


def run_nibabies_command(j_args, *args):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    """
    # Get nibabies options from parameter file and turn them into flags
    nibabies_args = list()
    for nibabies_arg in [j_args["common"]["age_months"],
                         j_args["nibabies"]["work_dir"], ]:
        nibabies_args.append(as_cli_arg(nibabies_arg))
        nibabies_args.append(nibabies_arg)

    # Get cohort number from template_description.json
    template_description = extract_from_json(
        j_args["common"]["template_description_json"]
    )
    cohorts = template_description["cohort"]  # TODO Remove this functionality when/if it's implemented in nibabies
    for cohort_num, cohort_details in cohorts:
        if (int(cohort_details["age"][0]) < int(j_args["common"]["age_months"])
                                            < int(cohort_details["age"][1])):
            cohort = cohort_num
            break

    # Run nibabies
    myseg_dir = os.path.join(j_args["common"]["derivatives"], "myseg")
    subprocess.check_call([
        j_args["nibabies"]["script_path"],
        j_args["common"]["bids_dir"],
        j_args["common"]["nibabies_dir"],
        "participant",
        "--derivatives", myseg_dir,
        "--output-spaces", "MNIInfant:cohort-{}".format(cohort),
        *nibabies_args
    ])


def run_XCP(j_args):
    return "placeholder"


if __name__ == '__main__':
    main()
