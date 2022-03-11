#!/usr/bin/env python3
# coding: utf-8

"""
Common source for utility functions used by CABINET :)
Greg Conan: gconan@umn.edu
Created: 2021-11-12
Updated: 2022-03-10
"""

# Import standard libraries
import argparse
import json
import nibabel as nib
from nipype.interfaces import fsl
import numpy as np
import os
import pdb  # TODO Remove this line, which is used for debugging by adding pdb.set_trace() before a problematic line
import shutil
import subprocess
import sys
from datetime import datetime  # for seeing how long scripts take to run
from glob import glob

# Chirality-checking constants
CHIRALITY_CONST = dict(UNKNOWN=0, LEFT=1, RIGHT=2, BILATERAL=3)
LEFT = "Left-"
RIGHT = "Right-"

# Other constants: Directory containing the main pipeline script, and 
# SLURM-/SBATCH-related arguments' default names
SCRIPT_DIR = os.path.dirname(os.path.dirname(__file__))


def align_ACPC_1_img(j_args, logger, xfm_ACPC_args, roi2full, output_var, t):
    """ 
    Functionality copied from the DCAN Infant Pipeline:
    github.com/DCAN-Labs/dcan-infant-pipeline/blob/master/PreFreeSurfer/scripts/ACPCAlignment_with_crop.sh
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :param xfm_ACPC_args: Dictionary mapping strings (ACPC input arguments'
                          names) to strings (ACPC arguments, file/dir paths)
    :param roi2full: String, valid path to existing roi2full.mat file
    :param output_var: String (with {} in it), a key in xfm_ACPC_args mapped to
                       the T1w and T2w valid output image file path strings 
    :param t: Int, either 1 or 2 (to signify T1w or T2w respectively)
    :return: String, valid path to newly-made output image file
    """
    # Get paths to ACPC alignment's ref image, output dir, and output images
    mni_ref_img_path = xfm_ACPC_args["ref_ACPC"].format(t)
    work_dir = xfm_ACPC_args["out_dir"]  # Working directory for ACPC alignment
    input_img = xfm_ACPC_args["T{}w_img".format(t)]  # Cropped img, ACPC input
    in_work_dir = lambda fname: os.path.join(work_dir, fname)  # work_dir is a preBIBSnet working directory
    output_img =  xfm_ACPC_args[output_var.format(t)]

    run_FSL_sh_script(j_args, logger, "flirt", "-interp", "spline",  
                      "-ref", mni_ref_img_path, "-in", input_img,
                      "-omat", in_work_dir("roi2std.mat"),  # TODO Name this differently to save T1w and T2w outputs separately
                      "-out", in_work_dir("acpc_final.nii.gz"),
                      "-searchrx", "-45", "45", "-searchry", "-30", "30",
                      "-searchrz", "-30", "30")

    # Invert roi2full to get full2roi
    run_FSL_sh_script(j_args, logger, "convert_xfm", "-inverse", roi2full,
                      "-omat", in_work_dir("full2roi.mat"))

    run_FSL_sh_script(  # Combine ACPC-alignment with robustFOV output
        j_args, logger, "convert_xfm", "-omat", in_work_dir("full2std.mat"),
        "-concat", in_work_dir("roi2std.mat"), in_work_dir("full2roi.mat")
    )

    # Transform 12 dof matrix to 6 dof approximation matrix
    output_matrix = in_work_dir("rigidbody2std.mat")
    run_FSL_sh_script(j_args, logger, "aff2rigid", in_work_dir("full2std.mat"),  # TODO Name this differently to save T1w and T2w outputs separately
                      output_matrix)

    # Apply ACPC alignment to the data
    run_FSL_sh_script(j_args, logger, "applywarp", "--rel", "--interp=spline",
                      "-i", input_img, "-r", mni_ref_img_path,
                      "--premat=" + output_matrix,  # TODO Name this differently to save T1w and T2w outputs separately
                      "-o", output_img)


def always_true(*_):
    """
    This function is useful when e.g. every type except 1 has a corresponding
    input validation function, because this function is used for that extra 1 
    :return: True, regardless of what the input arguments are 
    """
    return True


def argify(argname, argval):
    """
    :param argname: String naming a parameter for a script called from terminal
    :param argval: Object to assign in string form as the value of the argument
    :return: String, a parameter assignment for a script called from terminal
    """
    return "--{}={}".format(argname, argval)


def as_cli_arg(arg_str):
    """
    :param arg_str: String naming a stored argument taken from the command line
    :return: String which is the command-line argument form of arg_str
    """
    return "--" + arg_str.replace("_", "-")


def as_cli_attr(cli_arg_str):
    """
    :param cli_arg_str: String in command-line argument form
    :return: cli_arg_str, but formatted as a stored command-line argument
    """
    return cli_arg_str.strip("-").replace("-", "_")


def calculate_eta(img_paths):
    """
    :param img_paths: Dictionary mapping "T1w" and "T2w" to strings that are
                      valid paths to the existing respective image files
    :return: Int(?), the eta value
    """
    # get the data from each nifti image as a flattened vector
    vectors = dict()
    for t in (1, 2):
        anat = "T{}w".format(t)
        vectors[anat] = reshape_volume_to_array(nib.load(img_paths[anat]))

    # mean value over all locations in both images
    m_grand = (np.mean(vectors["T1w"]) + np.mean(vectors["T2w"])) / 2

    # mean value matrix for each location in the 2 images
    m_within = (vectors["T1w"] + vectors["T2w"]) / 2

    sswithin = (sum(np.square(vectors["T1w"] - m_within))
                + sum(np.square(vectors["T2w"] - m_within)))
    sstot = (sum(np.square(vectors["T1w"] - m_grand))
             + sum(np.square(vectors["T2w"] - m_grand)))

    # NOTE SStot = SSwithin + SSbetween so eta can also be
    #      written as SSbetween/SStot
    return 1 - sswithin / sstot


def check_and_correct_region(should_be_left, region, segment_name_to_number,
                             new_data, chirality, floor_ceiling, scanner_bore):
    """
    Ensures that a voxel in NIFTI data is in the correct region by flipping
    the label if it's mislabeled
    :param should_be_left (Boolean): This voxel *should be on the LHS of the head
    :param region: String naming the anatomical region
    :param segment_name_to_number (map<str, int>): Map from anatomical regions 
                                                   to identifying numbers
    :param new_data (3-d in array): segmentation data passed by reference to be fixed if necessary
    :param chirality: x-coordinate into new_data
    :param floor_ceiling: y-coordinate into new_data
    :param scanner_bore: z-coordinate into new_data
    """
    # expected_prefix, wrong_prefix = (LEFT, RIGHT) if should_be_left else (RIGHT, LEFT)
    if should_be_left:
        expected_prefix = LEFT
        wrong_prefix = RIGHT
    else:
        expected_prefix = RIGHT
        wrong_prefix = LEFT
    if region.startswith(wrong_prefix):
        flipped_region = expected_prefix + region[len(wrong_prefix):]
        flipped_id = segment_name_to_number[flipped_region]
        new_data[chirality][floor_ceiling][scanner_bore] = flipped_id


def copy_and_rename_file(old_file, new_file):
    """
    Rename a file and copy it to a new location
    :param old_file: String, valid path to an existing file to copy
    :param new_file: String, valid path to what will be a copy of old_file
    """
    # shutil.move()
    # os.rename(shutil.copy2(old_file, os.path.dirname(new_file)), new_file)
    shutil.copy2(old_file, new_file)


def correct_chirality(nifti_input_file_path, segment_lookup_table,
                      left_right_mask_nifti_file, chiral_out_dir,
                      t1w_path, j_args, logger):
    """
    Creates an output file with chirality corrections fixed.
    :param nifti_input_file_path: String, path to a segmentation file with possible chirality problems
    :param segment_lookup_table: String, path to a FreeSurfer-style look-up table
    :param left_right_mask_nifti_file: String, path to a mask file that distinguishes between left and right
    :param nifti_output_file_path: String, path to location to write the corrected file
    :param t1w_path:
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    """
    sub_ses = get_subj_ID_and_session(j_args)  # subj_ID, session = 
    msg = "{} chirality correction on {}"
    nifti_output_file_path = os.path.join(chiral_out_dir,
                                          j_args["BIBSnet"]["aseg_outfile"]) 

    logger.info(msg.format("Running", nifti_input_file_path))
    free_surfer_label_to_region = get_id_to_region_mapping(segment_lookup_table)
    segment_name_to_number = {v: k for k, v in free_surfer_label_to_region.items()}
    img = nib.load(nifti_input_file_path)
    data = img.get_data()
    left_right_img = nib.load(left_right_mask_nifti_file)
    left_right_data = left_right_img.get_data()

    new_data = data.copy()
    data_shape = img.header.get_data_shape()
    left_right_data_shape = left_right_img.header.get_data_shape()
    width = data_shape[0]
    height = data_shape[1]
    depth = data_shape[2]
    assert \
        width == left_right_data_shape[0] and height == left_right_data_shape[1] and depth == left_right_data_shape[2]
    for i in range(width):
        for j in range(height):
            for k in range(depth):
                voxel = data[i][j][k]
                region = free_surfer_label_to_region[voxel]
                chirality_voxel = int(left_right_data[i][j][k])
                if not (region.startswith(LEFT) or region.startswith(RIGHT)):
                    continue
                if chirality_voxel == CHIRALITY_CONST["LEFT"] or chirality_voxel == CHIRALITY_CONST["RIGHT"]:
                    check_and_correct_region(
                        chirality_voxel == CHIRALITY_CONST["LEFT"], region, segment_name_to_number, new_data, i, j, k)
    fixed_img = nib.Nifti1Image(new_data, img.affine, img.header)
    nib.save(fixed_img, nifti_output_file_path)

    # Undo resizing right here (do inverse transform) using RobustFOV so padding isn't necessary; revert aseg to native space
    dummy_copy = "_dummy".join(split_2_exts(nifti_output_file_path))
    copy_and_rename_file(nifti_output_file_path, dummy_copy)
    concat_output = os.path.join(chiral_out_dir, "{}_concatenated.mat"  # TODO Give this a more descriptive name?
                                                 .format("_".join(sub_ses)))
    roi2full_path = os.path.join(j_args["optional_out_dirs"]["preBIBSnet"],
                                 *sub_ses, "cropped", "T1w",  # TODO Ask about this: Will we have to average the different T1w/roi2full.mat and T2w/roi2full.mat files?
                                 "roi2full.mat")  # TODO Define this path outside of stages because it's used by preBIBSnet and postBIBSnet 
    transformed_output = os.path.join(chiral_out_dir, "resize_to_T1w.mat")
    run_FSL_sh_script(j_args, logger, "convert_xfm", "-omat",
                      transformed_output, "-inverse",
                      j_args["optimal_resized"]["T1w"])  # NOTE postBIBSnet ran until here and then crashed on 2022-03-10

    # Invert transformed_T1w_out and transform it back to its original space  # TODO Do we need to invert T1w and T2w?
    run_FSL_sh_script(j_args, logger, "convert_xfm", "-omat", concat_output,
                      "-concat", roi2full_path, transformed_output)
    logger.info("Transforming {} image back to its original space"
                .format(dummy_copy))
    run_FSL_sh_script(j_args, logger, "flirt", dummy_copy, t1w_path,
                      "-applyxfm", "-init", concat_output,  # TODO -applyxfm might need to be changed to -applyisoxfm with resolution
                      "-o", nifti_output_file_path)
    logger.info(msg.format("Finished", nifti_input_file_path))


def create_anatomical_average(avg_params):
    """
    Creates a NIFTI file whose voxels are the average of the voxel values of the input files.
    :param avg_params: Dictionary with 4 keys:
    {"T1w_input": List (possibly empty) of t1 image file path strings
     "T2w_input": List (possibly empty) of t2 image file path strings
     "T1w_avg": String, average T1w output file path
     "T2w_avg": String, average T2w output file path}
    """   
    for t in (1, 2):
        if avg_params["T{}w_input".format(t)]:
            register_and_average_files(avg_params["T{}w_input".format(t)],
                                       avg_params["T{}w_avg".format(t)])


def create_avg_image(output_file_path, registered_files):
    """
    :param output_file_path: String, valid path to average image file to make
    :param registered_files: List of strings; each is a valid path to an
                             existing image file to add to the average
    """
    np.set_printoptions(precision=2, suppress=True)  # Set numpy to print only 2 decimal digits for neatness
    first_nifti_file = registered_files[0]
    n1_img = nib.load(first_nifti_file)
    header = n1_img.header
    data_dtype = header.get_data_dtype()
    sum_matrix = n1_img.get_fdata().copy()
    n = len(registered_files)
    for j in range(1, n):
        img = nib.load(registered_files[j])
        data = img.get_fdata().copy()
        sum_matrix += data
    avg_matrix = sum_matrix / n
    if data_dtype == np.int16:
        avg_matrix = avg_matrix.astype(int)
    new_header = n1_img.header.copy()
    new_img = nib.nifti1.Nifti1Image(avg_matrix, n1_img.affine.copy(), header=new_header)
    nib.save(new_img, output_file_path)


def crop_image(input_avg_img, output_crop_img, j_args, logger):
    """
    Run robustFOV to crop image
    :param input_avg_img: String, valid path to averaged (T1w or T2w) image
    :param output_crop_img: String, valid path to save cropped image file at
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return: String, path to roi2full.mat file in same dir as output_crop_img
    """
    output_crop_dir = os.path.dirname(output_crop_img)
    roi2full = os.path.join(output_crop_dir, "roi2full.mat")  # TODO Define this path outside of stages because it's used by preBIBSnet and postBIBSnet
    run_FSL_sh_script(j_args, logger, "robustfov", "-i", input_avg_img, 
                      "-m", roi2full, "-r", output_crop_img,
                      "-b", j_args["preBIBSnet"]["brain_z_size"])
    return roi2full


def dict_has(a_dict, a_key):
    """
    :param a_dict: Dictionary (any)
    :param a_key: Object (any)
    :return: True if and only if a_key is mapped to something truthy in a_dict
    """
    return a_key in a_dict and a_dict[a_key]


def ensure_dict_has(a_dict, a_key, new_value):
    """
    :param a_dict: Dictionary (any)
    :param a_key: Object which will be a key in a_dict
    :param new_value: Object to become the value mapped to a_key in a_dict
                      unless a_key is already mapped to a value
    :return: a_dict, but with a_key mapped to some value
    """
    if not dict_has(a_dict, a_key):
        a_dict[a_key] = new_value
    return a_dict


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


def extract_from_json(json_path):
    """
    :param json_path: String, a valid path to a real readable .json file
    :return: Dictionary, the contents of the file at json_path
    """
    with open(json_path, "r") as infile:
        return json.load(infile)


def get_and_make_preBIBSnet_work_dirs(j_args):
    """ 
    :param j_args: Dictionary containing all args from parameter .JSON file
    :return: Dictionary mapping j_args[preBIBSnet] dir keys to preBIBSnet
             subdirectories and "avg" to this dictionary:
             {"T?w_input": Lists (possibly empty) of T?w img file path strings
              "T?w_avg": Strings, average T?w output file paths}
    """
    # Get subject ID, session, and directory of subject's BIDS-valid input data
    sub_ses = get_subj_ID_and_session(j_args)  # subj_ID, session = 

    # Get and make working directories to run pre-BIBSnet processing in
    preBIBSnet_paths = {"parent": os.path.join(
                            j_args["optional_out_dirs"]["preBIBSnet"], *sub_ses
                        )}
    for jarg, work_dirname in j_args["preBIBSnet"].items():

        # Just get the preBIBSnet parameters that are subdirectories
        split = jarg.split("_")
        if split[-1] == "dir":

            # Build path to, and make, preBIBSnet working directories
            preBIBSnet_paths[split[0]] = os.path.join(
                preBIBSnet_paths["parent"], work_dirname
            )
            os.makedirs(preBIBSnet_paths[split[0]], exist_ok=True)
            # os.chmod(preBIBSnet_paths[split[0]], 0o775)

    # Build paths to BIDS anatomical input images and (averaged, 
    # nnU-Net-renamed) output images
    preBIBSnet_paths["avg"] = dict()
    for t in (1, 2):
        preBIBSnet_paths["avg"]["T{}w_input".format(t)] = list()
        for eachfile in glob(os.path.join(j_args["common"]["bids_dir"],
                                          *sub_ses, "anat", "*T{}w*.nii.gz"
                                                            .format(t))):
            preBIBSnet_paths["avg"]["T{}w_input".format(t)].append(eachfile)
        avg_img_name = "{}_000{}{}".format("_".join(sub_ses), t-1, ".nii.gz")
        preBIBSnet_paths["avg"]["T{}w_avg".format(t)] = os.path.join(  
            preBIBSnet_paths["averaged"], avg_img_name  
        )  
  
        # Get paths to, and make, cropped image subdirectories  
        crop_dir = os.path.join(preBIBSnet_paths["cropped"], "T{}w".format(t))  
        preBIBSnet_paths["crop_T{}w".format(t)] = os.path.join(crop_dir,  
                                                               avg_img_name)  
        os.makedirs(crop_dir, exist_ok=True)
    return preBIBSnet_paths


def get_and_print_time_since(event_name, event_time, logger):
    """
    Print and return a string showing how much time has passed since the
    current running script reached a certain part of its process
    :param event_name: String to print after "Time elapsed since "
    :param event_time: datetime object representing a time in the past
    :return: String with an easily human-readable message showing how much time
             has passed since {event_time} when {event_name} happened.
    """
    timestamp = ("Time elapsed since {}: {}"
                 .format(event_name, datetime.now() - event_time))
    logger.info(timestamp)
    return timestamp


def get_default_ext_command(cmd_name):
    """
    Try to get valid path to external software command file without user input
    :param cmd_name: String naming the executable command file
    :return: String, path to the command if the user has the command alias in
             their .bashrc / $PATH; otherwise None
    """
    try:  # If the command path is already defined, then use it
        cmd = subprocess.check_output(("which", cmd_name)
                                      ).decode("utf-8").split()[-1]
    except subprocess.CalledProcessError:
        cmd = None
    return cmd


def get_id_to_region_mapping(mapping_file_name, separator=None):
    """
    Author: Paul Reiners
    Create a map from region ID to region name from a from a FreeSurfer-style
    look-up table. This function parses a FreeSurfer-style look-up table. It
    then returns a map that maps region IDs to their names.
    :param mapping_file_name: String, the name or path to the look-up table
    :param separator: String delimiter separating parts of look-up table lines
    :return: Dictionary, a map from the ID of a region to its name
    """
    with open(mapping_file_name, 'r') as infile:
        lines = infile.readlines()

    id_to_region = {}
    for line in lines:
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        if separator:
            parts = line.split(separator)
        else:
            parts = line.split()
        region_id = int(parts[0])
        region = parts[1]
        id_to_region[region_id] = region
    return id_to_region


def get_stage_name(stage_fn):
    """ 
    :param stage_fn: Function to run one stage of CABINET. Its name must start
                     with "run_", e.g. "run_nibabies" or "run_preBIBSnet"
    :return: String naming the CABINET stage to run
    """
    return stage_fn.__name__[4:]


def get_sub_base(j_args, run_num=None):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param run_num: Whole number as an int or string defining which run this is
    :return: String identifying a subject, session, task, and maybe run
    """
    parts = [get_subj_ses(j_args), "task-" + j_args["common"]["task_id"]]
    if run_num is not None:
        parts.append("run-{}".format(run_num))
    return "_".join(parts)


def get_subj_ID_and_session(j_args):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :return: List of 2 strings (subject ID and session from parameter file,
             with their correct "sub-" and "ses-" prefixes) if the parameter
             file has a session, otherwise just with the prefixed subject ID
    """ 
    sub = ensure_prefixed(j_args["common"]["participant_label"], "sub-")
    return [sub, ensure_prefixed(j_args["common"]["session"], "ses-")
            ] if j_args["common"]["session"] else [sub]


def get_subj_ses(j_args):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :return: String combining subject ID and session from parameter file
    """
    return "_".join(get_subj_ID_and_session(j_args))


def glob_and_copy(dest_dirpath, *path_parts_to_glob):
    """
    Collect all files matching a glob string, then copy those files
    :param dest_dirpath: String, a valid path of a directory to copy files into
    :param path_parts_to_glob: Unpacked list of strings which join to form a
                               glob string of a path to copy files from
    """
    for file_src in glob(os.path.join(*path_parts_to_glob)):
        shutil.copy(file_src, dest_dirpath)


def optimal_realigned_imgs(xfm_imgs_non_ACPC, xfm_imgs_ACPC_and_reg, logger):
    """
    Check whether the cost function shows that only the registration-T2-to-T1
    or the ACPC-alignment-and-T2-to-T1-registration is better (check whether
    ACPC alignment improves the T2-to-T1 registration; compare the T2-to-T1
    with and without first doing the ACPC registration)
    :param logger: logging.Logger object to raise warning
    """
    msg = "Using {} T2w-to-T1w registration for resizing.\nT1w: {}\nT2w: {}"
    if calculate_eta(xfm_imgs_non_ACPC) > calculate_eta(xfm_imgs_ACPC_and_reg):
        optimal_resize = xfm_imgs_non_ACPC
        logger.info(msg.format("only", optimal_resize["T1w"],
                               optimal_resize["T2w"]))  # TODO Verify that these print the absolute path
    else:
        optimal_resize = xfm_imgs_ACPC_and_reg
        logger.info(msg.format("ACPC and", optimal_resize["T1w"],
                               optimal_resize["T2w"]))  # TODO Verify that these print the absolute path
    return optimal_resize
                                       

def register_and_average_files(input_file_paths, output_file_path):
    reference = input_file_paths[0]
    if len(input_file_paths) > 1:
        registered_files = register_files(input_file_paths, reference)

        create_avg_image(output_file_path, registered_files)
    else:
        shutil.copyfile(reference, output_file_path)


def register_files(input_file_paths, reference):
    registered_files = [reference]
    flt = fsl.FLIRT(bins=640, cost_func='mutualinfo')
    flt.inputs.reference = reference
    flt.inputs.output_type = "NIFTI_GZ"
    for structural in input_file_paths[1:]:
        flt.inputs.in_file = structural
        print(flt.cmdline)
        out_index = flt.cmdline.find('-out')
        start_index = out_index + len('-out') + 1
        end_index = flt.cmdline.find(' ', start_index)
        out = flt.cmdline[start_index:end_index]
        registered_files.append(out)
        res = flt.run()
        stderr = res.runtime.stderr
        if stderr:
            err_msg = f'flirt error message: {stderr}'
            raise RuntimeError(err_msg)
    return registered_files
    

def registration_T2w_to_T1w(j_args, logger, xfm_vars):
    """
    T2w to T1w registration for use in preBIBSnet
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :param xfm_vars: Dictionary containing paths to files used in registration
    :return: Dictionary mapping "T1w" and "T2w" to their respective newly
             registered image file paths
    """
    # Make T2w-to-T1w matrix
    t2_to_t1_matrix = os.path.join(xfm_vars["out_dir"], "T2toT1.mat") 
    run_FSL_sh_script(j_args, logger, "flirt", "-ref", xfm_vars["T1w_img"],
                      "-in", xfm_vars["T2w_img"], "-omat", t2_to_t1_matrix)

    # Make transformed T1ws and T2ws
    xfms_matrices = {"T1w": xfm_vars["ident_mx"], "T2w": t2_to_t1_matrix}
    transformed_images = dict()
    for img in xfms_matrices.keys():
        tmpl = img + "_template"
        transformed_images[tmpl] = os.path.join(
            xfm_vars["out_dir"], "{}_to_BIBS_template.mat".format(img)
        )
        transformed_images[img] = xfm_vars["output_{}_img".format(img)]

        run_FSL_sh_script(
            j_args, logger, "flirt", "-in", xfm_vars["{}_img".format(img)],
            "-ref", xfm_vars["ref_reg"], "-applyisoxfm", 
            xfm_vars["resolution"], "-init", xfms_matrices[img],
            "-o", transformed_images[img], "-omat", transformed_images[tmpl]
        )
    return transformed_images


def reshape_volume_to_array(array_img):
    """ 
    :param array_img: nibabel.Nifti1Image (or Nifti2Image?)
    :return: numpy.ndarray (?), array_img's data matrix but flattened
    """
    image_data = array_img.get_fdata()
    return image_data.flatten()


def resize_images(cropped_imgs, output_dir, ref_images, ident_mx,
                  roi2full, j_args, logger):
    """
    Resize the images to match the dimensions of images trained in the model,
    and ensure that the first image (presumably a T1) is co-registered to the
    second image (presumably a T2) before resizing. Use multiple alignments
    of both images, and return whichever one is better (higher eta squared)
    :param cropped_imgs: Dictionary mapping ints, (T) 1 or 2, to strings (valid
                         paths to existing image files to resize)
    :param output_dir: String, valid path to a dir to save resized images into
    :param ref_images: Dictionary mapping string keys to valid paths to real
                       image file strings for "ACPC" (alignment) and (T2-to-T1)
                       "reg"(istration) for flirt to use as a reference image.
                       The ACPC string has a "{}" in it to represent (T) 1 or 2
    :param ident_mx: String, valid path to existing identity matrix .mat file
    :param roi2full: String, valid path to existing roi2full.mat file
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    """
    # Build dictionaries of variables used for image transformations with and
    # without ACPC alignment
    xfm_vars = {"out_dir": os.path.join(output_dir, "xfms"), "resolution": "1",
                "ident_mx": ident_mx, **ref_images} # "ref_img": ref_img_path,
    xfm_ACPC_args = xfm_vars.copy()
    xfm_ACPC_args["out_dir"] = os.path.join(output_dir, "ACPC_align")
    out_var = "output_T{}w_img"
    for t, img_path in cropped_imgs.items():
        xfm_vars["T{}w_img".format(t)] = img_path
        xfm_ACPC_args["T{}w_img".format(t)] = img_path
        out_img = os.path.join(xfm_vars["out_dir"],
                               "{}" + os.path.basename(img_path))
        xfm_vars[out_var.format(t)] = out_img.format("")
        xfm_ACPC_args[out_var.format(t)] = out_img.format("ACPC_")
    if j_args["common"]["verbose"]:
        msg_xfm = "Arguments for {}ACPC image transformation:\n{}"
        logger.info(msg_xfm.format("non-", xfm_vars))
        logger.info(msg_xfm.format("", xfm_ACPC_args))

    # Make output directories for transformed images
    for each_xfm_vars_dict in (xfm_vars, xfm_ACPC_args):
        os.makedirs(each_xfm_vars_dict["out_dir"], exist_ok=True)

    # Comparison of T2w to T1w registration approaches
    # Do direct T1w-T2w alignment
    xfm_imgs_non_ACPC = registration_T2w_to_T1w(j_args, logger, xfm_vars)
    for t in (1, 2):
        if j_args["common"]["verbose"]:
            logger.info("Now resizing " + str(xfm_vars["T{}w_img".format(t)]))
        
        # Run ACPC alignment
        align_ACPC_1_img(j_args, logger, xfm_ACPC_args, roi2full, out_var, t)

    # T1w-T2w alignment of ACPC-aligned images
    xfm_ACPC_and_registered_imgs = registration_T2w_to_T1w(j_args, logger,
                                                           xfm_ACPC_args)   # TODO Save ACPC T1w and T2w images output from this function to j_args[optional_out_dirs][preBIBSnet]/resized/ACPC_align/ dir

    # Return the best of the 2 resized images
    return optimal_realigned_imgs(xfm_imgs_non_ACPC,
                                  xfm_ACPC_and_registered_imgs, logger)


def run_FSL_sh_script(j_args, logger, fsl_fn_name, *fsl_args):
    """
    Run any FSL function in a Bash subprocess, unless its outputs exist and the
    parameter file said not to overwrite outputs
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :param fsl_fn_name: String naming the FSL function which is an
                        executable file in j_args[common][fsl_bin_path]
    """
    # FSL command to (maybe) run in a subprocess
    to_run = [os.path.join(j_args["common"]["fsl_bin_path"], fsl_fn_name)
              ] + [str(f) for f in fsl_args]

    # If the output image(s) exist(s) and j_args[common][overwrite] is False,
    # then skip the entire FSL command and tell the user
    outputs = list()
    for i in range(len(to_run)):
        if to_run[i].strip('-') in ("o", "omat", "out"):
            outputs.append(to_run[i + 1])
    if ((not j_args["common"]["overwrite"]) and outputs and
            [os.path.exists(output) for output in outputs]):
        if j_args["common"]["verbose"]:
            logger.info("Skipping FSL {} command because its output image(s) "
                        "listed below exist(s) and overwrite=False.\n{}"
                        .format(fsl_fn_name, "\n".join(outputs)))

    else:  # Otherwise, just run the FSL command
        if j_args["common"]["verbose"]:
            logger.info("Now running FSL command:\n{}"
                        .format(" ".join(to_run)))
        subprocess.check_call(to_run)


def run_all_stages(all_stages, start, end, params_for_every_stage, logger):
    """
    Run stages sequentially, starting and ending at stages specified by user
    :param all_stages: List of functions in order where each runs one stage
    :param start: String naming the first stage the user wants to run
    :param end: String naming the last stage the user wants to run
    :param params_for_every_stage: Dictionary of all args needed by each stage
    :param logger: logging.Logger object to show messages and raise warnings
    """
    running = False
    for stage in all_stages:
        name = get_stage_name(stage)
        if name == start:
            running = True
        if running:
            stage_start = datetime.now()
            stage(params_for_every_stage, logger)
            get_and_print_time_since(name + " started", stage_start, logger)
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
                    

def valid_float_0_to_1(val):
    """
    :param val: Object to check, then throw an error if it is invalid
    :return: val if it is a float between 0 and 1 (otherwise invalid)
    """
    return validate(val, lambda x: 0 <= float(x) <= 1, float,
                    "{} is not a number between 0 and 1")


def valid_output_dir(path):
    """
    Try to make a folder for new files at path; throw exception if that fails
    :param path: String which is a valid (not necessarily real) folder path
    :return: String which is a validated absolute path to real writeable folder
    """
    return validate(path, lambda x: os.access(x, os.W_OK),
                    valid_readable_dir, "Cannot create directory at {}",
                    lambda y: os.makedirs(y, exist_ok=True))


def valid_output_dir_or_false(path):
    """
    Try to make a folder for new files at path, unless "path" is just False.
    Throw exception if that fails
    :param path: String which should be either a valid (not necessarily real)
                 folder path or False
    :return: Either False or a validated absolute path to real writeable folder
    """
    return path if path is False else valid_output_dir(path)


def valid_positive_float(to_validate):
    """
    Throw argparse exception unless to_validate is a positive float
    :param to_validate: Object to test whether it is a positive float
    :return: to_validate if it is a positive float
    """
    return validate(to_validate, lambda x: float(x) >= 0, float,
                    "{} is not a positive number")


def valid_readable_dir(path):
    """
    :param path: Parameter to check if it represents a valid directory path
    :return: String representing a valid directory path
    """
    return validate(path, os.path.isdir, valid_readable_file,
                    "Cannot read directory at '{}'")


def valid_readable_file(path):
    """
    Throw exception unless parameter is a valid readable filepath string. Use
    this, not argparse.FileType("r") which leaves an open file handle.
    :param path: Parameter to check if it represents a valid filepath
    :return: String representing a valid filepath
    """
    return validate(path, lambda x: os.access(x, os.R_OK),
                    os.path.abspath, "Cannot read file at '{}'")


def valid_readable_json(path):
    """
    :param path: Parameter to check if it represents a valid .json file path
    :return: String representing a valid .json file path
    """
    return validate(path, lambda _: os.path.splitext(path)[-1] == ".json",
                    valid_readable_file,
                    "'{}' is not a path to a readable .json file")


def valid_subj_ses(in_arg, pfx, name): 
    """
    :param in_arg: Object to check if it is a valid subject ID or session name
    :param pfx: String that's the prefix to an ID; "sub-" or "ses-"
    :param name: String describing what in_arg should be (e.g. "subject")
    :return: True if in_arg is a valid subject ID or session name; else False
    """
    return validate(in_arg, always_true, lambda x: ensure_prefixed(x, pfx),
                    "'{}'" + " is not a valid {}".format(name))


def valid_template_filename(fname):
    """
    :param fname: Parameter to check if it represents a .fsf file name
    :return: String representing the .fsf file name
    """
    return validate(fname, lambda x: os.path.splitext(x)[-1] == ".fsf",
                    lambda y: y, "'{}' is not an .fsf file name")


# TODO For nibabies --output-spaces type validation, see https://fmriprep.org/en/latest/spaces.html
# TODO Use --clean-env flag to prevent contamination of Singularity run by outside environment variables?
#   https://3.basecamp.com/5032058/buckets/21517584/messages/4545156874


def valid_time_str(in_arg):
    """
    :param in_arg: Object to check if it's a time string in the HH:MM:SS format
    :return: True if in_arg is a time limit string in that format; else False
    """
    try:
        split = in_arg.split(":")
        assert len(split) == 3
        for each_num in split:
            assert each_num.isdigit()
            assert int(each_num) >= 0
        return in_arg
    except (TypeError, AssertionError, ValueError):
        raise argparse.ArgumentTypeError("'{}' is not a valid time string"
                                         .format(in_arg))


def valid_whole_number(to_validate):
    """
    Throw argparse exception unless to_validate is a positive integer
    :param to_validate: Object to test whether it is a positive integer
    :return: to_validate if it is a positive integer
    """
    return validate(to_validate, lambda x: int(x) >= 0, int,
                    "{} is not a positive integer")


def validate(to_validate, is_real, make_valid, err_msg, prepare=None):
    """
    Parent/base function used by different type validation functions. Raises an
    argparse.ArgumentTypeError if the input object is somehow invalid.
    :param to_validate: String to check if it represents a valid object 
    :param is_real: Function which returns true iff to_validate is real
    :param make_valid: Function which returns a fully validated object
    :param err_msg: String to show to user to tell them what is invalid
    :param prepare: Function to run before validation
    :return: to_validate, but fully validated
    """
    try:
        if prepare:
            prepare(to_validate)
        assert is_real(to_validate)
        return make_valid(to_validate)
    except (OSError, TypeError, AssertionError, ValueError,
            argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError(err_msg.format(to_validate))


def validate_parameter_types(j_args, j_types, param_json, parser, stage_names):
    """
    Verify that every parameter in j_args is the correct data-type and the 
    right kind of value. If any parameter is invalid, crash with an error.
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param j_types: Dictionary mapping every argument in j_args to its type
    :param param_json: String, path to readable .JSON file with all parameters
    :param parser: argparse.ArgumentParser to raise error if anything's invalid
    :param stage_names: List of strings; each names a stage to run
    """
    # Define functions to validate arguments of each data type
    type_validators = {"bool": bool, "int": int,
                       "existing_directory_path": valid_readable_dir,
                       "existing_file_path": valid_readable_file,
                       "existing_json_file_path": valid_readable_json,
                       "float_0_to_1": valid_float_0_to_1,
                       "new_directory_path": valid_output_dir,
                       "new_file_path": always_true,  # TODO Make "valid_output_filename" function?
                       "optional_new_dirpath": valid_output_dir_or_false,
                       "positive_float": valid_positive_float,
                       "positive_int": valid_whole_number, 
                       "str": always_true}

    # Get a list of all stages after the last stage to run
    after_end = stage_names[stage_names.index(j_args["stage_names"]["end"])+1:]

    # Verify parameters in each section
    to_delete = list()
    for section_name, section_dict in j_types.items():

        # Skip the j_args sections for stages not being run
        if section_name in stage_names and section_name in after_end:
            to_delete.append(section_name)

        # Only include resource_management if we're in SLURM/SBATCH job(s)
        elif not (section_name == "resource_management"
                  and not j_args["meta"]["slurm"]):

            # Validate every parameter in the section
            for arg_name, arg_type in section_dict.items():
                validate_1_parameter(j_args, arg_name, arg_type, section_name,
                                    type_validators, param_json, parser)
            
    # Remove irrelevant parameters
    for section_name in to_delete:
        del j_args[section_name]
    return j_args


def validate_1_parameter(j_args, arg_name, arg_type, section_name,
                         type_validators, param_json, parser):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param arg_name: String naming the parameter to validate
    :param arg_type: Either a string naming the data type of the parameter to 
                     validate or a list of options the parameter must be in
    :param section_name: String that's a subcategory in the param_json file
    :param type_validators: Dict mapping each arg_type to a validator function
    :param param_json: String, path to readable .JSON file with all parameters
    :param parser: argparse.ArgumentParser to raise error if anything's invalid
    """
    to_validate = j_args[section_name][arg_name]  # Parameter to validate
    err_msg = ("'{}' is not a valid '{}' parameter in the '{}' section "
               "of {} (Problem: {})")  # Message for if to_validate is invalid
    try:
        # Run a type validation function unless arg_type is a list
        {str: type_validators[arg_type],

        # Verify that the parameter is a valid member of a choices list
         list: lambda pm: pm if pm in arg_type else parser.error(
            err_msg.format(pm, arg_name, section_name, param_json,
                            "Valid {} values: {}"
                            .format(arg_name, ", ".join(arg_type)))
         )
        }[type(arg_type)](to_validate)

    # If type validation fails then inform the user which parameter
    # has an invalid type and what the valid types are
    except (argparse.ArgumentTypeError, KeyError, TypeError, ValueError) as e:
        parser.error(err_msg.format(to_validate, arg_name,
                                    section_name, param_json, e))


def warn_user_of_conditions(warning, logger, **to_check):
    """
    Show warning message to user based on boolean conditions 
    :param warning: String with warning message to put problems into and log
    :param logger: logging.Logger object to raise warning
    """
    problems = list()
    for condition, problem in to_check.items():
        if condition:
            problems.append(problem)
    logger.warn(warning.format(" and ".join(problems)))


def will_run_stage(a_stage, start_stage, end_stage, all_stage_names):
    """
    :param a_stage: String naming a stage to run
    :param start_stage: String naming the first stage to run
    :param end_stage: String naming the last stage to run
    :param all_stage_names: List of strings; each names a stage to run
    :return: True if a_stage is between start_stage and end_stage; else False
    """
    return (all_stage_names.index(start_stage)
            <= all_stage_names.index(a_stage)
            <= all_stage_names.index(end_stage))