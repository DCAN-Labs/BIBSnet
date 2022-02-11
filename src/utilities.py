#!/usr/bin/env python3
# coding: utf-8

"""
Common source for utility functions used by CABINET :)
Greg Conan: gconan@umn.edu
Created: 2021-11-12
Updated: 2022-02-04
"""

# Import standard libraries
import argparse
import json
import logging
import nibabel as nib
from nipype.interfaces import fsl
import numpy as np
import os
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
SLURM_ARGS = ("account", "cpus", "memory", "print_progress", "sleep", "time")


def add_slurm_args_to(parser):
    """
    :param parser: argparse.ArgumentParser with some command-line arguments 
    :return: parser with all CLI arguments needed to run parallel SLURM jobs
    """
    default_cpus = 1
    default_gb_mem = 8
    default_sleep = 10
    default_time_limit = "01:00:00"
    parser.add_argument(
        "-A", "--account",
        help="Name of the account to submit the SBATCH job under."
    )
    parser.add_argument(
        "-c", "--cpus", type=valid_whole_number, default=default_cpus,
        help=("Number of CPUs to use for each Python job. By default, this "
              "argument's value will be {}.".format(default_cpus))
    )
    parser.add_argument(
        "-mem", "--memory", type=valid_whole_number, default=default_gb_mem,
        help=("Memory in gigabytes (GB) to assign to each sbatch job. The "
              "default number is {} GB.".format(default_gb_mem))
    )
    parser.add_argument(
        "-progress", "--print-progress", action="store_true",
        help=("Include this flag for the script to print updates about its "
              "progress at intervals defined by --sleep. This will also print "
              "every command that is run to submit a pipeline batch job.")
    )
    parser.add_argument(
        "-sleep", "--sleep", type=valid_whole_number, default=default_sleep,
        help=("Number of seconds to wait between batch job submissions. The "
              "default number is {}.".format(default_sleep))
    )
    parser.add_argument(
        "-time", "--time", metavar="SLURM_JOB_TIME_LIMIT",
        type=valid_time_str, default=default_time_limit,
        help=("Time limit for each automated_subset_analysis batch job. The "
              "time limit must be formatted specifically as HH:MM:SS where HH "
              "is hours, MM is minutes, and SS is seconds. {} is the default "
              "time limit.".format(default_time_limit))
    )
    return parser


def always_true(*args):
    """
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
    :param arg_str: String in command-line argument form
    :return: arg_str, but formatted as a stored argument from the command line
    """
    return cli_arg_str.strip("-").replace("-", "_")


def check_and_correct_region(should_be_left, region, segment_name_to_number,
                             new_data, chirality, floor_ceiling, scanner_bore):
    """
    Ensures that a voxel in NIFTI data is in the correct region by flipping
    the label if it's mislabeled
    :param should_be_left (Boolean): This voxel *should be on the LHS of the head
    :param region: String naming the anatomical region
    :param segment_name_to_number (map<str, int>): Map from anatomical regions to identifying numbers
    :param new_data (3-d in array): segmentation data passed by reference to be fixed if necessary
    :param chirality: x-coordinate into new_data
    :param floor_ceiling: y-coordinate into new_data
    :param scanner_bore: z-coordinate into new_data
    """
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
    os.rename(shutil.copy2(old_file, os.path.dirname(new_file)), new_file)


def correct_chirality(nifti_input_file_path, segment_lookup_table,
                      left_right_mask_nifti_file, nifti_output_file_path,
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

    # TODO Undo resizing right here (do inverse transform) using RobustFOV so padding isn't necessary; revert aseg to native space
    dummy_copy = nifti_output_file_path + "_dummy"
    copy_and_rename_file(nifti_output_file_path, dummy_copy)
    concat_output = "" # TODO
    roi2full_path = "" # TODO
    transformed_output = "resize_to_T1w.mat"  
    run_FSL_sh_script(j_args, "convert_xfm", "-omat", transformed_output,
                      "-inverse", j_args["transformed_images"]["T1w"])

    # Invert transformed_T1w_out and transform it back to its original space
    run_FSL_sh_script(j_args, "convert_xfm", "-omat", concat_output, "-concat",
                      roi2full_path, transformed_output)
    logger.info("Transforming {} image back to its original space"
                .format(dummy_copy))
    run_FSL_sh_script(j_args, "flirt", dummy_copy, t1w_path,
                      "-applyxfm", "-init", concat_output,  # TODO -applyxfm might need to be changed to -applyisoxfm with resolution
                      "-o", nifti_output_file_path)


def crop_images(input_avg_dir, output_crop_dir, j_args):
    """
    [summary] 
    :param input_avg_dir: String, valid path to existing input directory with
                          averaged (T1w and T2w) images
    :param output_crop_dir: String, valid path to existing output directory
                            to save cropped files into
    :param j_args: Dictionary containing all args from parameter .JSON file
    """
    run_FSL_sh_script(j_args, "robustfov", '-i', input_avg_dir, '-m',
                      os.path.join(output_crop_dir, 'roi2full.mat'),
                      '-r', os.path.join(output_crop_dir, 'robustroi.nii.gz'),
                      j_args["preBIBSnet"]["brain_size"])


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


def get_and_print_time_since(event_name, event_time):
    """
    Print and return a string showing how much time has passed since the
    current running script reached a certain part of its process
    :param event_name: String to print after "Time elapsed since "
    :param event_time: datetime object representing a time in the past
    :return: String with an easily human-readable message showing how much time
             has passed since {event_time} when {event_name} happened.
    """
    timestamp = ("\nTime elapsed since {}: {}"
                 .format(event_name, datetime.now() - event_time))
    print(timestamp)
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
    file = open(mapping_file_name, 'r')
    lines = file.readlines()

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


def get_optional_cli_args(cli_args, drop_slurm=False):
    """
    :param cli_args: Dictionary with all validated command-line arguments,
                     all of which are used by this function
    :param drop_slurm: True to exclude SLURM arguments; else False
    :return: List of most cli_args optional arguments and their values
    """
    optional_args = list()
    for arg in cli_args.keys():
        if cli_args[arg] and not (drop_slurm and arg in SLURM_ARGS):
            optional_args.append(as_cli_arg(arg))
            if isinstance(cli_args[arg], list):
                for el in cli_args[arg]:
                    optional_args.append(str(el))
            elif not isinstance(cli_args[arg], bool):
                optional_args.append(str(cli_args[arg]))
    return optional_args


def get_sbatch_args(cli_args, job):
    """
    :param cli_args: Dictionary containing all command-line arguments from user
    :param job: String 1-8 characters long naming the SBATCH job
    :return: List of strings, SLURM-related arguments to pass to the main
             script or level 1 analysis script for parallelization
    """
    return [argify("time", cli_args["time"]), "-c", str(cli_args["cpus"]),
            "-J", job, argify("mem", "{}gb".format(cli_args["memory"]))]


def get_stage_name(stage_fn):
    """ 
    :param stage_fn: Function to run one stage of CABINET. Its name starts
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
    :return: Tuple of 2 strings, subject ID and session from parameter file,
             with their correct "sub-" and "ses-" prefixes
    """
    sub = ensure_prefixed(j_args["common"]["participant_label"], "sub-")
    ses = ensure_prefixed(j_args["common"]["session"], "ses-")
    return sub, ses


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


def resize_images(input_dir, output_dir, reference_image_path, ident_mx,
                  logger, j_args):
    """
    Resize the images to match the dimensions of images trained in the model,
    and ensure that the first image (presumably a T1) is co-registered to the
    second image (presumably a T2) before resizing
    :param input_dir: String, valid path to existing directory containing
                      image files to resize
    :param output_dir: String, valid path to existing directory to save
                       resized image files into
    :param reference_image_path: String, valid path to existing image file for
                                 flirt to use as a reference image
    :param ident_mx: String, valid path to existing identity matrix .MAT file
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    """
    only_files = [f for f in os.listdir(input_dir)  # TODO Add comment explaining what this is
                  if os.path.isfile(os.path.join(output_dir, f))]

    # os.system("module load fsl")  # TODO This will only work on MSI, so change it (import it somehow?)

    # TODO Include option to also do the full HCP ACPC-only alignment using MNI template, then discard the worse one judged by a cost function, and log the choice that the cost function makes? 
    # See https://github.com/DCAN-Labs/dcan-infant-pipeline/blob/master/PreFreeSurfer/scripts/ACPCAlignment_with_crop.sh

    xfm_vars = {"xfms_out_dir": os.path.join(output_dir, "xfms"), "resln": "1",
                "ref_img": reference_image_path, "ident_mx": ident_mx}
    xfm_ACPC_args = xfm_vars.copy()
    out_var = "output_t{}w_img"
    t1or2 = {"0000": 1, "0001": 2}
    for eachfile in only_files:
        scan_type = eachfile.rsplit("_", 1)[-1].split(".", 1)[0]
        if j_args["common"]["verbose"]:
            print("Now resizing {}".format(eachfile))
        t = t1or2[scan_type]
        xfm_vars["t{}w_img".format(t)] = os.path.join(input_dir, eachfile) 
        xfm_vars[out_var.format(t)] = os.path.join(output_dir, eachfile)
        xfm_ACPC_args[out_var.format(t)] = os.path.join(output_dir,
                                                        "ACPC_" + eachfile)

    # Make output directory for transformed images
    xfm_vars["xfms_out_dir"] = os.path.join(output_dir, "xfms")
    os.makedirs(xfm_vars["xfms_out_dir"], exist_ok=True)

    # Include 2 steps: 1 to run this, and another to run ACPC alignment
    xfm_imgs_non_ACPC = registration_T1w_to_T2w(j_args, xfm_vars)
    xfm_imgs_ACPC_aligned = align_ACPC(j_args)
    for t in (1, 2):
        xfm_ACPC_args["t{}w_img".format(t)] = xfm_imgs_ACPC_aligned["T{}w".format(t)]
    xfm_ACPC_and_registered_imgs = registration_T1w_to_T2w(j_args, xfm_ACPC_args)

    return optimal_realigned_imgs(xfm_imgs_non_ACPC,
                                  xfm_ACPC_and_registered_imgs, logger)


def reshape_volume_to_array(array_img):
    image_data = array_img.get_fdata()
    return image_data.flatten()


def calculate_eta(t1_image_file, t2_image_file):
    """
    [summary] 
    :param t1_image_file: [type], [description]
    :param t2_image_file: [type], [description]
    :return: Int(?), eta value
    """
    t1_image = nib.load(t1_image_file)
    t2_image = nib.load(t2_image_file)
    t1_vector = reshape_volume_to_array(t1_image)
    t2_vector = reshape_volume_to_array(t2_image)

    # mean value over all locations in both images
    m_grand = (np.mean(t1_vector) + np.mean(t2_vector)) / 2

    # mean value matrix for each location in the 2 images
    m_within = (t1_vector + t2_vector) / 2

    sswithin = sum(np.square(t1_vector - m_within)) + sum(np.square(t2_vector - m_within))
    sstot = sum(np.square(t2_vector - m_grand)) + sum(np.square(t2_vector - m_grand))
    # N.B. SStot = SSwithin + SSbetween so eta can also be written as SSbetween/SStot
    return 1 - sswithin / sstot


def registration_T1w_to_T2w(j_args, xfm_vars):
    """
    T1w to T2w registration 
    :param j_args: [type], [description]
    :param xfm_vars: [type], [description]
    :return: [type], [description]
    """
    t2_to_t1_matrix = os.path.join(xfm_vars["xfms_out_dir"], "T2toT1.mat")    
    run_FSL_sh_script(j_args, "flirt", xfm_vars["t2w_img"], xfm_vars["t1w_img"],
                      "-omat", t2_to_t1_matrix)
    xfms_matrices = {"T1w": xfm_vars["ident_mx"], "T2w": t2_to_t1_matrix}
    transformed_images = dict()
    for img in xfms_matrices.keys():
        transformed_images[img] = os.path.join(
            xfm_vars["xfms_out_dir"], "{}_to_BIBS_template.mat".format(img)
        )
        run_FSL_sh_script(
            j_args, "flirt", xfm_vars["t1w_img"], xfm_vars["ref_img"],
            "-applyisoxfm", xfm_vars["resln"], "-init", xfms_matrices[img],
            "-o", xfm_vars["output_t1w_img"], "-omat", transformed_images[img]
        )
    
    return transformed_images


def optimal_realigned_imgs(xfm_imgs_non_ACPC, xfm_ACPC_and_registered_imgs,
                           logger):
    """
    Check whether the cost function shows that only the registration-T1-to-T2
    or the ACPC-alignment-and-T1-to-T2-registration is better (check whether
    ACPC alignment improves the T1-to-T2 registration; compare the T1-to-T2
    with and without first doing the ACPC registration)
    """
    msg = "Using {} T1w-to-T2w registration for resizing."
    if (calculate_eta(xfm_imgs_non_ACPC["T1w"], xfm_imgs_non_ACPC["T2w"]) >
        calculate_eta(xfm_ACPC_and_registered_imgs["T1w"],
                      xfm_ACPC_and_registered_imgs["T2w"])):
        optimal_resize = xfm_imgs_non_ACPC
        logger.info(msg.format("only"))
    else:
        optimal_resize = xfm_ACPC_and_registered_imgs
        logger.info(msg.format("ACPC and"))
    return optimal_resize


def align_ACPC(j_args):  # TODO Assign input arguments
    mni_path = "/home/feczk001/gconan/CABINET/data/MNI_templates/INFANT_MNI_T{}_1mm.nii.gz"
    output_T1 = "" # TODO
    output_T2 = "" # TODO
    # TODO Assign paths (replace any path starting with '$')
    run_FSL_sh_script(j_args, "flirt", "-interp", "spline", "-in",
                      "$WD/robustroi.nii.gz", "-ref", "$Reference", "-omat",
                      "$WD/roi2std.mat", "-out", "$WD/acpc_final.nii.gz",
                      "-searchrx", "-45", "45", "-searchry", "-30", "30",
                      "-searchrz", "-30", "30")
    run_FSL_sh_script(j_args, "convert_xfm", "-omat", "$WD/full2std.mat", # Combine ACPC-alignment with robustFOV output
                      "-concat", "$WD/roi2std.mat", "$WD/full2roi.mat")
    # TODO Does this need to be Python 2?
    # Can it be run in a different way that doesn't require adding a Python 2 script path to the parameter file?
    # Can we just make a copy of aff2rigid, convert it to Python3, put it in the repo, and import it?
    subprocess.check_call(("${PYTHON2}", os.path.join(j_args["common"]["fsl_bin_path"], "aff2rigid"),
                           "$WD/full2std.mat", "$OutputMatrix"))
    run_FSL_sh_script(j_args, "applywarp", "--rel", "--interp=spline", "-i",
                      "${Input}", "-r", "$Reference", "--premat=$OutputMatrix",
                      "-o", "${Output}")
    return {"T1w": output_T1, "T2w": output_T2}


def run_FSL_sh_script(j_args, fsl_function_name, *fsl_args):
    subprocess.check_call([os.path.join(j_args["common"]["fsl_bin_path"],
                                        fsl_function_name), *fsl_args])


def run_all_stages(all_stages, start, end, params_for_every_stage, logger):
    """
    Run stages sequentially, starting and ending at stages specified by user
    :param all_stages: List of functions in order where each runs one stage
    :param start: String naming the first stage the user wants to run
    :param end: String naming the last stage the user wants to run
    :param params_for_every_stage: Dictionary of all args needed by each stage
    """
    running = False
    for stage in all_stages:
        stage_name = get_stage_name(stage)
        if stage_name == start:
            running = True
        if running:
            stage_start = datetime.now()
            stage(params_for_every_stage, logger)
            get_and_print_time_since(stage_name + " started", stage_start)
        if stage_name == end:
            running = False


def create_anatomical_average(
        t1_image_file_paths=list(), t2_image_file_paths=list(),
        t1_avg_output_file_path=None, t2_avg_output_file_path=None
    ):
    """
    Creates a NIFTI file whose voxels are the average of the voxel values of the input files.
    :param t1_image_file_paths: List (possibly empty) of t1 image file paths
    :param t2_image_file_paths: List (possibly empty) of t2 image file paths
    :param t1_avg_output_file_path: t1 output file path
    :param t2_avg_output_file_path: t1 output file path
    """
    for i in range(2):
        if i == 0:
            input_file_paths = t1_image_file_paths
            output_file_path = t1_avg_output_file_path
        else:
            input_file_paths = t2_image_file_paths
            output_file_path = t2_avg_output_file_path
        if input_file_paths:
            register_and_average_files(input_file_paths, output_file_path)


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


def create_avg_image(output_file_path, registered_files):
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


def valid_subj_ses(in_arg, pfx, name):  # , *keywords):
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

    for section_name, section_dict in j_types.items():

        # Verify parameters in each section, only including the
        # resource_management parameters if we're in SLURM/SBATCH job(s)
        if ((section_name not in stage_names) and not (
                section_name == "resource_management" 
                and not j_args["meta"]["slurm"]

            # Skip the j_args sections for stages not being run
             )) or (section_name in stage_names and will_run_stage(
                section_name, j_args["stage_names"]["start"],
                j_args["stage_names"]["end"], stage_names
            )):

            for arg_name, arg_type in section_dict.items():
                validate_1_parameter(j_args, arg_name, arg_type, section_name,
                                     type_validators, param_json, parser)


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
    except (argparse.ArgumentTypeError, TypeError, ValueError) as e:
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