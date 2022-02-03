#!/usr/bin/env python3
# coding: utf-8

"""
Common source for utility functions used by CABINET :)
Greg Conan: gconan@umn.edu
Created: 2021-11-12
Updated: 2022-02-03
"""

# Import standard libraries
import argparse
import json
import logging
import nibabel as nib
import os
import random  # only used by rand_string
import shutil
# from src.util.look_up_tables import get_id_to_region_mapping
import string  # only used by rand_string
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


def add_arg_if_in_arg_names(arg_name, all_args, parser, *shortnames, **kwargs):
    """
    Wrapper for argparse.ArgumentParser.add_argument. Nearly identical, but 
    will only add the argument to the parser if arg_name is in all_args.
    :param arg_name: String naming the argument to (maybe) add to parser
    :param all_args: Set of strings; each names a command-line argument
    :param parser: argparse.ArgumentParser
    :param shortnames: Unpacked list of strings; each is arg_name shortened
    :param kwargs: Unpacked dictionary of argparse attributes to give the arg
    :return: parser, but (maybe) with the argument named arg_name added
    """
    if arg_name in all_args:
        cli_arg = as_cli_arg(arg_name)
        parser.add_argument(
            cli_arg[1:], cli_arg, *shortnames, **kwargs
        )
    return parser


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
                      transformed_T1w_out, t1w_path):
    """
    Creates an output file with chirality corrections fixed.
    nifti_input_file_path: String, path to a segmentation file with possible chirality problems
    segment_lookup_table: String, path to a FreeSurfer-style look-up table
    left_right_mask_nifti_file: String, path to a mask file that distinguishes between left and right
    nifti_output_file_path: String, path to location to write the corrected file
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

    output_copy = nifti_output_file_path + "_dummy"
    copy_and_rename_file(nifti_output_file_path, output_copy)
    transformed_output = "resize_to_T1w.mat"  # TODO
    subprocess.check_call(("convert_xfm", "-omat", transformed_output, "-inverse", transformed_T1w_out))  # Invert transformed_T1w_out  # TODO get path to FSL tool convert_xfm
    run_flirt_resize("flirt", output_copy, t1w_path,
                     "-applyxfm", "-init", transformed_T1w_out,  # TODO -applyxfm might need to be changed to -applyisoxfm with resolution
                     "-o", nifti_output_file_path)


def crop_images(image_dir, output_dir, z_min=80, z_max=320):  # TODO Save out these hardcoded parameters
    """
    Resize Images.
    Usage:
    crop_images <input_folder> <output_folder>
    crop_images -h | --help
    Options:
    -h --help     Show this screen.
    """
    image_files = sorted([f for f in os.listdir(image_dir)
                          if os.path.isfile(os.path.join(image_dir, f))])
    for eachfile in image_files:
        # fslroi sub-CENSORED_ses-20210412_T1w sub-CENSORED_ses-20210412_T1w_cropped 0 144 0 300 103 320
        input_file = os.path.join(image_dir, eachfile)
        img = nib.load(input_file)
        cropped_img = img.slicer[:208, :300, z_min:z_max, ...]  # TODO Save out these hardcoded parameters
        print(cropped_img.shape)
        output_file = os.path.join(output_dir, eachfile)
        nib.save(cropped_img, output_file)


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


def rand_string(length):
    """
    :param length: Integer, length of the string to randomly generate
    :return: String (of the given length L) of random characters
    """
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def resize_images(input_folder, output_folder, reference_image_path, ident_mx):
    """
    Resize the images to match the dimensions of images trained in the model,
    and ensure that the first image (presumably a T1) is co-registered to the
    second image (presumably a T2) before resizing
    :param input_folder: String, valid path to existing directory containing
                         image files to resize
    :param output_folder: String, valid path to existing directory to save
                          resized image files into
    :param reference_image_path: String, valid path to existing image file for
                                 flirt to use as a reference image
    :param ident_mx: String, valid path to existing identity matrix .MAT file
    """
    only_files = [f for f in os.listdir(input_folder)  # TODO Add comment explaining what this is
                  if os.path.isfile(os.path.join(input_folder, f))]

    os.system("module load fsl")  # TODO This will only work on MSI, so change it (import it somehow?)
    resolution = "1"
    # count = 1

    for eachfile in only_files:
        scan_type = eachfile.rsplit("_", 1)[-1].split(".", 1)[0]
        print(eachfile)  # TODO remove this line?

        if scan_type == "0000": # count == 1:
            t1w_image = os.path.join(input_folder, eachfile)  # TODO Add explanatory comments!
            output_t1w_img = os.path.join(output_folder, eachfile)
        elif scan_type == "0001": # count == 2:
            t2w_image = os.path.join(input_folder, eachfile)
            output_t2w_img = os.path.join(output_folder, eachfile)

    # Make output directory for transformed images
    xfms_out_dir = os.path.join(output_folder, "xfms")
    os.makedirs(xfms_out_dir, exist_ok=True)
    t2_to_t1_matrix = os.path.join(xfms_out_dir, "T2toT1.mat")    

    run_flirt_resize("flirt", t2w_image, t1w_image,  # TODO Add flirt path as input parameter
                     "-omat", t2_to_t1_matrix)

    xfms_matrices = {"T1w": ident_mx, "T2w": t2_to_t1_matrix}
    transformed_images = dict()
    for img in xfms_matrices.keys():
        transformed_images[img] = os.path.join(
            xfms_out_dir, "{}_to_BIBS_template.mat".format(img)
        )
        run_flirt_resize("flirt", t1w_image, reference_image_path,
                        "-applyisoxfm", resolution, "-init", xfms_matrices[img],
                        "-o", output_t1w_img, "-omat", transformed_images[img])
    return transformed_images


def revert_aseg_to_native_space(input_folder, output_folder, ref_image_path,
                                ident_mx):
    """
    Resize the images to match the dimensions of images trained in the model,
    and ensure that the first image (presumably a T1) is co-registered to the
    second image (presumably a T2) before resizing
    :param input_folder: String, valid path to existing directory containing
                         image files to resize (chirality folder)
    :param output_folder: String, valid path to existing directory to save
                          resized image files into
    :param ref_image_path: String, valid path to existing image file for
                           flirt to use as a reference image
    :param ident_mx: String, valid path to existing identity matrix .MAT file
    """
    only_files = [f for f in os.listdir(input_folder)  # TODO Add comment explaining what this is
                  if os.path.isfile(os.path.join(input_folder, f))]

    os.system("module load fsl")  # TODO This will only work on MSI, so change it (import it somehow?)
    resolution = "1"
    # count = 1

    for eachfile in only_files:
        scan_type = eachfile.rsplit("_", 1)[-1].split(".", 1)[0]
        print(eachfile)  # TODO remove this line?

        if scan_type == "T1w": # count == 1:
            t1w_image = os.path.join(input_folder, eachfile)  # TODO Add explanatory comments!
            output_t1w_img = os.path.join(output_folder, eachfile)
        elif scan_type == "T2w": # count == 2:
            t2w_image = os.path.join(input_folder, eachfile)
            output_t2w_img = os.path.join(output_folder, eachfile)

    # Make output directory for transformed images
    xfms_out_dir = os.path.join(output_folder, "xfms")
    os.makedirs(xfms_out_dir, exist_ok=True)
    t2_to_t1_matrix = os.path.join(xfms_out_dir, "T2toT1.mat")    

    run_flirt_resize("flirt", t2w_image, t1w_image,  # TODO Add flirt path as input parameter
                     "-omat", t2_to_t1_matrix)
    transformed_T1w_out = os.path.join(xfms_out_dir, "T1w_to_BIBS_template.mat")
    transformed_T2w_out = os.path.join(xfms_out_dir, "T2w_to_BIBS_template.mat")
    run_flirt_resize("flirt", t1w_image, ref_image_path,
                     "-applyisoxfm", resolution, "-init", ident_mx,
                     "-o", output_t1w_img, "-omat", transformed_T1w_out)
    run_flirt_resize("flirt", t2w_image, ref_image_path,
                     "-applyisoxfm", resolution, "-init", t2_to_t1_matrix,
                     "-o", output_t2w_img, "-omat", transformed_T2w_out)


def run_all_stages(all_stages, start, end, params_for_every_stage):
    """
    Run stages sequentially, starting and ending at stages specified by user
    :param all_stages: List of functions in order where each runs one stage
    :param start: String naming the first stage the user wants to run
    :param end: String naming the last stage the user wants to run
    :param params_for_every_stage: Dictionary of all args needed by each stage
    """
    logger = logging.getLogger(os.path.basename(sys.argv[0]))
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


def run_flirt_resize(flirt_path, in_path, ref_path, *args):
    """
    Run flirt command to resize an image
    :param flirt_path: String, valid path to existing flirt executable file
    :param ref_path: String, valid path to existing reference image .nii.gz
    :param in_path: String, valid path to existing input image file
    :param args: List of strings; each is an additional flirt parameter
    """
    subprocess.check_call([flirt_path, "-interp", "spline", "-in",
                           in_path, "-ref", ref_path, *args])
                    

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


def valid_subj_ses(in_arg, prefix, name):  # , *keywords):
    """
    :param in_arg: Object to check if it is a valid subject ID or session name
    :param prefix: String, "sub-" or "ses-"
    :param name: String describing what in_arg should be (e.g. "subject")
    :return: True if in_arg is a valid subject ID or session name; else False
    """
    return validate(in_arg, lambda _: True,
                    lambda y: (y if y[:len(prefix)] == prefix else prefix + y),
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
                       "new_file_path": lambda _: True,  # TODO Make "valid_output_filename" function?
                       "optional_new_dirpath": valid_output_dir_or_false,
                       "positive_float": valid_positive_float,
                       "positive_int": valid_whole_number, 
                       "str": lambda _: True}

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
    :param arg_type: String naming the data type of the parameter to validate
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