#!/usr/bin/env python3
# coding: utf-8

"""
Common source for utility functions used by CABINET :)
Greg Conan: gconan@umn.edu
Created: 2021-11-12
Updated: 2022-01-13
"""

# Import standard libraries
import argparse
import json
import os
import random  # only used by rand_string
import shutil
import string  # only used by rand_string
import subprocess
import sys
from datetime import datetime  # for seeing how long scripts take to run
from glob import glob
from os import listdir

import nibabel as nib

# Constants: Name of scanner-info command-line argument, directory containing
# the main pipeline script, SLURM-/SBATCH-related arguments' default names, and
# name of the argument to get the directory containing the main wrapper script
SCAN_ARG = 'scanners_info'
SCRIPT_DIR = os.path.dirname(os.path.dirname(__file__))
SLURM_ARGS = ('account', 'cpus', 'memory', 'print_progress', 'sleep', 'time')
WRAPPER_LOC = 'wrapper_location'


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
        '-A', '--account',
        help="Name of the account to submit the SBATCH job under."
    )
    parser.add_argument(
        '-c', '--cpus', type=valid_whole_number, default=default_cpus,
        help=('Number of CPUs to use for each Python job. By default, this '
              'argument\'s value will be {}.'.format(default_cpus))
    )
    parser.add_argument(
        '-mem', '--memory', type=valid_whole_number, default=default_gb_mem,
        help=("Memory in gigabytes (GB) to assign to each sbatch job. The "
              "default number is {} GB.".format(default_gb_mem))
    )
    parser.add_argument(
        '-progress', '--print-progress', action='store_true',
        help=('Include this flag for the script to print updates about its '
              'progress at intervals defined by --sleep. This will also print '
              'every command that is run to submit a pipeline batch job.')
    )
    parser.add_argument(
        '-sleep', '--sleep', type=valid_whole_number, default=default_sleep,
        help=("Number of seconds to wait between batch job submissions. The "
              "default number is {}.".format(default_sleep))
    )
    parser.add_argument(
        '-time', '--time', metavar="SLURM_JOB_TIME_LIMIT",
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


def copy_and_rename_file(old_file, new_file):
    """
    Rename a file and copy it to a new location
    :param old_file: String, valid path to an existing file to copy
    :param new_file: String, valid path to what will be a copy of old_file
    """
    os.rename(shutil.copy2(old_file, os.path.dirname(new_file)), new_file)


def crop_images(image_dir, output_dir, z_min=80, z_max=320):
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
        cropped_img = img.slicer[:208, :300, z_min:z_max, ...]
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


def exit_with_time_info(start_time, exit_code=0):
    """
    Terminate the pipeline after displaying a message showing how long it ran
    :param start_time: datetime.datetime object of when the script started
    :param exit_code: exit code
    """
    print('The pipeline for this subject took this long to run {}: {}'
          .format('successfully' if exit_code == 0 else 'and then crashed',
                  datetime.now() - start_time))
    sys.exit(exit_code)


def extract_from_json(json_path):
    """
    :param json_path: String, a valid path to a real readable .json file
    :return: Dictionary, the contents of the file at json_path
    """
    with open(json_path, 'r') as infile:
        return json.load(infile)


def get_and_print_time_since(event_name, event_time):
    """
    Print and return a string showing how much time has passed since the
    current running script reached a certain part of its process
    :param event_name: String to print after 'Time elapsed since '
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
                                      ).decode('utf-8').split()[-1]
    except subprocess.CalledProcessError:
        cmd = None
    return cmd


def get_main_pipeline_arg_names():
    """
    :return: Set containing strings naming all command-line arguments included
             by default in the main script, pipeline_wrapper.py
    """
    return {'bids_dir', 'censor', 'events_dir', 'fd', 'filter', 'fsl_dir',
            'keep_all', 'levels', 'no_parallel', 'output', 'runs', 'ses',
            'spat_smooth', 'subject', 'surf_smooth', 'study_dir', 'study_name',
            'task', 'temp_dir', 'templates', 'template1', 'template2',
            'vol_smooth', 'wb_command', WRAPPER_LOC}


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


def get_pipeline_cli_argparser(arg_names=None):
    """
    :param arg_names: Set containing strings naming all command-line arguments
    :return: argparse.ArgumentParser with all command-line arguments 
             needed to run pipeline_wrapper.py
    """
    # Default values for user input arguments
    if arg_names is None:
        arg_names = get_main_pipeline_arg_names()
    default_bids_dir = 'abcd-hcp-pipeline'
    generic_dtseries_path = os.path.join(
        '(--study-dir)', 'derivatives', '(--bids-dir)',
        '(--subject)', '(--ses)', 'func',
        'sub-(--subject)_ses-(--ses)_task-(--task)_'
        'run-(--runs)_bold_timeseries.dtseries.nii'
    )
    generic_output_dirpath = os.path.join('(--study-dir)', 'derivatives',
                                          'abcd-bids-tfmri-pipeline',
                                          '(--subject)', '(--ses)')

    # Strings used in multiple help messages
    msg_default = ' By default, this argument\'s value(s) will be {}.'
    msg_pipeline = 'Name of the {} that you are running the pipeline on.'

    # Create parser with command-line arguments from user
    parser = argparse.ArgumentParser(description=(
        'ABCD fMRI Task Prep pipeline. Inputs must be in the same format '
        'as ABCD-HCP-Pipeline outputs after running filemap.'
    ))
    parser = add_arg_if_in_arg_names('parameter_file', arg_names,
                                     parser, )  # TODO Make the script use a parameter file instead of just adding all
    # of the options from the nibabies, XCP, etc.?
    parser = add_arg_if_in_arg_names('bids_dir', arg_names, parser,
                                     metavar='NAME_OF_BIDS_DERIVATIVES_PIPELINE_DIRECTORY',
                                     default=default_bids_dir,
                                     help=('Name of the BIDS-standard file-mapped directory with subject '
                                           'data in the "derivatives" subdirectory of your --study-dir. '
                                           'This path should be valid: ' + generic_dtseries_path +
                                           msg_default.format(default_bids_dir))
                                     )
    parser = add_arg_if_in_arg_names('output', arg_names, parser,
                                     '-out', metavar='OUTPUT_DIRECTORY', type=valid_output_dir,  # required=True,
                                     help=('Directory path to save pipeline outputs into.'
                                           + msg_default.format(generic_output_dirpath))
                                     )
    # Which task you are running the pipeline on
    parser = add_arg_if_in_arg_names('task', arg_names, parser,
                                     metavar='TASK_NAME', required=True,
                                     help=msg_pipeline.format('task')  # + msg_choices(choices_tasks)
                                     )
    parser = add_arg_if_in_arg_names('temp_dir', arg_names, parser,
                                     type=valid_readable_dir, metavar='TEMPORARY_DIRECTORY',
                                     help='Valid path to existing directory to save temporary files into.'
                                     )
    # Argument used to get this script's dir
    parser = add_arg_if_in_arg_names(WRAPPER_LOC, arg_names, parser,
                                     type=valid_readable_dir, required=True,
                                     help=('Valid path to existing ABCD-BIDS-task-fmri-pipeline directory '
                                           'that contains pipeline_wrapper.py')
                                     )
    return parser


def get_sbatch_args(cli_args, job):
    """
    :param cli_args: Dictionary containing all command-line arguments from user
    :param job: String 1-8 characters long naming the SBATCH job
    :return: List of strings, SLURM-related arguments to pass to the main
             script or level 1 analysis script for parallelization
    """
    return [argify('time', cli_args['time']), '-c', str(cli_args['cpus']),
            '-J', job, argify('mem', '{}gb'.format(cli_args["memory"]))]


def get_sub_base(cli_args, run_num=None):
    """
    :param cli_args: Dictionary containing all command-line arguments from user
    :param run_num: Whole number as an int or string defining which run this is
    :return: String identifying a subject, session, task, and maybe run
    """
    parts = [get_subj_ses(cli_args), 'task-' + cli_args['task']]
    if run_num is not None:
        parts.append('run-{}'.format(run_num))
    return '_'.join(parts)


def get_subj_ses(cli_args):
    """
    :param cli_args: Dictionary containing all command-line arguments from user
    :return: String which combines --subject and --ses from command line
    """
    return '_'.join((cli_args['subject'], cli_args['ses']))


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
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


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
    only_files = [f for f in listdir(input_folder)  # TODO Add comment explaining what this is
                  if os.path.isfile(os.path.join(input_folder, f))]

    os.system('module load fsl')  # TODO This will only work on MSI, so change it (import it somehow?)
    resolution = "1"
    count = 1
    for eachfile in only_files:
        input_image = os.path.join(input_folder, eachfile)
        output_image = os.path.join(output_folder, eachfile)
        print(eachfile)  # TODO remove this line?

        if count == 1:
            reference_image = os.path.join(input_folder, only_files[1])  # TODO Add explanatory comments!
            t1_to_t2_matrix = os.path.join(output_folder, 'T1toT2.mat')
            run_flirt_resize('flirt', input_image, reference_image,  # TODO Add flirt path as input parameter
                             '-omat', t1_to_t2_matrix)
            init_matrix = t1_to_t2_matrix
        elif count == 2:
            init_matrix = ident_mx
        run_flirt_resize('flirt', input_image, reference_image_path,
                         '-applyisoxfm', resolution, '-init', init_matrix,
                         '-o', output_image)
        count += 1 


def run_all_stages(all_stages, start, end, params_for_every_stage):
    """
    Run stages sequentially, starting and ending at stages specified by user
    :param all_stages: Dictionary mapping a string naming a stage to the
                       function to run that stage
    :param start: String naming the first stage the user wants to run
    :param end: String naming the last stage the user wants to run
    :param params_for_every_stage: Dictionary of all args needed by each stage
    """
    running = False
    for stage_name in all_stages.keys():
        if stage_name == start:
            running = True
        if running:
            stage_start = datetime.now()
            # globals()["run_" + stage](params_for_every_stage)
            all_stages[stage_name](params_for_every_stage)
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
    subprocess.check_call([flirt_path, '-interp', 'spline', '-in',
                           in_path, '-ref', ref_path, *args])


def valid_float_0_to_1(val):
    """
    :param val: Object to check, then throw an error if it is invalid
    :return: val if it is a float between 0 and 1 (otherwise invalid)
    """
    return validate(val, lambda x: 0 <= float(x) <= 1, float,
                    'Value must be a number between 0 and 1')


def valid_output_dir(path):
    """
    Try to make a folder for new files at path; throw exception if that fails
    :param path: String which is a valid (not necessarily real) folder path
    :return: String which is a validated absolute path to real writeable folder
    """
    return validate(path, lambda x: os.access(x, os.W_OK),
                    valid_readable_dir, 'Cannot create directory at {}',
                    lambda y: os.makedirs(y, exist_ok=True))


def valid_readable_dir(path):
    """
    :param path: Parameter to check if it represents a valid directory path
    :return: String representing a valid directory path
    """
    return validate(path, os.path.isdir, valid_readable_file,
                    'Cannot read directory at {}')


def valid_readable_file(path):
    """
    Throw exception unless parameter is a valid readable filepath string. Use
    this, not argparse.FileType('r') which leaves an open file handle.
    :param path: Parameter to check if it represents a valid filepath
    :return: String representing a valid filepath
    """
    return validate(path, lambda x: os.access(x, os.R_OK),
                    os.path.abspath, 'Cannot read file at {}')


def valid_readable_json(path):
    """
    :param path: Parameter to check if it represents a valid .json file path
    :return: String representing a valid .json file path
    """
    return validate(path, lambda _: os.path.splitext(path)[-1] == '.json',
                    valid_readable_file, '{} is not a readable .json filepath')


def valid_subj_ses(in_arg, prefix, name):  # , *keywords):
    """
    :param in_arg: Object to check if it is a valid subject ID or session name
    :param prefix: String, 'sub-' or 'ses-'
    :param name: String describing what in_arg should be (e.g. 'subject')
    :return: True if in_arg is a valid subject ID or session name; else False
    """
    return validate(in_arg, lambda _: True,  # lambda x: any([key in x for key in [prefix, *keywords]]),
                    lambda y: (y if y[:len(prefix)] == prefix else prefix + y),
                    '{}' + ' is not a valid {}'.format(name))


def valid_template_filename(fname):
    """
    :param fname: Parameter to check if it represents a .fsf file name
    :return: String representing the .fsf file name
    """
    return validate(fname, lambda x: os.path.splitext(x)[-1] == '.fsf',
                    lambda y: y, '{} is not an .fsf file name')


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
        raise argparse.ArgumentTypeError('Invalid time string.')


def valid_whole_number(to_validate):
    """
    Throw argparse exception unless to_validate is a positive integer
    :param to_validate: Object to test whether it is a positive integer
    :return: to_validate if it is a positive integer
    """
    return validate(to_validate, lambda x: int(x) >= 0, int,
                    '{} is not a positive integer')


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


def validate_cli_args(cli_args, arg_names=None):
    """
    Validate types and set defaults for any arg whose validation depends on
    another arg and therefore was not possible in get_pipeline_cli_argparser
    :param cli_args: Dictionary containing all command-line arguments from user
    :param arg_names: Set containing SCAN_ARG if that argument is needed
    :return: cli_args, but fully validated
    """
    # Default output directory. Avoiding ensure_dict_has to
    if arg_names is None:
        set()
    if not dict_has(cli_args, 'output'):  # prevent permissions error from
        cli_args['output'] = valid_output_dir(  # valid_output_dir making dirs.
            os.path.join(cli_args['study_dir'], 'derivatives', 'abcd-bids-tfm'
                                                               'ri-pipeline', cli_args['subject'], cli_args['ses'])
        )
    return cli_args
