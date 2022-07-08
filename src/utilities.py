#!/usr/bin/env python3
# coding: utf-8

"""
Common source for utility functions used by CABINET :)
Greg Conan: gconan@umn.edu
Created: 2021-11-12
Updated: 2022-07-08
"""
# Import standard libraries
import argparse
import json
import nibabel as nib
from nipype.interfaces import fsl
import numpy as np
import os
import pdb
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


# NOTE All functions below are in alphabetical order.


def align_ACPC_1_img(j_args, logger, xfm_ACPC_vars, crop2full, output_var, t,
                     averaged_image):
    """ 
    Functionality copied from the DCAN Infant Pipeline:
    github.com/DCAN-Labs/dcan-infant-pipeline/blob/master/PreFreeSurfer/scripts/ACPCAlignment_with_crop.sh
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :param xfm_ACPC_vars: Dictionary mapping strings (ACPC input arguments'
                          names) to strings (ACPC arguments, file/dir paths)
    :param crop2full: String, valid path to existing crop2full.mat file
    :param output_var: String (with {} in it), a key in xfm_ACPC_vars mapped to
                       the T1w and T2w valid output image file path strings 
    :param t: Int, either 1 or 2 (to signify T1w or T2w respectively)
    :return: Dictionary mapping .mat file names (without extensions) to their
             respective paths
    """
    # Get paths to ACPC ref image, output dir, output images, and .mat files
    mni_ref_img_path = xfm_ACPC_vars["ref_img"].format(t)
    work_dir = xfm_ACPC_vars["out_dir"]  # Working directory for ACPC alignment
    input_img = xfm_ACPC_vars["crop_T{}w_img".format(t)]  # Cropped img, ACPC input
    output_img =  xfm_ACPC_vars[output_var.format(t)]  # ACPC-aligned image
    mats = {fname: os.path.join(work_dir, "T{}w_{}.mat".format(t, fname))
            for fname in ("crop2acpc", "full2acpc", "full2crop",
                          "acpc2rigidbody")}  # .mat file paths

    # acpc_final_img = os.path.join(work_dir, "T{}w_acpc_final.nii.gz".format(t))
    
    run_FSL_sh_script(j_args, logger, "flirt", "-interp", "spline",  
                      "-ref", mni_ref_img_path, "-in", input_img,
                      "-omat", mats["crop2acpc"], # "-out", acpc_final_img,
                      "-searchrx", "-45", "45", "-searchry", "-30", "30",
                      "-searchrz", "-30", "30")

    # Invert crop2full to get full2crop
    run_FSL_sh_script(j_args, logger, "convert_xfm", "-inverse", crop2full,
                      "-omat", mats["full2crop"])

    run_FSL_sh_script(  # Combine ACPC-alignment with robustFOV output
        j_args, logger, "convert_xfm", "-omat", mats["full2acpc"],
        "-concat", mats["crop2acpc"], mats["full2crop"]
    )

    # Transform 12 dof matrix to 6 dof approximation matrix
    run_FSL_sh_script(j_args, logger, "aff2rigid", mats["full2acpc"],
                      mats["acpc2rigidbody"])

    # run_FSL_sh_script(j_args, logger, "convert_xfm", "-inverse", mats["rigidbody2acpc"], "-omat", mats["acpc2rigidbody"])

    # Apply ACPC alignment to the data
    # Create a resampled image (ACPC aligned) using spline interpolation  # TODO Only run this command in debug mode
    # if j_args["common"]["debug"]:
    run_FSL_sh_script(j_args, logger, "applywarp", "--rel", "--interp=spline",  
                      "-i", averaged_image, "-r", mni_ref_img_path,  # Changed input_img to average_image 2022-06-16
                      "--premat=" + mats["acpc2rigidbody"], "-o", output_img)
    # pdb.set_trace()  # TODO Add "debug" flag?
    return mats


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
    sub_ses = get_subj_ID_and_session(j_args)
    msg = "{} chirality correction on {}"
    nifti_corrected_file_path = os.path.join(
        chiral_out_dir, "corrected_" + os.path.basename(nifti_input_file_path)
    )# j_args["BIBSnet"]["aseg_outfile"]) 
    nifti_output_file_path = os.path.join(
        chiral_out_dir, "native_" + os.path.basename(nifti_input_file_path)
    )# j_args["BIBSnet"]["aseg_outfile"]) 

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
    nib.save(fixed_img, nifti_corrected_file_path)

    # TODO Make everything below its own function called "reverse_registration" or "revert_to_native" or something

    # Undo resizing right here (do inverse transform) using RobustFOV so padding isn't necessary; revert aseg to native space
    dummy_copy = "_dummy".join(split_2_exts(nifti_corrected_file_path))
    shutil.copy2(nifti_corrected_file_path, dummy_copy)

    seg_to_T1w_nat = os.path.join(chiral_out_dir, "seg_reg_to_T1w_native.mat")
    preBIBSnet_mat = os.path.join(j_args["optional_out_dirs"]["postBIBSnet"],
                                  *sub_ses, "preBIBSnet_crop_T1w_to_BIBS_template.mat") # "preBIBSnet_T1w_final.mat")   crop_T{}w_to_BIBS_template.mat
    run_FSL_sh_script(j_args, logger, "convert_xfm", "-omat",
                      seg_to_T1w_nat, "-inverse", preBIBSnet_mat)  # TODO Define preBIBSnet_mat path outside of stages because it's used by preBIBSnet and postBIBSnet

    run_FSL_sh_script(j_args, logger, "flirt", "-applyxfm", "-ref", t1w_path,
                      "-in", dummy_copy, "-init", seg_to_T1w_nat,
                      "-o", nifti_output_file_path, "-interp", "nearestneighbour")
    logger.info(msg.format("Finished", nifti_input_file_path))
    return nifti_output_file_path


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
    :return: String, path to crop2full.mat file in same dir as output_crop_img
    """
    output_crop_dir = os.path.dirname(output_crop_img)
    crop2full = os.path.join(output_crop_dir, "crop2full.mat")  # TODO Define this path outside of stages because it's used by preBIBSnet and postBIBSnet
    run_FSL_sh_script(j_args, logger, "robustfov", "-i", input_avg_img, 
                      "-m", crop2full, "-r", output_crop_img,
                      "-b", j_args["preBIBSnet"]["brain_z_size"])  # TODO Use head radius for -b
    return crop2full


def dict_has(a_dict, a_key):
    """
    :param a_dict: Dictionary (any)
    :param a_key: Object (any)
    :return: True if and only if a_key is mapped to something truthy in a_dict
    """
    return a_key in a_dict and a_dict[a_key]


def dilate_LR_mask(sub_LRmask_dir, anatfile):
    """
    Taken from https://github.com/DCAN-Labs/SynthSeg/blob/master/SynthSeg/dcan/img_processing/chirality_correction/dilate_LRmask.py
    :param sub_LRmask_dir: String, valid path to placeholder
    :param anatfile: String, valid path to placeholder
    """
    os.chdir(sub_LRmask_dir)
    if not os.path.exists('lrmask_dil_wd'):
        os.mkdir('lrmask_dil_wd')

    # anatfile = 'LRmask.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-thr 1 -uthr 1',
                           out_file='lrmask_dil_wd/Lmask.nii.gz')
    maths.run()

    maths = fsl.ImageMaths(in_file=anatfile, op_string='-thr 2 -uthr 2',
                           out_file='lrmask_dil_wd/Rmask.nii.gz')
    maths.run()

    maths.run()
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-thr 3 -uthr 3',
                           out_file='lrmask_dil_wd/Mmask.nii.gz')
    maths.run()

    # dilate, fill, and erode each mask in order to get rid of holes
    # (also binarize L and M images in order to perform binary operations)
    anatfile = 'lrmask_dil_wd/Lmask.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-dilM -dilM -dilM -fillh -ero',
                           out_file='lrmask_dil_wd/L_mask_holes_filled.nii.gz')
    maths.run()

    anatfile = 'lrmask_dil_wd/Rmask.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-bin -dilM -dilM -dilM -fillh -ero',
                           out_file='lrmask_dil_wd/R_mask_holes_filled.nii.gz')
    maths.run()

    anatfile = 'lrmask_dil_wd/Mmask.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-bin -dilM -dilM -dilM -fillh -ero',
                           out_file='lrmask_dil_wd/M_mask_holes_filled.nii.gz')
    maths.run()

    # Reassign values of 2 and 3 to R and M masks (L mask already a value of 1)
    anatfile = 'lrmask_dil_wd/R_mask_holes_filled.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-mul 2',
                           out_file='lrmask_dil_wd/R_mask_holes_filled_label2.nii.gz')
    maths.run()

    anatfile = 'lrmask_dil_wd/M_mask_holes_filled.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-mul 3',
                           out_file='lrmask_dil_wd/M_mask_holes_filled_label3.nii.gz')
    maths.run()

    # recombine new L, R, and M mask files
    anatfile_left = 'lrmask_dil_wd/L_mask_holes_filled.nii.gz'
    anatfile_right = 'lrmask_dil_wd/R_mask_holes_filled_label2.nii.gz'
    anatfile_mid = 'lrmask_dil_wd/M_mask_holes_filled_label3.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile_left, op_string='-add {}'.format(anatfile_right),
                           out_file='lrmask_dil_wd/recombined_mask_LR.nii.gz')
    maths.run()

    maths = fsl.ImageMaths(in_file=anatfile_mid, op_string='-add lrmask_dil_wd/recombined_mask_LR.nii.gz',
                           out_file='lrmask_dil_wd/dilated_LRmask.nii.gz')
    maths.run()

    ## Fix incorrect values resulting from recombining dilated components
    orig_LRmask_img = nib.load('LRmask.nii.gz')
    orig_LRmask_data = orig_LRmask_img.get_fdata()

    fill_LRmask_img = nib.load('lrmask_dil_wd/dilated_LRmask.nii.gz')
    fill_LRmask_data = fill_LRmask_img.get_fdata()

    # Flatten numpy arrays
    orig_LRmask_data_2D = orig_LRmask_data.reshape((182, 39676), order='C')
    orig_LRmask_data_1D = orig_LRmask_data_2D.reshape(7221032, order='C')

    fill_LRmask_data_2D = fill_LRmask_data.reshape((182, 39676), order='C')
    fill_LRmask_data_1D = fill_LRmask_data_2D.reshape(7221032, order='C')

    # grab index values of voxels with a value greater than 2.0 in filled L/R mask
    voxel_check = np.where(fill_LRmask_data_1D > 2.0)

    # Replace possible overlapping label values with corresponding label values from initial mask
    for i in voxel_check[:]:
        fill_LRmask_data_1D[i] = orig_LRmask_data_1D[i]

    # reshape numpy array
    fill_LRmask_data_2D = fill_LRmask_data_1D.reshape((182, 39676), order='C')
    fill_LRmask_data_3D = fill_LRmask_data_2D.reshape((182, 218, 182), order='C')

    # save new numpy array as image
    empty_header = nib.Nifti1Header()
    out_img = nib.Nifti1Image(fill_LRmask_data_3D, orig_LRmask_img.affine, empty_header)
    out_fpath = os.path.join(sub_LRmask_dir, 'LRmask_dil.nii.gz')
    nib.save(out_img, out_fpath)

    #remove working directory with intermediate outputs
    #shutil.rmtree('lrmask_dil_wd')

    return out_fpath


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


def get_optional_args_in(a_dict):
    """
    :param a_dict: Dictionary with validated parameters,
                   all of which are used by this function
    :return: List of most a_dict optional arguments and their values
    """
    optional_args = list()
    for arg in a_dict.keys():
        if a_dict[arg]:
            optional_args.append(as_cli_arg(arg))
            if isinstance(a_dict[arg], list):
                for el in a_dict[arg]:
                    optional_args.append(str(el))
            elif not isinstance(a_dict[arg], bool):
                optional_args.append(str(a_dict[arg]))
    return optional_args


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


def get_template_age_closest_to(age, templates_dir):
    template_ages = list()
    template_ranges = dict()

    # Get list of all int ages (in months) that have template files
    for tmpl_path in glob(os.path.join(templates_dir,
                                        "*mo_template_LRmask.nii.gz")):
        tmpl_age = os.path.basename(tmpl_path).split("mo", 1)[0]
        if "-" in tmpl_age: # len(tmpl_age) <3:
            for each_age in tmpl_age.split("-"):
                template_ages.append(int(each_age))
                template_ranges[template_ages[-1]] = tmpl_age
            # template_ages.append(int(tmpl_age.split("-")))
        else:
            template_ages.append(int(tmpl_age))
    
    # Get template age closest to subject age, then return template age
    closest_age = template_ages[np.argmin(np.abs(np.array(template_ages)-age))]
    return (template_ranges[closest_age] if closest_age
            in template_ranges else str(closest_age)) #final_template_age
    # template_ages = [os.path.basename(f).split("mo", 1)[0] for f in glob(globber)]


def glob_and_copy(dest_dirpath, *path_parts_to_glob):
    """
    Collect all files matching a glob string, then copy those files
    :param dest_dirpath: String, a valid path of a directory to copy files into
    :param path_parts_to_glob: Unpacked list of strings which join to form a
                               glob string of a path to copy files from
    """
    for file_src in glob(os.path.join(*path_parts_to_glob)):
        shutil.copy(file_src, dest_dirpath)


def log_stage_finished(stage_name, event_time, logger):
    """
    Print and return a string showing how much time has passed since the
    current running script reached a certain part of its process
    :param stage_name: String, name of event that just finished
    :param stage_start: datetime object representing when {stage_name} started
    :return: String with an easily human-readable message showing how much time
             has passed since {stage_start} when {stage_name} started.
    """
    logger.info("{0} finished. Time elapsed since {0} started: {1}"
                .format(stage_name, datetime.now() - event_time))


def make_given_or_default_dir(dirs_dict, dirname_key, default_dirpath):
    """
    :param dirs_dict: Dictionary which must map dirname_key to a valid path
    :param dirname_key: String which dirs_dict must map to a valid path
    :param default_dirpath: String, valid directory path to map dirname_key to
                            unless dirname_key's already mapped to another path
    :return: dirs_dict, but with dirname_key mapped to a valid directory path
    """
    dirs_dict = ensure_dict_has(dirs_dict, dirname_key, default_dirpath)
    os.makedirs(dirs_dict[dirname_key], exist_ok=True)
    return dirs_dict


def optimal_realigned_imgs(xfm_imgs_non_ACPC, xfm_imgs_ACPC_and_reg, j_args, logger):
    """
    Check whether the cost function shows that only the registration-T2-to-T1
    or the ACPC-alignment-and-T2-to-T1-registration is better (check whether
    ACPC alignment improves the T2-to-T1 registration; compare the T2-to-T1
    with and without first doing the ACPC registration)
    :param j_args:
    :param logger: logging.Logger object to raise warning
    """
    sub_ses = get_subj_ID_and_session(j_args)
    msg = "Using {} T2w-to-T1w registration for resizing.\nT1w: {}\nT2w: {}"
    if calculate_eta(xfm_imgs_non_ACPC) > calculate_eta(xfm_imgs_ACPC_and_reg):
        optimal_resize = xfm_imgs_non_ACPC
        logger.info(msg.format("only", optimal_resize["T1w"],
                               optimal_resize["T2w"]))  # TODO Verify that these print the absolute path
    else:
        optimal_resize = xfm_imgs_ACPC_and_reg
        logger.info(msg.format("ACPC and", optimal_resize["T1w"],
                               optimal_resize["T2w"]))  # TODO Verify that these print the absolute path

    # Create symlinks with the same name regardless of which is chosen, so 
    # postBIBSnet can use the correct/chosen .mat file
    concat_mat = optimal_resize["T1w_crop2BIBS_mat"]
    # TODO Rename T2w_crop2BIBS.mat to T2w_crop_to_T1w_to_BIBS.mat or something
    out_mat_fpath = os.path.join(  # TODO Pass this in (or out) from the beginning so we don't have to build the path twice (once here and once in postBIBSnet)
        j_args["optional_out_dirs"]["postBIBSnet"],
        *sub_ses, "preBIBSnet_" + os.path.basename(concat_mat)
    )
    """
    print("\nNow linking {0} to {1}\n{0} does {2}exist\n{1} does {3}exist\n".format(concat_mat, out_mat_fpath, "" if os.path.exists(concat_mat) else "not ", "" if os.path.exists(out_mat_fpath) else "not "))
    """
    if not os.path.exists(out_mat_fpath):
        os.symlink(concat_mat, out_mat_fpath)
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
    

def registration_T2w_to_T1w(j_args, logger, xfm_vars, reg_input_var, acpc):
    """
    T2w to T1w registration for use in preBIBSnet
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :param xfm_vars: Dictionary containing paths to files used in registration
    :param reg_input_var: String naming the key in xfm_vars mapped to the path
                          to the image to use as an input for registration
    :return: Dictionary mapping "T1w" and "T2w" to their respective newly
             registered image file paths
    """
    # String naming the key in xfm_vars mapped to the path
    # to the image to use as an input for registration
    logger.info("Input images for T1w registration:\nT1w: {}\nT2w: {}"
                .format(xfm_vars[reg_input_var.format(1)],
                        xfm_vars[reg_input_var.format(2)]))

    # Define paths to registration output matrices and images
    registration_outputs = {"cropT1tocropT1": xfm_vars["ident_mx"],
                            "cropT2tocropT1": os.path.join(xfm_vars["out_dir"], "cropT2tocropT1.mat")}

    """
    ACPC Order:
    1. T1w Save cropped and aligned T1w image 
    2. T2w Make T2w-to-T1w matrix

    NonACPC Order:
    1. T1w Make transformed
    2. T2w Make T2w-to-T1w matrix
    3. T2w Make transformed
    """
    nonACPC_xfm_params_T = dict()
    for t in (1, 2):
        # Define paths to registration output files
        registration_outputs["T{}w_crop2BIBS_mat".format(t)] = os.path.join(
            xfm_vars["out_dir"], "crop_T{}w_to_BIBS_template.mat".format(t)
        )
        registration_outputs["T{}w".format(t)] = xfm_vars["output_T{}w_img".format(t)]
        registration_outputs["T{}w_to_BIBS".format(t)] = os.path.join(
            xfm_vars["out_dir"], "T{}w_to_BIBS.nii.gz".format(t)
        )

        if t == 2:  # Make T2w-to-T1w matrix
            run_FSL_sh_script(j_args, logger, "flirt",
                            "-ref", xfm_vars[reg_input_var.format(1)],
                            "-in", xfm_vars[reg_input_var.format(2)],
                            "-omat", registration_outputs["cropT2tocropT1"],
                            "-out", registration_outputs["T2w"],
                            '-cost', 'mutualinfo',
                            '-searchrx', '-15', '15', '-searchry', '-15', '15',
                            '-searchrz', '-15', '15', '-dof', '6')  # Added changes suggested by Luci on 2022-03-30

        elif acpc:  # Save cropped and aligned T1w image 
            shutil.copy2(xfm_vars[reg_input_var.format(1)],
                         registration_outputs["T1w"])

        # Make transformed T1ws and T2ws
        if not acpc:  # TODO Should this go in its own function?
            run_FSL_sh_script(  # TODO Should the output image even be created here, or during applywarp?
                j_args, logger, "flirt",
                "-in", xfm_vars[reg_input_var.format(t)] if t == 1 else registration_outputs["T2w"],  # Input: Cropped image
                "-ref", xfm_vars["ref_img"].format(t),
                "-applyisoxfm", xfm_vars["resolution"],
                "-init", xfm_vars["ident_mx"], # registration_outputs["cropT{}tocropT1".format(t)],
                "-o", registration_outputs["T{}w_to_BIBS".format(t)], # registration_outputs["T{}w".format(t)],  # TODO Should we eventually exclude the (unneeded?) -o flags?
                "-omat", registration_outputs["T{}w_crop2BIBS_mat".format(t)]
            )
    # pdb.set_trace()  # TODO Add "debug" flag?
    return registration_outputs


def reshape_volume_to_array(array_img):
    """ 
    :param array_img: nibabel.Nifti1Image (or Nifti2Image?)
    :return: numpy.ndarray (?), array_img's data matrix but flattened
    """
    image_data = array_img.get_fdata()
    return image_data.flatten()


def resize_images(cropped_imgs, output_dir, ref_image, ident_mx,
                  crop2full, averaged_imgs, j_args, logger):
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
    :param crop2full: String, valid path to existing crop2full.mat file
    :param averaged_imgs: Dictionary mapping ints, (T) 1 or 2, to strings
                          (valid paths to existing image files to resize)
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    """
    # Build dictionaries of variables used for image transformations with and
    # without ACPC alignment
    xfm_non_ACPC_vars = {"out_dir": os.path.join(output_dir, "xfms"),
                         "resolution": "1", "ident_mx": ident_mx,
                         "ref_img": ref_image}
    xfm_ACPC_vars = xfm_non_ACPC_vars.copy()
    xfm_ACPC_vars["out_dir"] = os.path.join(output_dir, "ACPC_align")
    out_var = "output_T{}w_img"
    reg_in_var = "reg_input_T{}w_img"

    for t, crop_img_path in cropped_imgs.items():
        img_ext = split_2_exts(crop_img_path)[-1]

        # Non-ACPC input to registration
        # for keyname in ("crop_", "reg_input_"):
        xfm_non_ACPC_vars["crop_T{}w_img".format(t)] = crop_img_path  # TODO This variable appears to be unused for non-ACPC
        xfm_non_ACPC_vars[reg_in_var.format(t)] = crop_img_path

        # Non-ACPC outputs to registration
        outfname = "T{}w_registered_to_T1w".format(t) + img_ext
        xfm_non_ACPC_vars[out_var.format(t)] = os.path.join(
            xfm_non_ACPC_vars["out_dir"], outfname
        )

        # ACPC inputs to align and registration
        xfm_ACPC_vars["crop_T{}w_img".format(t)] = crop_img_path
        xfm_ACPC_vars[reg_in_var.format(t)] = os.path.join(
            xfm_ACPC_vars["out_dir"], "ACPC_aligned_T{}w".format(t) + img_ext
        )
        xfm_ACPC_vars[out_var.format(t)] = os.path.join(
            xfm_ACPC_vars["out_dir"], "ACPC_" + outfname
        )

    if j_args["common"]["verbose"]:
        msg_xfm = "Arguments for {}ACPC image transformation:\n{}"
        logger.info(msg_xfm.format("non-", xfm_non_ACPC_vars))
        logger.info(msg_xfm.format("", xfm_ACPC_vars))

    # Make output directories for transformed images
    for each_xfm_vars_dict in (xfm_non_ACPC_vars, xfm_ACPC_vars):
        os.makedirs(each_xfm_vars_dict["out_dir"], exist_ok=True)

    xfm_imgs_non_ACPC = registration_T2w_to_T1w(
        j_args, logger, xfm_non_ACPC_vars, reg_in_var, acpc=False
    )

    # Do direct T1w-T2w alignment
    for t in (1, 2):

        # Run ACPC alignment
        xfm_ACPC_vars["mats_T{}w".format(t)] = align_ACPC_1_img(
            j_args, logger, xfm_ACPC_vars, crop2full[t], reg_in_var, t,
            averaged_imgs["T{}w_avg".format(t)]
        )

    # T1w-T2w alignment of ACPC-aligned images
    xfm_ACPC_and_registered_imgs = registration_T2w_to_T1w(
        j_args, logger, xfm_ACPC_vars, reg_in_var, acpc=True
    )

    # TODO End function here and start a new function below? Maybe put everything above in "register_all_preBIBSnet_imgs" and everything below in "apply_final_preBIBSnet_xfm" ?

    # ACPC
    preBIBS_ACPC_out = dict()
    preBIBS_nonACPC_out = dict()
    for t in (1, 2):
        preBIBS_ACPC_out["T{}w".format(t)] = os.path.join(
            xfm_ACPC_vars["out_dir"],
            "preBIBSnet_final_000{}.nii.gz".format(t-1)
        )

        # Concatenate rigidbody2acpc.mat and registration (identity/cropT2tocropT1.mat)
        # First concatenate rigidbody2acpc with registration, then concatenate
        # the output .mat with the template
        acpc2rigidbody = xfm_ACPC_vars["mats_T{}w".format(t)]["acpc2rigidbody"]
        to_rigidbody_final_mat = os.path.join(xfm_ACPC_vars["out_dir"], 
                                              "T2w_to_rigidbody.mat"
                                              ) if t == 2 else acpc2rigidbody
        
        # final_mat differs between T1w and T2w because T2w has to go into T1w
        # space before ACPC and T1w does not 
        if t == 2:
            run_FSL_sh_script( 
                j_args, logger, "convert_xfm", "-omat", to_rigidbody_final_mat,
                "-concat",
                xfm_ACPC_and_registered_imgs["cropT{}tocropT1".format(t)],
                acpc2rigidbody
            )

        crop2BIBS_mat_symlink = os.path.join(xfm_ACPC_vars["out_dir"],
                                     "crop_T{}w_to_BIBS_template.mat".format(t))
        if not os.path.exists(crop2BIBS_mat_symlink):
            os.symlink(to_rigidbody_final_mat, crop2BIBS_mat_symlink)
        preBIBS_ACPC_out["T{}w_crop2BIBS_mat".format(t)] = crop2BIBS_mat_symlink

        # Do the applywarp FSL command from align_ACPC_1_img (for T1w and T2w, for ACPC)
        # applywarp output is optimal_realigned_imgs input
        # Apply registration and ACPC alignment to the T1ws and the T2ws
        run_FSL_sh_script(j_args, logger, "applywarp", "--rel", 
                          "--interp=spline", "-i", averaged_imgs["T{}w_avg".format(t)],
                          "-r", xfm_ACPC_vars["ref_img"].format(t),
                          "--premat=" + crop2BIBS_mat_symlink, # preBIBS_ACPC_out["T{}w_crop2BIBS_mat".format(t)],
                          "-o", preBIBS_ACPC_out["T{}w".format(t)])
        # pdb.set_trace()  # TODO Add "debug" flag?

    # Non-ACPC  # TODO MODULARIZE (put this into a function and call it once for ACPC and once for non to eliminate redundancy)
        preBIBS_nonACPC_out["T{}w".format(t)] = os.path.join(
            xfm_non_ACPC_vars["out_dir"],
            "preBIBSnet_final_000{}.nii.gz".format(t-1)
        )
        
        # Do convert_xfm to combine 2 .mat files (non-ACPC
        # registration_T2w_to_T1w's cropT2tocropT1.mat, and then non-ACPC
        # registration_T2w_to_T1w's crop_T1_to_BIBS_template.mat)
        preBIBS_nonACPC_out["T{}w_crop2BIBS_mat".format(t)] = os.path.join(
            xfm_non_ACPC_vars["out_dir"], "full_crop_T{}w_to_BIBS_template.mat".format(t)
        )
        full2cropT1w_mat = os.path.join(xfm_non_ACPC_vars["out_dir"],
                                        "full2cropT1w.mat")
        run_FSL_sh_script( 
            j_args, logger, "convert_xfm",
            "-omat", full2cropT1w_mat,
            "-concat", xfm_ACPC_vars["mats_T{}w".format(t)]["full2crop"], 
            xfm_imgs_non_ACPC["cropT{}tocropT1".format(t)]
        )
        run_FSL_sh_script( 
            j_args, logger, "convert_xfm",
            "-omat", preBIBS_nonACPC_out["T{}w_crop2BIBS_mat".format(t)],
            "-concat", full2cropT1w_mat,
            xfm_imgs_non_ACPC["T{}w_crop2BIBS_mat".format(t)]
        )
        # Do the applywarp FSL command from align_ACPC_1_img (for T2w and not T1w, for non-ACPC)
        # applywarp output is optimal_realigned_imgs input
        # Apply registration to the T1ws and the T2ws
        run_FSL_sh_script(j_args, logger, "applywarp", "--rel",
                          "--interp=spline", "-i", averaged_imgs["T{}w_avg".format(t)], # cropped_imgs[t],
                          "-r", xfm_non_ACPC_vars["ref_img"].format(t),
                          "--premat=" + preBIBS_nonACPC_out["T{}w_crop2BIBS_mat".format(t)],  # full2BIBS_mat, # 
                          "-o", preBIBS_nonACPC_out["T{}w".format(t)])

    # Outputs: 1 .mat file for ACPC and 1 for non-ACPC (only retain the -to-T1w .mat file after this point)

    # Return the best of the 2 resized images
    # pdb.set_trace()  # TODO Add "debug" flag?
    return optimal_realigned_imgs(preBIBS_nonACPC_out,
                                  preBIBS_ACPC_out, j_args, logger)
  

def run_FSL_sh_script(j_args, logger, fsl_fn_name, *fsl_args):
    """
    Run any FSL function in a Bash subprocess, unless its outputs exist and the
    parameter file said not to overwrite outputs
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
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
            logger.info("Skipping FSL {} command because its output image(s) "
                        "listed below exist(s) and overwrite=False.\n{}"
                        .format(fsl_fn_name, "\n".join(outputs)))

    # Otherwise, just run the FSL command
    else:
        if j_args["common"]["verbose"]:
            logger.info("Now running FSL command:\n{}"
                        .format(" ".join(to_run)))
        subprocess.check_call(to_run)

    # pdb.set_trace()  # TODO Add "debug" flag?


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
            log_stage_finished(name, stage_start, logger)
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


def valid_output_dir_or_none(path):
    """
    Try to make a folder for new files at path, unless "path" is just False.
    Throw exception if that fails
    :param path: String which should be either a valid (not necessarily real)
                 folder path or None
    :return: Either None or a validated absolute path to real writeable folder
    """
    return path if path is None else valid_output_dir(path)


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


# TODO Use --clean-env flag to prevent contamination of any Singularity run by outside environment variables
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
                       "new_file_path": always_true,  # TODO Make "valid_output_filename" function to ensure that filenames don't have spaces or slashes, and maaaaybe to ensure that the new file's parent directory exists?
                       "optional_new_dirpath": valid_output_dir_or_none,
                       "optional_real_dirpath": valid_output_dir_or_none,
                       "positive_float": valid_positive_float,
                       "positive_int": valid_whole_number, 
                       "str": always_true}

    required_for_stage = {
        "nibabies": ["cifti_output", "fd_radius", "work_dir"],
        "XCPD": ["cifti", "combineruns", "fd_thresh",
                 "head_radius", "input_type"]
    }

    # Get a list of all stages after the last stage to run
    after_end = stage_names[stage_names.index(j_args["stage_names"]["end"])+1:]

    # Verify parameters in each section
    to_delete = list()
    for section_name, section_dict in j_types.items():

        # Skip the j_args sections for stages not being run
        if section_name in stage_names and section_name in after_end:
            if section_name in j_args:
                to_delete.append(section_name)

        # Only include resource_management if we're in SLURM/SBATCH job(s)
        elif not (section_name == "resource_management"
                  and not j_args["meta"]["slurm"]):

            # Validate every parameter in the section
            for arg_name, arg_type in section_dict.items():

                # Ignore XCP-D and nibabies parameters that are null
                arg_value = j_args[section_name][arg_name]
                if not (arg_value is None and
                        section_name in ("nibabies", "XCPD") and
                        arg_value not in required_for_stage[section_name]):
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
        if isinstance(arg_type, str):
            type_validators[arg_type](to_validate)
            
        # Verify that the parameter is a valid member of a choices list
        elif isinstance(arg_type, list) and to_validate not in arg_type:
            parser.error(
                err_msg.format(to_validate, arg_name, section_name, param_json,
                               "Valid {} values: {}"
                               .format(arg_name, ", ".join(arg_type)))
            )

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
    for thing in (warning, logger, to_check):
        print(thing, type(thing))
    problems = list()
    for condition, problem in to_check.items():
        if condition:
            problems.append(problem)
    print(type(warning))
    logger.warning(warning.format(" and ".join(problems)))


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