#!/usr/bin/env python3
# coding: utf-8

"""
Common source for utility functions used by CABINET :)
Greg Conan: gconan@umn.edu
Created: 2021-11-12
Updated: 2023-01-26
"""
# Import standard libraries
import argparse
from datetime import datetime  # for seeing how long scripts take to run
from glob import glob
import json
import nibabel as nib
from nipype.interfaces import fsl
import numpy as np
import os
import pdb
import shutil
import subprocess
import sys
from scipy import ndimage

# Chirality-checking constants
CHIRALITY_CONST = dict(UNKNOWN=0, LEFT=1, RIGHT=2, BILATERAL=3)
LEFT = "Left-"
RIGHT = "Right-"

# Other constant: Directory containing the main pipeline script
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
    input_img = xfm_ACPC_vars[f"crop_T{t}w_img"]  # Cropped img, ACPC input
    output_img =  xfm_ACPC_vars[output_var.format(t)]  # ACPC-aligned image
    mats = {fname: os.path.join(work_dir, f"T{t}w_{fname}.mat")
            for fname in ("crop2acpc", "full2acpc", "full2crop",
                          "acpc2rigidbody")}  # .mat file paths
    
    run_FSL_sh_script(j_args, logger, "flirt", "-interp", "spline",  
                      "-ref", mni_ref_img_path, "-in", input_img,
                      "-omat", mats["crop2acpc"], # "-out", os.path.join(work_dir, "T{}w_acpc_final.nii.gz".format(t)),
                      "-searchrx", "-45", "45", "-searchry", "-30", "30",
                      "-searchrz", "-30", "30")

    # Invert crop2full to get full2crop
    run_FSL_sh_script(j_args, logger, "convert_xfm", "-inverse", crop2full,
                      "-omat", mats["full2crop"])  # TODO Move this to right after making crop2full to use it in both T?w-only and here

    run_FSL_sh_script(  # Combine ACPC-alignment with robustFOV output
        j_args, logger, "convert_xfm", "-omat", mats["full2acpc"],
        "-concat", mats["crop2acpc"], mats["full2crop"]
    )

    # Transform 12 dof matrix to 6 dof approximation matrix
    run_FSL_sh_script(j_args, logger, "aff2rigid", mats["full2acpc"],
                      mats["acpc2rigidbody"])

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


def apply_final_ACPC_xfm(xfm_vars, xfm_imgs, avg_imgs, outputs,
                         t, j_args, logger):
    """
    Apply entire image transformation (from cropped image to BIBSnet format)
    with ACPC transformation to a T1w or T2w image 
    :param xfm_vars: Dict with variables specific to images to transform
    :param xfm_imgs: Dict with paths to image files
    :param avg_imgs: Dictionary mapping ints, (T) 1 or 2, to strings
                     (valid paths to existing image files to resize)
    :param outputs: Dict that will have T1w &/or T2w ACPC transformed images
    :param t: Int, either 1 or 2 (to signify T1w or T2w respectively)
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return: outputs, with paths to ACPC-transformed image and transform .mat
    """
    outputs[f"T{t}w"] = get_preBIBS_final_img_fpath_T(
        t, xfm_vars["out_dir"], j_args["ID"]
    )

    # Concatenate rigidbody2acpc.mat and registration (identity/cropT2tocropT1.mat)
    # First concatenate rigidbody2acpc with registration, then concatenate
    # the output .mat with the template
    acpc2rigidbody = xfm_vars[f"mats_T{t}w"]["acpc2rigidbody"]
    to_rigidbody_final_mat = os.path.join(xfm_vars["out_dir"], 
                                            "T2w_to_rigidbody.mat"
                                            ) if t == 2 else acpc2rigidbody
    
    # final_mat differs between T1w and T2w because T2w has to go into T1w
    # space before ACPC and T1w does not 
    if t == 2:
        run_FSL_sh_script( 
            j_args, logger, "convert_xfm", "-omat", to_rigidbody_final_mat,
            "-concat", xfm_imgs[f"cropT{t}tocropT1"],
            acpc2rigidbody
        )

    crop2BIBS_mat = os.path.join(xfm_vars["out_dir"],
                                 f"crop_T{t}w_to_BIBS_template.mat")
    if not os.path.exists(crop2BIBS_mat):
        shutil.copy2(to_rigidbody_final_mat, crop2BIBS_mat)
        if j_args["common"]["verbose"]:
            logger.info("Copying {} to {}".format(to_rigidbody_final_mat,
                                                    crop2BIBS_mat))
    outputs[f"T{t}w_crop2BIBS_mat"] = crop2BIBS_mat

    # Do the applywarp FSL command from align_ACPC_1_img (for T1w and T2w, for ACPC)
    # applywarp output is optimal_realigned_imgs input
    # Apply registration and ACPC alignment to the T1ws and the T2ws
    run_FSL_sh_script(j_args, logger, "applywarp", "--rel", 
                        "--interp=spline", "-i", avg_imgs[f"T{t}w_avg"],
                        "-r", xfm_vars["ref_img"].format(t),
                        "--premat=" + crop2BIBS_mat, # preBIBS_ACPC_out["T{}w_crop2BIBS_mat".format(t)],
                        "-o", outputs[f"T{t}w"])
    # pdb.set_trace()  # TODO Add "debug" flag?

    return outputs


def apply_final_non_ACPC_xfm(xfm_vars, xfm_imgs, avg_imgs,
                             outputs, t, full2crop_ACPC, j_args, logger):
    """
    Apply entire image transformation (from cropped image to BIBSnet format)
    without ACPC transformation to a T1w or T2w image 
    :param xfm_vars: Dict with variables specific to images to transform
    :param xfm_ACPC_imgs: Dict with paths to image files
    :param avg_imgs: Dictionary mapping ints, (T) 1 or 2, to strings
                     (valid paths to existing image files to resize)
    :param outputs: Dict that will have T1w &/or T2w ACPC transformed images
    :param t: Int, either 1 or 2 (to signify T1w or T2w respectively)
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return: outputs, with paths to ACPC-transformed image and transform .mat
    """
    outputs[f"T{t}w"] = get_preBIBS_final_img_fpath_T(
        t, xfm_vars["out_dir"], j_args["ID"]
    )
    
    # Do convert_xfm to combine 2 .mat files (non-ACPC
    # registration_T2w_to_T1w's cropT2tocropT1.mat, and then non-ACPC
    # registration_T2w_to_T1w's crop_T1_to_BIBS_template.mat)
    outputs[f"T{t}w_crop2BIBS_mat"] = os.path.join(
        xfm_vars["out_dir"], f"full_crop_T{t}w_to_BIBS_template.mat"  
    )  # NOTE Changed crop_T{}w... back to full_crop_T{}w... on 2022-08-30
    full2crop_mat = os.path.join(xfm_vars["out_dir"],
                                 f"full2cropT{t}w.mat")
    run_FSL_sh_script( 
        j_args, logger, "convert_xfm",
        "-omat", full2crop_mat,
        "-concat", full2crop_ACPC, xfm_imgs["cropT1tocropT1"]
    )
    if t == 1:
        run_FSL_sh_script( 
            j_args, logger, "convert_xfm",
            "-omat", outputs[f"T{t}w_crop2BIBS_mat"],
            "-concat", full2crop_mat, 
            xfm_imgs[f"T{t}w_crop2BIBS_mat"]
        )
    else: # if t == 2:
        crop_and_reg_mat = os.path.join(xfm_vars["out_dir"],
                                        "full2cropT2toT1.mat")
        run_FSL_sh_script( 
            j_args, logger, "convert_xfm", "-omat", crop_and_reg_mat,
            "-concat", xfm_imgs[f"cropT{t}tocropT1"], full2crop_mat 
        )
        run_FSL_sh_script(j_args, logger, "convert_xfm", "-omat",
                          outputs[f"T{t}w_crop2BIBS_mat"], "-concat",
                          xfm_imgs[f"T{t}w_crop2BIBS_mat"], crop_and_reg_mat)

    # Do the applywarp FSL command from align_ACPC_1_img
    # (for T2w and not T1w, for non-ACPC)
    # applywarp output is optimal_realigned_imgs input
    # Apply registration to the T1ws and the T2ws
    run_FSL_sh_script(j_args, logger, "applywarp", "--rel",
                      "--interp=spline", "-i", avg_imgs[f"T{t}w_avg"],
                      "-r", xfm_vars["ref_img"].format(t),
                      "--premat=" + outputs[f"T{t}w_crop2BIBS_mat"],
                      "-o", outputs[f"T{t}w"])
    return outputs


def apply_final_prebibsnet_xfms(regn_non_ACPC, regn_ACPC, averaged_imgs,
                                j_args, logger):
    """
    Resize the images to match the dimensions of images trained in the model,
    and ensure that the first image (presumably a T1) is co-registered to the
    second image (presumably a T2) before resizing. Use multiple alignments
    of both images, and return whichever one is better (higher eta squared)
    :param regn_non_ACPC: Dict mapping "img_paths" to a dict of paths to image
                          files and "vars" to a dict of other variables.
                          {"vars": {...}, "imgs": {...}}
    :param regn_ACPC: Dict mapping "img_paths" to a dict of paths to image
                      files and "vars" to a dict of other variables.
                      {"vars": {...}, "imgs": {...}} 
    :param averaged_imgs: Dictionary mapping ints, (T) 1 or 2, to strings
                          (valid paths to existing image files to resize)
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return: Dict with paths to either ACPC- or non-ACPC image & .mat files
    """
    out_ACPC = dict()
    out_non_ACPC = dict()

    for t in (1, 2):
        # Apply ACPC-then-registration transforms for this subject session & T
        out_ACPC.update(apply_final_ACPC_xfm(
            regn_ACPC["vars"], regn_ACPC["img_paths"],
            averaged_imgs, out_ACPC, t, j_args, logger
        ))

        # Retrieve path to ACPC full2crop.mat file (to use for non-ACPC xfms)
        full2crop_ACPC = regn_ACPC["vars"][f"mats_T{t}w"]["full2crop"]

        # Apply registration-only transforms for this subject session (and T)
        out_non_ACPC.update(apply_final_non_ACPC_xfm(
            regn_non_ACPC["vars"], regn_non_ACPC["img_paths"],
            averaged_imgs, out_non_ACPC, t, full2crop_ACPC, j_args, logger
        ))

    # Outputs: 1 .mat file for ACPC and 1 for non-ACPC (only retain the -to-T1w .mat file after this point)

    # Return the best of the 2 resized images
    return optimal_realigned_imgs(out_non_ACPC,  # TODO Add 'if' statement to skip eta-squared functionality if T1-/T2-only, b/c only one T means we'll only register to ACPC space
                                  out_ACPC, j_args, logger)


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
    :return: Float(?), the eta value
    """  
    # get the data from each nifti image as a flattened vector
    vectors = dict()
    for t in (1, 2):  # TODO Make this also work for (T1-only or?) T2-only by comparing to the registered image instead of the other T
        anat = f"T{t}w"
        vectors[anat] = reshape_volume_to_array(nib.load(img_paths[anat]))  # np.abs()
        negatives = vectors[anat][vectors[anat] < 0]
        print("{} has {} negatives.".format(anat, len(negatives)))  # TODO REMOVE LINE

    print(f"Vectors: {vectors}")  # TODO REMOVE LINE
    """
    medians = {
        "grand": (np.median(vectors["T1w"]) + np.median(vectors["T2w"])) / 2,
        "within": np.median(np.concatenate((vectors["T1w"], vectors["T2w"])))
    }
    """
    # mean value over all locations in both images  # TODO Add if statement to not average if T1-/T2-only 
    m_grand = (np.mean(vectors["T1w"]) + np.mean(vectors["T2w"])) / 2  # TODO Try using np.median instead of np.mean?

    # mean value matrix for each location in the 2 images
    m_within = (vectors["T1w"] + vectors["T2w"]) / 2  # TODO Try combining both arrays and taking the median of the result?
    print("Mean Within: {}\nMean Total: {}".format(m_within, m_grand))  # TODO REMOVE LINE

    # sswithin = (sum(np.square(vectors["T1w"] - m_within)) + sum(np.square(vectors["T2w"] - m_within)))
    # sstot = (sum(np.square(vectors["T1w"] - m_grand)) + sum(np.square(vectors["T2w"] - m_grand)))

    sswithin = sum_of_2_sums_of_squares_of(vectors["T1w"], vectors["T2w"], m_within)  # medians["within"])
    sstot = sum_of_2_sums_of_squares_of(vectors["T1w"], vectors["T2w"], m_grand)  # medians["grand"])

    # NOTE SStot = SSwithin + SSbetween so eta can also be
    #      written as SSbetween/SStot
    print("SumSq Within: {}\nSumSq Total: {}".format(sswithin, sstot))  # TODO REMOVE LINE
    return 1 - sswithin / sstot  # Should there be parentheses around (1 - sswithin)?


def check_and_correct_region(should_be_left, region, segment_name_to_number,
                             new_data, chirality, floor_ceiling, scanner_bore):
    """
    Ensures that a voxel in NIFTI data is in the correct region by flipping
    the label if it's mislabeled
    :param should_be_left (Boolean): This voxel *should be on the head's LHS 
    :param region: String naming the anatomical region
    :param segment_name_to_number (map<str, int>): Map from anatomical regions 
                                                   to identifying numbers
    :param new_data (3-d in array): segmentation data passed by reference to 
                                    be fixed if necessary
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
                      nii_fpath_LR_mask, chiral_out_dir):
    """
    Creates an output file with chirality corrections fixed.
    :param nifti_input_file_path: String, path to a segmentation file with
                                  possible chirality problems
    :param segment_lookup_table: String, path to FreeSurfer-style look-up table
    :param nii_fpath_LR_mask: String, path to a mask file that
                              distinguishes between left and right
    :param xfm_ref_img: String, path to (T1w, unless running in T2w-only mode) 
                        image to use as a reference when applying transform
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return: Dict with paths to native and chirality-corrected images
    """
    nifti_file_paths = dict()
    for which_nii in ("native-T1", "native-T2", "corrected"):
        nifti_file_paths[which_nii] = os.path.join(chiral_out_dir, "_".join((
            which_nii, os.path.basename(nifti_input_file_path)
        )))

    free_surfer_label_to_region = get_id_to_region_mapping(segment_lookup_table)
    segment_name_to_number = {v: k for k, v in free_surfer_label_to_region.items()}
    img = nib.load(nifti_input_file_path)
    data = img.get_data()
    left_right_img = nib.load(nii_fpath_LR_mask)
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
    nib.save(fixed_img, nifti_file_paths["corrected"])
    return nifti_file_paths


def create_anatomical_averages(avg_params, logger):
    """
    Creates a NIFTI file whose voxels are the average of the voxel values of the input files.
    :param avg_params: Dictionary with 4 keys:
    {"T1w_input": List (possibly empty) of t1 image file path strings
     "T2w_input": List (possibly empty) of t2 image file path strings
     "T1w_avg": String, average T1w output file path
     "T2w_avg": String, average T2w output file path}
    """   
    for t in (1, 2):
        if avg_params.get(f"T{t}w_input"):
            register_and_average_files(avg_params[f"T{t}w_input"],
                                       avg_params[f"T{t}w_avg"], logger)


def create_avg_image(output_file_path, registered_files):
    """
    Create image which is an average of all registered_files,
    then save it to output_file_path
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
                      "-b", j_args["ID"]["brain_z_size"])  # TODO Use head radius for -b
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
    :param sub_LRmask_dir: String, path to real directory to make subdirectory
                           in; the subdirectory will contain mask files
    :param anatfile: String, valid path to existing anatomical image file
    """
    # Make subdirectory to save masks in & generic mask file name format-string
    parent_dir = os.path.join(sub_LRmask_dir, "lrmask_dil_wd")
    os.makedirs(parent_dir, exist_ok=True)
    mask = os.path.join(parent_dir, "{}mask{}.nii.gz")

    # Make left, right, and middle masks using FSL
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-thr 1 -uthr 1',
                           out_file=mask.format("L", ""))
    maths.run()
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-thr 2 -uthr 2',
                           out_file=mask.format("R", ""))
    maths.run()
    maths.run()
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-thr 3 -uthr 3',
                           out_file=mask.format("M", ""))
    maths.run()

    # dilate, fill, and erode each mask in order to get rid of holes
    # (also binarize L and M images in order to perform binary operations)
    maths = fsl.ImageMaths(in_file=mask.format("L", ""),
                           op_string='-dilM -dilM -dilM -fillh -ero',
                           out_file=mask.format("L", "_holes_filled"))
    maths.run()
    maths = fsl.ImageMaths(in_file=mask.format("R", ""),
                           op_string='-bin -dilM -dilM -dilM -fillh -ero',
                           out_file=mask.format("R", "_holes_filled"))
    maths.run()
    maths = fsl.ImageMaths(in_file=mask.format("M", ""),
                           op_string='-bin -dilM -dilM -dilM -fillh -ero',
                           out_file=mask.format("M", "_holes_filled"))
    maths.run()

    # Reassign values of 2 and 3 to R and M masks (L mask already a value of 1)
    label_anat_masks = {"L": mask.format("L", "_holes_filled"),
                        "R": mask.format("R", "_holes_filled_label2"),
                        "M": mask.format("M", "_holes_filled_label3")}
    maths = fsl.ImageMaths(in_file=mask.format("R", "_holes_filled"),
                           op_string='-mul 2', out_file=label_anat_masks["R"])
    maths.run()

    maths = fsl.ImageMaths(in_file=mask.format("M", "_holes_filled"),
                           op_string='-mul 3', out_file=label_anat_masks["M"])
    maths.run()

    # recombine new L, R, and M mask files and then dilate the result 
    masks_LR = {"dilated": mask.format("dilated_LR", ""),
                "recombined": mask.format("recombined_", "_LR")}
    maths = fsl.ImageMaths(in_file=label_anat_masks["L"],
                           op_string='-add {}'.format(label_anat_masks["R"]),
                           out_file=masks_LR["recombined"])
    maths.run()
    maths = fsl.ImageMaths(in_file=label_anat_masks["M"],
                           op_string="-add {}".format(masks_LR["recombined"]),
                           out_file=masks_LR["dilated"])
    maths.run()

    # Fix incorrect values resulting from recombining dilated components
    orig_LRmask_img = nib.load(os.path.join(sub_LRmask_dir, "LRmask.nii.gz"))
    orig_LRmask_data = orig_LRmask_img.get_fdata()

    fill_LRmask_img = nib.load(masks_LR["dilated"])
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
    out_fpath = mask.format("LR", "_dil")  # os.path.join(sub_LRmask_dir, 'LRmask_dil.nii.gz')
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


def generate_sidecar_json(sub_ses, reference_path, derivs_dir, t, desc):
    """
    :param sub_ses: List with either only the subject ID str or the session too
    :param reference_path: String, filepath to the referenced image
    :param derivs_dir: String, directory to place the output JSON
    :param t: 1 or 2, T1w or T2w
    :param desc: the type of image the sidecar json is being paired with
    """
    template_path = os.path.join(SCRIPT_DIR, "data", "sidecar_template.json")
    with open(template_path) as file:
        sidecar = json.load(file)

    version = os.environ['CABINET_VERSION']
    bids_version = "1.4.0"

    reference = os.path.basename(reference_path)
    spatial_reference = '/'.join(sub_ses) + f"/anat/{reference}"

    sidecar["SpatialReference"] = spatial_reference
    sidecar["BIDSVersion"] = bids_version
    sidecar["GeneratedBy"][0]["Version"] = version
    sidecar["GeneratedBy"][0]["Container"]["Tag"] = f"dcanumn/cabinet:{version}"
    
    filename = '_'.join(sub_ses) + f"_space-T{t}w_desc-{desc}.json"
    file_path = os.path.join(derivs_dir, filename)

    with open(file_path, "w+") as file:
        json.dump(sidecar, file)

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
                            j_args["optional_out_dirs"]["prebibsnet"], *sub_ses
                        )}
    for work_dirname in ("averaged", "cropped", "resized"):
        preBIBSnet_paths[work_dirname] = os.path.join(
            preBIBSnet_paths["parent"], work_dirname
        )
        os.makedirs(preBIBSnet_paths[work_dirname], exist_ok=True)

    # Build paths to BIDS anatomical input images and (averaged, 
    # nnU-Net-renamed) output images
    preBIBSnet_paths["avg"] = dict()
    for t in (1, 2):  # TODO Make this also work for T1-only or T2-only by not creating unneeded T dir(s)
        preBIBSnet_paths["avg"][f"T{t}w_input"] = list()
        for eachfile in glob(os.path.join(j_args["common"]["bids_dir"],
                                          *sub_ses, "anat", 
                                          f"*T{t}w*.nii.gz")):
            preBIBSnet_paths["avg"][f"T{t}w_input"].append(eachfile)
        avg_img_name = "{}_000{}{}".format("_".join(sub_ses), t-1, ".nii.gz")
        preBIBSnet_paths["avg"][f"T{t}w_avg"] = os.path.join(  
            preBIBSnet_paths["averaged"], avg_img_name  
        )  
  
        # Get paths to, and make, cropped image subdirectories  
        crop_dir = os.path.join(preBIBSnet_paths["cropped"], f"T{t}w")  
        preBIBSnet_paths[f"crop_T{t}w"] = os.path.join(crop_dir, avg_img_name)
        os.makedirs(crop_dir, exist_ok=True)
    return preBIBSnet_paths


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


def get_preBIBS_final_img_fpath_T(t, parent_dir, sub_ses_ID):
    """
    Running in T1-/T2-only mode means the image name should always be
    preBIBSnet_final_0000.nii.gz and otherwise it's _000{t-1}.nii.gz
    :param t: Int, either 1 or 2 (to signify T1w or T2w respectively)
    :param parent_dir: String, valid path to directory to hold output images
    :param sub_ses_ID: Dictionary mapping subject-session-specific input
                       parameters' names (as strings) to their values for
                       this subject session; the same as j_args[ID]
    :return: String, valid path to a preBIBSnet final image file
    """
    return os.path.join(parent_dir, "preBIBSnet_final_000{}.nii.gz".format(
        get_preBIBS_final_digit_T(t, sub_ses_ID)
    ))


def get_spatial_resolution_of(image_fpath, j_args, logger, fn_name="fslinfo"):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :param fn_name: String naming the function which is an
                    executable file in j_args[common][fsl_bin_path]
    """  # TODO Do we even need this function?
    # FSL command to run in a subprocess
    to_run = [os.path.join(j_args["common"]["fsl_bin_path"], fn_name),
              image_fpath]

    # Run FSL command and read the file information into a dictionary
    if j_args["common"]["verbose"]:
        logger.info("Now running FSL command:\n{}"
                    .format(" ".join(to_run)))
    img_info = dict()
    for eachline in subprocess.check_output(to_run).decode("utf-8").split("\n"):
        split = eachline.split()
        img_info[split[0]] = split[-1]

    return img_info["pixdim3"]  # A.K.A. brain_z_size


def get_stage_name(stage_fn):
    """ 
    :param stage_fn: Function to run one stage of CABINET. Its name must start
                     with "run_", e.g. "run_nibabies" or "run_preBIBSnet"
    :return: String naming the CABINET stage to run
    """
    return stage_fn.__name__[4:].lower()


def get_sub_base(j_args, run_num=None):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param run_num: Whole number as an int or string defining which run this is
    :return: String identifying a subject, session, task, and maybe run
    """
    parts = [get_subj_ses(j_args), "task-" + j_args["common"]["task_id"]]
    if run_num is not None:
        parts.append(f"run-{run_num}")
    return "_".join(parts)


def get_subj_ID_and_session(j_args):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :return: List of 2 strings (subject ID and session from parameter file,
             with their correct "sub-" and "ses-" prefixes) if the parameter
             file has a session, otherwise just with the prefixed subject ID
    """ 
    sub = ensure_prefixed(j_args["ID"]["subject"], "sub-")
    return [sub, ensure_prefixed(j_args["ID"]["session"], "ses-")
            ] if dict_has(j_args["ID"], "session") else [sub]


def get_subj_ses(j_args):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :return: String combining subject ID and session from parameter file
    """
    return "_".join(get_subj_ID_and_session(j_args))


def get_template_age_closest_to(age, templates_dir):
    """
    :param age: Int, participant age in months
    :param templates_dir: String, valid path to existing directory which
                          contains template image files
    :return: String, the age (or range of ages) in months closest to the
             participant's with a template image file in templates_dir
    """
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
        else:
            template_ages.append(int(tmpl_age))
    
    # Get template age closest to subject age, then return template age
    closest_age = get_age_closest_to(age, template_ages)
    return (template_ranges[closest_age] if closest_age
            in template_ranges else str(closest_age))


def get_age_closest_to(subject_age, all_ages):
    """
    :param subject_age: Int, participant's actual age in months
    :param all_ages: List of ints, each a potential participant age in months
    :return: Int, the age in all_ages which is closest to subject_age
    """
    return all_ages[np.argmin(np.abs(np.array(all_ages)-subject_age))]
    

def glob_and_copy(dest_dirpath, *path_parts_to_glob):
    """
    Collect all files matching a glob string, then copy those files
    :param dest_dirpath: String, a valid path of a directory to copy files into
    :param path_parts_to_glob: Unpacked list of strings which join to form a
                               glob string of a path to copy files from
    """
    for file_src in glob(os.path.join(*path_parts_to_glob)):
        shutil.copy(file_src, dest_dirpath)


def log_stage_finished(stage_name, event_time, sub_ses, logger):
    """
    Print and return a string showing how much time has passed since the
    current running script reached a certain part of its process
    :param stage_name: String, name of event that just finished
    :param event_time: datetime object representing when {stage_name} started
    :param sub_ses: List with either only the subject ID str or the session too
    :return: String with an easily human-readable message showing how much time
             has passed since {stage_start} when {stage_name} started.
    """
    logger.info("{0} finished on subject {1}. "
                "Time elapsed since {0} started: {2}"
                .format(stage_name, " session ".join(sub_ses),
                        datetime.now() - event_time))


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


def optimal_realigned_imgs(xfm_imgs_non_ACPC, xfm_imgs_ACPC_and_reg, j_args, logger):
    """
    Check whether the cost function shows that only the registration-T2-to-T1
    or the ACPC-alignment-and-T2-to-T1-registration is better (check whether
    ACPC alignment improves the T2-to-T1 registration; compare the T2-to-T1
    with and without first doing the ACPC registration)
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    """
    msg = "Using {} T2w-to-T1w registration for resizing.\nT1w: {}\nT2w: {}"
    eta = dict()
    logger.info("\nACPC:")
    eta["ACPC"] = calculate_eta(xfm_imgs_ACPC_and_reg)
    logger.info("\nNon-ACPC:")
    eta["non-ACPC"] = calculate_eta(xfm_imgs_non_ACPC)
    logger.info(f"Eta-Squared Values: {eta}")
    if eta["non-ACPC"] > eta["ACPC"]:
        optimal_resize = xfm_imgs_non_ACPC
        logger.info(msg.format("only", optimal_resize["T1w"],
                               optimal_resize["T2w"]))  # TODO Verify that these print the absolute path
    else:
        optimal_resize = xfm_imgs_ACPC_and_reg
        logger.info(msg.format("ACPC and", optimal_resize["T1w"],
                               optimal_resize["T2w"]))  # TODO Verify that these print the absolute path
    return optimal_resize


def register_preBIBSnet_imgs_ACPC(cropped_imgs, output_dir, xfm_non_ACPC_vars,
                                  crop2full, averaged_imgs, j_args, logger):
    """
    :param cropped_imgs: Dictionary mapping ints, (T) 1 or 2, to strings (valid
                         paths to existing image files to resize)
    :param output_dir: String, valid path to a dir to save resized images into
    :param xfm_non_ACPC_vars: Dict TODO Fix this function description
    :param crop2full: String, valid path to existing crop2full.mat file
    :param averaged_imgs: Dictionary mapping ints, (T) 1 or 2, to strings
                          (valid paths to existing image files to resize)
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    """
    # Build dict of variables used for image transformation with ACPC alignment
    xfm_ACPC_vars = xfm_non_ACPC_vars.copy()
    xfm_ACPC_vars["out_dir"] = os.path.join(output_dir, "ACPC_align")
    out_var = "output_T{}w_img"
    reg_in_var = "reg_input_T{}w_img"

    for t, crop_img_path in cropped_imgs.items():
        img_ext = split_2_exts(crop_img_path)[-1]

        # ACPC inputs to align and registration
        outfname = f"T{t}w_registered_to_T1w" + img_ext
        xfm_ACPC_vars[f"crop_T{t}w_img"] = crop_img_path
        xfm_ACPC_vars[reg_in_var.format(t)] = os.path.join(
            xfm_ACPC_vars["out_dir"], f"ACPC_aligned_T{t}w" + img_ext
        )
        xfm_ACPC_vars[out_var.format(t)] = os.path.join(
            xfm_ACPC_vars["out_dir"], "ACPC_" + outfname
        )

    # Make output directories for transformed images
    os.makedirs(xfm_ACPC_vars["out_dir"], exist_ok=True)

    # Do direct T2w-T1w alignment
    for t in (1, 2):

        # Run ACPC alignment
        xfm_ACPC_vars[f"mats_T{t}w"] = align_ACPC_1_img(
            j_args, logger, xfm_ACPC_vars, crop2full[t], reg_in_var, t,
            averaged_imgs[f"T{t}w_avg"]
        )

    # T2w-T1w alignment of ACPC-aligned images
    xfm_ACPC_and_reg_imgs = registration_T2w_to_T1w(
        j_args, logger, xfm_ACPC_vars, reg_in_var, acpc=True
    )

    # pdb.set_trace()  # TODO Add "debug" flag?

    return {"vars": xfm_ACPC_vars, "img_paths": xfm_ACPC_and_reg_imgs}


def register_preBIBSnet_imgs_non_ACPC(cropped_imgs, output_dir, ref_image, 
                                      ident_mx, resolution, j_args, logger):
    """
    :param cropped_imgs: Dictionary mapping ints, (T) 1 or 2, to strings (valid
                         paths to existing image files to resize)
    :param output_dir: String, valid path to a dir to save resized images into
    :param ref_images: Dictionary mapping string keys to valid paths to real
                       image file strings for "ACPC" (alignment) and (T2-to-T1)
                       "reg"(istration) for flirt to use as a reference image.
                       The ACPC string has a "{}" in it to represent (T) 1 or 2
    :param ident_mx: String, valid path to existing identity matrix .mat file
    :param resolution:
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    """
    # TODO Add 'if' to skip most of the functionality here for T1-only or T2-only

    # Build dictionaries of variables used for image transformations with and
    # without ACPC alignment
    xfm_non_ACPC_vars = {"out_dir": os.path.join(output_dir, "xfms"),
                         "resolution": resolution, "ident_mx": ident_mx,
                         "ref_img": ref_image}
    out_var = "output_T{}w_img"
    reg_in_var = "reg_input_T{}w_img"

    for t, crop_img_path in cropped_imgs.items():
        img_ext = split_2_exts(crop_img_path)[-1]

        # Non-ACPC input to registration
        # for keyname in ("crop_", "reg_input_"):
        xfm_non_ACPC_vars[reg_in_var.format(t)] = crop_img_path

        # Non-ACPC outputs to registration
        outfname = f"T{t}w_registered_to_T1w" + img_ext
        xfm_non_ACPC_vars[out_var.format(t)] = os.path.join(
            xfm_non_ACPC_vars["out_dir"], outfname
        )

    # Make output directory for transformed images
    os.makedirs(xfm_non_ACPC_vars["out_dir"], exist_ok=True)

    xfm_imgs_non_ACPC = registration_T2w_to_T1w(
        j_args, logger, xfm_non_ACPC_vars, reg_in_var, acpc=False
    )

    # pdb.set_trace()  # TODO Add "debug" flag?

    return {"vars": xfm_non_ACPC_vars, "img_paths": xfm_imgs_non_ACPC}


def register_and_average_files(input_file_paths, output_file_path, logger):
    """
    Register all input image files, and if there are multiple of them, then
    create an average of all of them
    :param input_file_paths: List of strings, each a valid path to an existing
                             image file to register
    :param output_file_path: String, valid path to image file to create by
                             averaging all of the input_file_paths images
    :param logger: logging.Logger object to show messages and raise warnings
    """
    reference = input_file_paths[0]
    out_dir=os.path.dirname(output_file_path)
    if len(input_file_paths) > 1:
        registered_files = register_files(input_file_paths, reference,
                                          out_dir, logger)

        create_avg_image(output_file_path, registered_files)
    else:
        shutil.copyfile(reference, output_file_path)


def register_files(input_file_paths, reference, out_dir, logger):
    """
    :param input_file_paths: List of strings, each a valid path to an existing
                             image file to register
    :param reference: String, valid path to existing img to register others to
    :param out_dir: String, valid path to existing directory to save registered
                    images into
    :param logger: logging.Logger object to show messages and raise warnings
    :raises RuntimeError: If FSL FLIRT command to register images fails
    :return: List of strings, each a valid path to a newly created image file,
             starting with the reference image and then every input_file_paths
             image registered to that reference image
    """
    registered_files = [reference]
    
    # Build FSL FLIRT object by first making name of output nifti & mat files
    ref_fname, ref_ext = split_2_exts(reference)
    ref_fname, tw = os.path.basename(ref_fname).split("_T")
    t = int(tw[0])
    flt = fsl.FLIRT(
        bins=640, cost_func='mutualinfo',
        out_file=os.path.join(out_dir, f"{ref_fname}_desc-avg_T{t}w{ref_ext}"),  # TODO This file is redundant, was added to try to prevent create_avg_image from trying to save an average image to a relative path, bc that breaks in a container
        out_matrix_file=os.path.join(out_dir, f"T{t}_avg.mat")
    )
    flt.inputs.reference = reference
    flt.inputs.output_type = "NIFTI_GZ"
    for structural in input_file_paths[1:]:

        # Build FSL command to register each file
        flt.inputs.in_file = structural
        logger.info("Now running FSL FLIRT:\n{}".format(flt.cmdline))
        out_index = flt.cmdline.find('-out')
        start_index = out_index + len('-out') + 1
        end_index = flt.cmdline.find(' ', start_index)
        out = flt.cmdline[start_index:end_index]
        registered_files.append(out)

        # Run each FSL command
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
    inputs_msg = "\n".join(["T{}w: {}".format(t, xfm_vars[reg_input_var.format(t)])
                            for t in only_Ts_needed_for_bibsnet_model(j_args["ID"])])
    logger.info("Input images for T1w registration:\n" + inputs_msg)

    # Define paths to registration output matrices and images
    registration_outputs = {"cropT1tocropT1": xfm_vars["ident_mx"],
                            "cropT2tocropT1": os.path.join(xfm_vars["out_dir"],
                                                           "cropT2tocropT1.mat")}
    """
    ACPC Order:
    1. T1w Save cropped and aligned T1w image 
    2. T2w Make T2w-to-T1w matrix

    NonACPC Order:
    1. T1w Make transformed
    2. T2w Make T2w-to-T1w matrix
    3. T2w Make transformed
    """   
    for t in (1, 2):
        # Define paths to registration output files
        registration_outputs[f"T{t}w_crop2BIBS_mat"] = os.path.join(
            xfm_vars["out_dir"], f"crop_T{t}w_to_BIBS_template.mat"
        )
        registration_outputs[f"T{t}w"] = xfm_vars[f"output_T{t}w_img"]
        registration_outputs[f"T{t}w_to_BIBS"] = os.path.join(
            xfm_vars["out_dir"], f"T{t}w_to_BIBS.nii.gz"
        )

        if t == 2:  # Make T2w-to-T1w matrix
            run_FSL_sh_script(j_args, logger, "flirt",
                            "-ref", xfm_vars[reg_input_var.format(1)],
                            "-in", xfm_vars[reg_input_var.format(2)],
                            "-omat", registration_outputs["cropT2tocropT1"],
                            "-out", registration_outputs["T2w"],
                            '-cost', 'mutualinfo',
                            '-searchrx', '-15', '15', '-searchry', '-15', '15',
                            '-searchrz', '-15', '15', '-dof', '6')

        elif acpc:  # Save cropped and aligned T1w image 
            shutil.copy2(xfm_vars[reg_input_var.format(1)],
                         registration_outputs["T1w"])

        # Make transformed T1ws and T2ws
        if not acpc:  # TODO Should this go in its own function?
            transform_image_T(
                t, (xfm_vars[reg_input_var.format(t)] if t == 1 else
                    registration_outputs["T2w"]),
                xfm_vars, registration_outputs, j_args, logger
            )
            run_FSL_sh_script(  # TODO Should the output image even be created here, or during applywarp?
                j_args, logger, "flirt",
                "-in", xfm_vars[reg_input_var.format(t)] if t == 1 else registration_outputs["T2w"],  # Input: Cropped image
                "-ref", xfm_vars["ref_img"].format(t),
                "-applyisoxfm", xfm_vars["resolution"],
                "-init", xfm_vars["ident_mx"], # registration_outputs["cropT{}tocropT1".format(t)],
                "-o", registration_outputs[f"T{t}w_to_BIBS"], # registration_outputs["T{}w".format(t)],  # TODO Should we eventually exclude the (unneeded?) -o flags?
                "-omat", registration_outputs[f"T{t}w_crop2BIBS_mat"]
            )
    # pdb.set_trace()  # TODO Add "debug" flag?
    return registration_outputs

def remove_extra_clusters_from_mask(path_to_mask, path_to_aseg = None):
    '''Function that removes smaller/unconnected clusters from brain mask
    
    Parameters
    ----------
    
    path_to_mask : str
        Path to the binary (0/1) brain mask file to be edited.
    path_to_aseg : str or None (default None)
        Optional path to the corresponding aseg image. If provided,
        the areas from small clusters (defined in mask space) will
        be set to zero in a new copy of this image
        
    Returns
    -------
    None
    
    Makes new copies that replace the input mask file and optionally
    aseg file. These new nifti images will have smaller non-brain regions
    (defined based on the mask image) set to zero.
    
    '''

    mask_img = nib.load(path_to_mask)
    #seg_img = nib.load(path_to_seg)
    
    temp_data = mask_img.get_fdata()
    labels, nb = ndimage.label(temp_data)
    largest_label_size = 0
    largest_label = 0
    for i in range(nb + 1):
        if i == 0:
            continue
        label_size = np.sum(labels == i)
        if label_size > largest_label_size:
            largest_label_size = label_size
            largest_label = i
    new_mask_data = np.zeros(temp_data.shape)
    new_mask_data[labels == largest_label] = 1
    new_mask = nib.nifti1.Nifti1Image(new_mask_data.astype(np.uint8), affine=mask_img.affine, header=mask_img.header)
    nib.save(new_mask, path_to_mask)
    
    if type(path_to_aseg) != type(None):
        aseg_img = nib.load(path_to_aseg)
        aseg_data = aseg_img.get_fdata()
        aseg_data[new_mask_data != 1] = 0
        new_aseg = nib.nifti1.Nifti1Image(aseg_data.astype(np.uint8), affine=aseg_img.affine, header=aseg_img.header)
        nib.save(new_aseg, path_to_aseg)

    return

def transform_image_T(t, cropped_in_img, xfm_vars, regn_outs, j_args, logger):
    """
    Run FSL command on a cropped input image to apply a .mat file transform 
    :param t: Int, either 1 or 2 (to signify T1w or T2w respectively)
    :param cropped_in_img: String, valid path to cropped T1w or T2w image
    :param xfm_vars: Dict with paths to reference image & identity matrix files
    :param regn_outs: Dict with paths to transformed output images to make
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    """
    run_FSL_sh_script(  # TODO Should the output image even be created here, or during applywarp?
        j_args, logger, "flirt",
        "-in", cropped_in_img, # xfm_vars[reg_input_var.format(t)] if t == 1 else registration_outputs["T2w"],  # Input: Cropped image
        "-ref", xfm_vars["ref_img"].format(t),
        "-applyisoxfm", xfm_vars["resolution"],
        "-init", xfm_vars["ident_mx"], # registration_outputs["cropT{}tocropT1".format(t)],
        "-o", regn_outs[f"T{t}w_to_BIBS"], # registration_outputs["T{}w".format(t)],  # TODO Should we eventually exclude the (unneeded?) -o flags?
        "-omat", regn_outs[f"T{t}w_crop2BIBS_mat"]
    )


def reshape_volume_to_array(array_img):
    """ 
    :param array_img: nibabel.Nifti1Image (or Nifti2Image?)
    :return: numpy.ndarray (?), array_img's data matrix but flattened
    """
    image_data = array_img.get_fdata()
    return image_data.flatten()


def reverse_regn_revert_to_native(nifti_file_paths, chiral_out_dir,
                                  xfm_ref_img, t, j_args, logger):
    """
    :param nifti_file_paths: Dict with valid paths to native and
                             chirality-corrected images
    :param chiral_out_dir: String, valid path to existing directory to save 
                           chirality-corrected images into
    :param xfm_ref_img: String, path to (T1w, unless running in T2w-only mode) 
                        image to use as a reference when applying transform
    :param t: 1 or 2, whether running on T1 or T2
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return: String, valid path to existing image reverted to native
    """
    sub_ses = get_subj_ID_and_session(j_args)

    # Undo resizing right here (do inverse transform) using RobustFOV so 
    # padding isn't necessary; revert aseg to native space
    dummy_copy = "_dummy".join(split_2_exts(nifti_file_paths["corrected"]))
    shutil.copy2(nifti_file_paths["corrected"], dummy_copy)

    seg2native = os.path.join(chiral_out_dir, f"seg_reg_to_T{t}w_native.mat")
    preBIBSnet_mat_glob = os.path.join(
        j_args["optional_out_dirs"]["postbibsnet"], *sub_ses, 
        f"preBIBSnet_*crop_T{t}w_to_BIBS_template.mat"  # TODO Name this outside of pre- and postBIBSnet then pass it to both
    )
    preBIBSnet_mat = glob(preBIBSnet_mat_glob).pop()
    run_FSL_sh_script(j_args, logger, "convert_xfm", "-omat",
                      seg2native, "-inverse", preBIBSnet_mat)
    # TODO Define preBIBSnet_mat path outside of stages because it's used by preBIBSnet and postBIBSnet

    run_FSL_sh_script(j_args, logger, "flirt", "-applyxfm",
                      "-ref", xfm_ref_img, "-in", dummy_copy,
                      "-init", seg2native, "-o", nifti_file_paths[f"native-T{t}"],
                      "-interp", "nearestneighbour")
    return nifti_file_paths[f"native-T{t}"]


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


def run_all_stages(all_stages, sub_ses_IDs, start, end,
                   ubiquitous_j_args, logger):
    """
    Run stages sequentially, starting and ending at stages specified by user
    :param all_stages: List of functions in order where each runs one stage
    :param sub_ses_IDs: List of dicts mapping "age_months", "subject",
                        "session", etc. to unique values per subject session
    :param start: String naming the first stage the user wants to run
    :param end: String naming the last stage the user wants to run
    :param ubiquitous_j_args: Dictionary of all args needed by each stage
    :param logger: logging.Logger object to show messages and raise warnings
    """
    if ubiquitous_j_args["common"]["verbose"]:
        logger.info("All parameters from input args and input .JSON file:\n{}"
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
        verify_CABINET_inputs_exist(sub_ses, sub_ses_j_args, logger)

        # ...run all stages that the user said to run
        for stage in all_stages:
            name = get_stage_name(stage)
            if name == start:
                running = True
            if running:
                stage_start = datetime.now()
                if sub_ses_j_args["common"]["verbose"]:
                    logger.info("Now running {} stage on:\n{}"
                                .format(name, sub_ses_j_args["ID"]))
                sub_ses_j_args = stage(sub_ses_j_args, logger)
                log_stage_finished(name, stage_start, sub_ses, logger)
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


def sum_of_2_sums_of_squares_of(np_vector1, np_vector2, a_mean):
    """
    :param np_vector1: Numpy array of numbers
    :param np_vector2: Numpy array of numbers
    :param a_mean: Float, _description_
    :return: Float, the sum of squares of each vector, added together
    """
    total_sum = 0
    for each_vec in (np_vector1, np_vector2):
        total_sum += sum(np.square(each_vec - a_mean))
    return total_sum
                    

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


def valid_subj_ses_ID(to_validate):
    """
    :param to_validate: Object to turn into a valid subject/session ID label
    :return: String, valid subject/session ID label
    """  # TODO Validate that subject/session exists 
    return validate(to_validate, always_true, lambda x: x.split("-")[-1],
                    "{} is not a valid subject/session ID.")


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
        "xcpd": ["cifti", "combineruns", "fd_thresh",
                 "head_radius", "input_type"]
    }

    # Get a list of all stages after the last stage to run
    after_end = stage_names[stage_names.index(j_args["stage_names"]["end"])+1:]

    # Verify parameters in each section
    to_delete = list()
    for section_orig_name, section_dict in j_types.items():
        section_name = section_orig_name.lower()  # TODO Should we change parameter types .JSON file to make section names already lowercase?

        # Skip the j_args sections for stages not being run
        if section_name in stage_names and section_name in after_end:
            if section_orig_name in j_args:
                to_delete.append(section_orig_name)

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


def verify_CABINET_inputs_exist(sub_ses, j_args, logger):
    """
    Given a stage, verify that all of the necessary inputs for that stage exist 
    :param a_stage: String naming a stage
    :param sub_ses: List with either only the subject ID str or the session too
    :param j_args: Dictionary containing all args from parameter .JSON file
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
                           "postbibsnet": [out_BIBSnet_seg, *subject_heads],
                           "nibabies": out_paths_BIBSnet,
                           "xcpd": list()}

    # For each stage that will be run, verify that its prereq input files exist
    all_stages = [s for s in stage_prerequisites.keys()]
    start_ix = all_stages.index(j_args["stage_names"]["start"]) 
    for stage in all_stages[:start_ix+1]:
        missing_files = list()
        for globbable in stage_prerequisites[stage]:
            if not glob(globbable):
                missing_files.append(globbable)
        if missing_files:
            logger.error("The file(s) below are needed to run the {} stage, "
                        "but they do not exist.\n{}\n"
                        .format(stage, "\n".join(missing_files)))
            sys.exit(1)

    logger.info("All required input files exist.")


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