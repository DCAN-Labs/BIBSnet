import os
import shutil
from glob import glob
import sys
import subprocess
from nipype.interfaces import fsl

from src.logger import LOGGER

from src.utilities import (
    correct_chirality, 
    dilate_LR_mask,
    generate_sidecar_json,
    get_subj_ID_and_session,
    get_template_age_closest_to,
    only_Ts_needed_for_bibsnet_model, 
    reverse_regn_revert_to_native
)

SCRIPT_DIR = os.path.dirname(os.path.dirname(__file__))
LR_REGISTR_PATH = os.path.join(SCRIPT_DIR, "bin", "LR_mask_registration.sh")


def run_postBIBSnet(j_args):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :return: j_args, unchanged
    """
    sub_ses = get_subj_ID_and_session(j_args)

    # Template selection values
    age_months = j_args["ID"]["age_months"]
    LOGGER.info("Age of participant: {} months".format(age_months))

    # Get template closest to age
    tmpl_age = get_template_age_closest_to(
        age_months, os.path.join(SCRIPT_DIR, "data", "chirality_masks")
    )
    if j_args["common"]["verbose"]:
        LOGGER.info("Closest template-age is {} months".format(tmpl_age))

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
        sub_ses, tmpl_age, t1or2, j_args
    )
    LOGGER.info("Left/right image registration completed")

    # Dilate the L/R mask and feed the dilated mask into chirality correction
    if j_args["common"]["verbose"]:
        LOGGER.info("Now dilating left/right mask")
    dilated_LRmask_fpath = dilate_LR_mask(
        os.path.join(j_args["optional_out_dirs"]["postbibsnet"], *sub_ses),
        left_right_mask_nifti_fpath
    )
    LOGGER.info("Finished dilating left/right segmentation mask")
    nifti_file_paths, chiral_out_dir, xfm_ref_img_dict = run_correct_chirality(dilated_LRmask_fpath, j_args)
    for t in only_Ts_needed_for_bibsnet_model(j_args["ID"]):
        nii_outfpath = reverse_regn_revert_to_native(
            nifti_file_paths, chiral_out_dir, xfm_ref_img_dict[t], t, j_args
        )
        
        LOGGER.info("The BIBSnet segmentation has had its chirality checked and "
                    "registered if needed. Now making aseg-derived mask.")

        # TODO Skip mask creation if outputs already exist and not j_args[common][overwrite]
        aseg_mask = make_asegderived_mask(j_args, chiral_out_dir, t, nii_outfpath)  # NOTE Mask must be in native T1 space too
        LOGGER.info(f"A mask of the BIBSnet T{t} segmentation has been produced")

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
        LOGGER.info("Working Directory removed at {}."
                    "To keep the working directory in the future,"
                    "set a directory with the --work-dir flag.\n"
                    .format(j_args['common']['work_dir']))
    LOGGER.info("PostBIBSnet has completed.")
    return j_args


def run_correct_chirality(l_r_mask_nifti_fpath, j_args):
    """
    :param l_r_mask_nifti_fpath: String, valid path to existing left/right
                                 registration output mask file
    :param j_args: Dictionary containing all args from parameter .JSON file
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
        LOGGER.error(f"There must be exactly one BIBSnet segmentation file: "
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
    LOGGER.info(msg.format("Now"))
    nii_fpaths = correct_chirality(
        seg_BIBSnet_outfiles[0], segment_lookup_table_path,
        l_r_mask_nifti_fpath, chiral_out_dir
    )
    LOGGER.info(msg.format("Finished"))

    return nii_fpaths, chiral_out_dir, chiral_ref_img_fpaths_dict


def run_left_right_registration(sub_ses, age_months, t1or2, j_args):
    """
    :param sub_ses: List with either only the subject ID str or the session too
    :param age_months: String or int, the subject's age [range] in months
    :param t1or2: Int, 1 to use T1w image for registration or 2 to use T2w
    :param j_args: Dictionary containing all args from parameter .JSON file
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
                LOGGER.info(msg.format("Now running", "\n".join(
                    (first_subject_head, " ".join(cmd_LR_reg))
                )))
            subprocess.check_call(cmd_LR_reg)

        # Tell the user if ANTS crashes due to a memory error
        except subprocess.CalledProcessError as e:
            if e.returncode == 143:
                LOGGER.error(msg.format("ANTS", first_subject_head)
                             + " failed because it ran without enough memory."
                             " Try running it again, but with more memory.\n")
            sys.exit(e)
    else:
        LOGGER.info(msg.format("Skipping",  "{} because output already exists at {}".format(
            first_subject_head, left_right_mask_nifti_fpath
        )))
    LOGGER.info(msg.format("Finished", first_subject_head))  # TODO Only print this message if not skipped (and do the same for all other stages)
    return left_right_mask_nifti_fpath


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