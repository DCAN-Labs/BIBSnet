import os
import shutil
from glob import glob
import sys
import subprocess
from nipype.interfaces import fsl
import nibabel as nib
import numpy as np
import json

from src.logger import LOGGER

from src.utilities import ( 
    get_subj_ID_and_session,
    get_age_closest_to,
    only_Ts_needed_for_bibsnet_model, 
    run_FSL_sh_script,
    split_2_exts
)

SCRIPT_DIR = os.path.dirname(os.path.dirname(__file__))
LR_REGISTR_PATH = os.path.join(SCRIPT_DIR, "bin", "LR_mask_registration.sh")
# Chirality-checking constants
CHIRALITY_CONST = dict(UNKNOWN=0, LEFT=1, RIGHT=2, BILATERAL=3)
LEFT = "Left-"
RIGHT = "Right-"


def run_postBIBSnet(j_args):
    """
    :param j_args: Dictionary containing all args
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
    if j_args["common"]["work_dir"] == os.path.join("/", "tmp", "bibsnet"):
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
    :param j_args: Dictionary containing all args
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
    :param j_args: Dictionary containing all args
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
    :param j_args: Dictionary containing all args
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
    :param j_args: Dictionary containing all args
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

    version = os.environ['BIBSNET_VERSION']
    bids_version = "1.4.0"

    reference = os.path.basename(reference_path)
    spatial_reference = '/'.join(sub_ses) + f"/anat/{reference}"

    sidecar["SpatialReference"] = spatial_reference
    sidecar["BIDSVersion"] = bids_version
    sidecar["GeneratedBy"][0]["Version"] = version
    sidecar["GeneratedBy"][0]["Container"]["Tag"] = f"dcanumn/bibsnet:{version}"
    
    filename = '_'.join(sub_ses) + f"_space-T{t}w_desc-{desc}.json"
    file_path = os.path.join(derivs_dir, filename)

    with open(file_path, "w+") as file:
        json.dump(sidecar, file)


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


def reverse_regn_revert_to_native(nifti_file_paths, chiral_out_dir,
                                  xfm_ref_img, t, j_args):
    """
    :param nifti_file_paths: Dict with valid paths to native and
                             chirality-corrected images
    :param chiral_out_dir: String, valid path to existing directory to save 
                           chirality-corrected images into
    :param xfm_ref_img: String, path to (T1w, unless running in T2w-only mode) 
                        image to use as a reference when applying transform
    :param t: 1 or 2, whether running on T1 or T2
    :param j_args: Dictionary containing all args
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
    run_FSL_sh_script(j_args, "convert_xfm", "-omat",
                      seg2native, "-inverse", preBIBSnet_mat)
    # TODO Define preBIBSnet_mat path outside of stages because it's used by preBIBSnet and postBIBSnet

    run_FSL_sh_script(j_args, "flirt", "-applyxfm",
                      "-ref", xfm_ref_img, "-in", dummy_copy,
                      "-init", seg2native, "-o", nifti_file_paths[f"native-T{t}"],
                      "-interp", "nearestneighbour")
    return nifti_file_paths[f"native-T{t}"]

