import os
import shutil
from glob import glob
from nipype.interfaces import fsl
import nibabel as nib
import numpy as np
import json
from scipy import ndimage

from src.logger import LOGGER

from src.utilities import (
    list_files,
    get_subj_ID_and_session,
    only_Ts_needed_for_bibsnet_model, 
    run_FSL_sh_script
)

SCRIPT_DIR = os.path.dirname(os.path.dirname(__file__))

def run_postBIBSnet(j_args):
    """
    :param j_args: Dictionary containing all args
    :return: j_args, unchanged
    """
    # Define variables and paths
    sub_ses = get_subj_ID_and_session(j_args)
    list_files(j_args["common"]["work_dir"])
    out_BIBSnet_seg = os.path.join(j_args["optional_out_dirs"]["bibsnet"], *sub_ses, "output", "{}_optimal_resized.nii.gz".format("_".join(sub_ses)))

    # Generate derivatives folders to output final files to
    LOGGER.info("Generating output derivatives folders")
    bibsnet_derivs_dir = os.path.join(j_args["optional_out_dirs"]["derivatives"], "bibsnet")
    derivs_dir = os.path.join(bibsnet_derivs_dir, *sub_ses, "anat")
    os.makedirs(derivs_dir, exist_ok=True)

    LOGGER.info("Now registering BIBSnet segmentation to native space to generate derivatives.")
    for t in only_Ts_needed_for_bibsnet_model(j_args["ID"]):
        # Take inverse of .mat file from prebibsnet
        seg2native = os.path.join(j_args["optional_out_dirs"]["postbibsnet"], *sub_ses, f"seg_reg_to_T{t}w_native.mat")
        preBIBSnet_mat_glob = os.path.join(j_args["optional_out_dirs"]["postbibsnet"], *sub_ses, 
        f"preBIBSnet_*crop_T{t}w_to_BIBS_template.mat") 

        preBIBSnet_mat = glob(preBIBSnet_mat_glob).pop()
        run_FSL_sh_script(j_args, "convert_xfm", "-omat",
                      seg2native, "-inverse", preBIBSnet_mat)
        
        # Revert segmentation to native space using average anatomical as reference image and write out to derivatives folder
        av_filename="{}_000{}.nii.gz".format("_".join(sub_ses), t-1)
        avg_anat = os.path.join(j_args["optional_out_dirs"]["prebibsnet"], *sub_ses, "averaged", av_filename)
        aseg=os.path.join(derivs_dir, ("{}_space-T{}w_desc-{}.nii.gz".format("_".join(sub_ses), t, "aseg_dseg")))

        run_FSL_sh_script(j_args, "flirt", "-applyxfm",
                    "-ref", avg_anat, "-in", out_BIBSnet_seg,
                    "-init", seg2native, "-o", aseg,
                    "-interp", "nearestneighbour")
        
        LOGGER.info(f"BIBSNet segmentation has been transformed into native T{t} space")

        # Generate brainmask from segmentation and write out to derivatives folder
        mask_temp=os.path.join(derivs_dir, ("{}_space-T{}w_desc-{}.nii.gz".format("_".join(sub_ses), t, "aseg_dseg")))
        make_asegderived_mask(j_args, aseg, t, mask_temp)

        LOGGER.info(f"A mask of the BIBSnet T{t} segmentation has been produced")

        # Generate sidecar jsons for derivatives
        input_path = os.path.join(j_args["common"]["bids_dir"],
                                               *sub_ses, "anat",
                                               f"*T{t}w.nii.gz")
        reference_path = glob(input_path)[0]
        generate_sidecar_json(sub_ses, reference_path, derivs_dir, t, "aseg_dseg")
        generate_sidecar_json(sub_ses, reference_path, derivs_dir, t, "brain_mask")

    # Copy dataset_description.json into bibsnet_derivs_dir directory for use in nibabies
    list_files(j_args["common"]["work_dir"])
    new_data_desc_json = os.path.join(bibsnet_derivs_dir, "dataset_description.json")
    if j_args["common"]["overwrite"]:
        os.remove(new_data_desc_json)
    if not os.path.exists(new_data_desc_json):
        shutil.copy2(os.path.join(SCRIPT_DIR, "data",
                                  "dataset_description.json"), new_data_desc_json)
    if j_args["common"]["work_dir"] == os.path.join("/", "tmp", "bibsnet"):
        cleanup_work_dir(j_args)
        
    list_files(j_args["common"]["work_dir"])

    return j_args

    # Write j_args out to logs
    #LOGGER.debug(j_args)

def save_nifti(data, affine, file_path):
    img = nib.Nifti1Image(data, affine)
    nib.save(img, file_path)

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

    remove_extra_clusters_from_mask(output_mask_fpath, path_to_aseg=nii_outfpath)
    
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
        json.dump(sidecar, file, indent = 4)



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
    LOGGER.info("Removing outlying clusters")
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
    LOGGER.info("Saving mask without outlying clusters")
    nib.save(new_mask, path_to_mask)
    
    if type(path_to_aseg) != type(None):
        aseg_img = nib.load(path_to_aseg)
        aseg_data = aseg_img.get_fdata()
        aseg_data[new_mask_data != 1] = 0
        new_aseg = nib.nifti1.Nifti1Image(aseg_data.astype(np.uint8), affine=aseg_img.affine, header=aseg_img.header)
        LOGGER.info("Saving aseg without outlying clusters")
        nib.save(new_aseg, path_to_aseg)

    return


def cleanup_work_dir(j_args):
    subses = [j_args["ID"]["subject"]]
    if "session" in j_args["ID"]:
        subses.append(j_args["ID"]["session"])

    stages = ["prebibsnet", "bibsnet", "postbibsnet"]

    for stage in stages:
        to_remove = os.path.join(j_args["common"]["work_dir"], stage, *subses)
        shutil.rmtree(to_remove)
        LOGGER.verbose(f"Working Directory removed at {to_remove}.")
        
    LOGGER.verbose("To keep the working directory in the future, set a directory with the --work-dir flag.")
