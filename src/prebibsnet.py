import os
import shutil
import nibabel as nib
from nipype.interfaces import fsl
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.interfaces.ants import DenoiseImage
from nipype.interfaces.ants import N4BiasFieldCorrection
from niworkflows.interfaces.nibabel import IntensityClip
import numpy as np
from glob import glob
from skimage.metrics import structural_similarity as ssim

from src.logger import LOGGER

from src.utilities import (
    list_files,
    get_subj_ID_and_session,
    only_Ts_needed_for_bibsnet_model,
    run_FSL_sh_script,
    split_2_exts,
    get_preBIBS_final_digit_T
) 

SCRIPT_DIR = os.path.dirname(os.path.dirname(__file__))

def run_preBIBSnet(j_args):
    """
    :param j_args: Dictionary containing all args
    :return: j_args, but with preBIBSnet working directory names added
    """
    list_files(j_args["common"]["work_dir"])
    completion_msg = "The anatomical images have been {} for use in BIBSnet"
    preBIBSnet_paths = get_and_make_preBIBSnet_work_dirs(j_args)
    sub_ses = get_subj_ID_and_session(j_args)

    # If there are multiple T1ws/T2ws, then average them
    create_anatomical_averages(preBIBSnet_paths["avg"])

    # On average image(s), run: intensity clip -> denoise -> N4 -> reclip
    for t in only_Ts_needed_for_bibsnet_model(j_args["ID"]):
        mod=f"T{t}w"
        denoise_and_n4(mod, preBIBSnet_paths["avg"][f"T{t}w_avg"])

    # Generate brainmask with SynthStrip, use as reference to determine axial cutting plane, and crop average T1w and T2w images
    cropped = dict()
    crop_dir= dict()
    brainmask = dict()
    crop2full = dict()
    for t in only_Ts_needed_for_bibsnet_model(j_args["ID"]):
        cropped[t] = preBIBSnet_paths[f"crop_T{t}w"]
        crop_dir[t] = os.path.dirname(cropped[t])

        brainmask[t] = synthstrip(preBIBSnet_paths["avg"][f"T{t}w_avg"], 
                                  crop_dir[t]
                                  )
        crop2full[t] = crop_image(preBIBSnet_paths["avg"][f"T{t}w_avg"],
                                  brainmask[t],
                                  cropped[t]
                                  )
    LOGGER.info(completion_msg.format("cropped"))

    # Resize T1w and T2w images if running a BIBSnet model using T1w and T2w
    reference_img = os.path.join(SCRIPT_DIR, "data", "MNI_templates",
                                 "INFANT_MNI_T{}_1mm.nii.gz")
    LOGGER.debug(f"reference_img: {reference_img}")
    id_mx = os.path.join(SCRIPT_DIR, "data", "identity_matrix.mat")
    # TODO Resolution is hardcoded; infer it or get it from the command-line
    resolution = "1"

    # Perform T2w-to-T1w registration via multiple methods if both T1w and T2w are present (including non-ACPC/xfms and ACPC)
    if j_args["ID"]["has_T1w"] and j_args["ID"]["has_T2w"]:
        msg_xfm = "Arguments for {}ACPC image transformation:\n{}"

        # Non-ACPC with and without restricted search space parameters 
        regn_non_ACPC = register_preBIBSnet_imgs_non_ACPC(
            cropped, preBIBSnet_paths["resized"], reference_img, 
            id_mx, resolution, j_args
        )
        LOGGER.verbose(msg_xfm.format("non-", regn_non_ACPC["vars"]))

        LOGGER.verbose(msg_xfm.format("non-", regn_non_ACPC["vars"]))

        # ACPC
        regn_ACPC = register_preBIBSnet_imgs_ACPC(
            cropped, preBIBSnet_paths["resized"], regn_non_ACPC["vars"],
            crop2full, preBIBSnet_paths["avg"], j_args
        )
        LOGGER.verbose(msg_xfm.format("", regn_ACPC["vars"]))

        # Apply final transforms to average images
        transformed_images = apply_final_prebibsnet_xfms(
            regn_non_ACPC, regn_ACPC, preBIBSnet_paths["avg"], j_args
        )
        LOGGER.info(completion_msg.format("resized"))

    # If running a T1w-only or T2w-only BIBSnet model, skip registration/resizing
    else:
        # Define variables and paths needed for the final (only) xfm needed
        t1or2 = 1 if j_args["ID"]["has_T1w"] else 2
        outdir = os.path.join(preBIBSnet_paths["resized"], "xfms")
        os.makedirs(outdir, exist_ok=True)
        out_img = get_preBIBS_final_img_fpath_T(t1or2, outdir, j_args["ID"])
        crop2BIBS_mat = os.path.join(outdir,
                                     "crop2BIBS_T{}w_only.mat".format(t1or2))
        out_mat = os.path.join(outdir, "full_crop_T{}w_to_BIBS_template.mat"
                                       .format(t1or2))

        run_FSL_sh_script(  # Get xfm moving the T1 (or T2) into BIBS space
            j_args, "flirt", "-in", cropped[t1or2],
            "-ref", reference_img.format(t1or2), "-applyisoxfm", resolution,
            "-init", id_mx, 
            "-omat", crop2BIBS_mat
        )

        # Invert crop2full to get full2crop
        # TODO Move this to right after making crop2full, then delete the 
        #      duplicated functionality in align_ACPC_1_image
        full2crop = os.path.join(
            os.path.dirname(preBIBSnet_paths["avg"][f"T{t}w_avg"]),
            f"full2crop_T{t}w_only.mat"
        )
        run_FSL_sh_script(j_args, "convert_xfm", "-inverse",
                          crop2full[t], "-omat", full2crop) 

        # Concatenate crop .mat to out_mat (in that order) and apply the concatenated .mat to the averaged image as the output
        # Treat that concatenated output .mat as the output to pass along to postBIBSnet, and the image output to BIBSnet
        run_FSL_sh_script(  # Combine ACPC-alignment with robustFOV output
            j_args, "convert_xfm", "-omat", out_mat,
            "-concat", full2crop, crop2BIBS_mat
        )
        run_FSL_sh_script(  # Apply concat xfm to crop and move into BIBS space
            j_args, "applywarp", "--rel", "--interp=spline",
            "-i", preBIBSnet_paths["avg"][f"T{t}w_avg"],
            "-r", reference_img.format(t1or2),
            "--premat=" + out_mat, "-o", out_img
        )
        transformed_images = {f"T{t1or2}w": out_img,
                              f"T{t1or2}w_crop2BIBS_mat": out_mat}
        LOGGER.debug(f"transformed_images: {transformed_images}")

    # TODO Copy this whole block to postBIBSnet, so it copies everything it needs first
    # Copy preBIBSnet outputs into BIBSnet input dir
    for t in only_Ts_needed_for_bibsnet_model(j_args["ID"]): 
        # Copy image files
        out_nii_fpath = j_args["optimal_resized"][f"T{t}w"]
        LOGGER.debug(f"out_nii_fpath: {out_nii_fpath}")
        os.makedirs(os.path.dirname(out_nii_fpath), exist_ok=True)
        if j_args["common"]["overwrite"]:  # TODO Should --overwrite delete old image file(s)?
            os.remove(out_nii_fpath)
        if not os.path.exists(out_nii_fpath): 
            shutil.copy2(transformed_images[f"T{t}w"], out_nii_fpath)

        # Copy .mat into postbibsnet dir with the same name regardless of which
        # is chosen, so postBIBSnet can use the correct/chosen .mat file
        concat_mat = transformed_images[f"T{t}w_crop2BIBS_mat"]
        LOGGER.debug(f"concat_mat: {concat_mat}")
       
         # TODO Pass this in (or out) from the beginning so we don't have to build the path twice (once here and once in postBIBSnet)
        out_mat_fpath = os.path.join(
            j_args["optional_out_dirs"]["postbibsnet"],
            *sub_ses, "preBIBSnet_" + os.path.basename(concat_mat)
        )

        list_files(j_args["optional_out_dirs"]["postbibsnet"])
        list_files(j_args["common"]["work_dir"])
        LOGGER.debug(f"out_mat_fath: {out_mat_fpath}")
        if not os.path.exists(out_mat_fpath):
            shutil.copy2(concat_mat, out_mat_fpath)
            LOGGER.verbose(f"Copying {concat_mat} to {out_mat_fpath}")

    return j_args

def denoise_and_n4(tmod, input_avg_img):
    #t, preBIBSnet_paths["avg"], preBIBSnet_paths["avg"][f"T{t}w_avg"])
    """
    Run robustFOV to crop image
    :param t: String, T1w or T2w
    :param input_avg_img: String, valid path to averaged (T1w or T2w) image
    """
    LOGGER.info("Denoising input avg image")
    wd=os.path.dirname(input_avg_img)
    wf = pe.Workflow(name=f'{tmod}_denoise_and_bfcorrect', base_dir=wd)

    # Skip if clipped image already exists
    src=os.path.join(wd, f'{tmod}_denoise_and_bfcorrect/final_clip/clipped.nii.gz')
    if os.path.exists(src):
        LOGGER.info(f"Skipping denoise and N4 correction, {src} already exists")
        return src

    else:
        inputnode = pe.Node(
            niu.IdentityInterface(fields=["in_anat"]),
            name="inputnode",
            )
        outputnode = pe.Node(
            niu.IdentityInterface(fields=["out_anat"]),
            name="outputnode",
        )

        inputnode.inputs.in_anat = input_avg_img

        clip = pe.Node(IntensityClip(p_min=10.0, p_max=99.5), name="clip")
        denoise = pe.Node(DenoiseImage(dimension=3, noise_model="Rician"), name="denoise")
        n4_correct=pe.Node(N4BiasFieldCorrection(
                dimension=3,
                bspline_fitting_distance=200,
                save_bias=True,
                copy_header=True,
                n_iterations=[50] * 5,
                convergence_threshold=1e-7,
                rescale_intensities=True,
                shrink_factor=4), 
                name="n4_correct")
        final_clip = pe.Node(IntensityClip(p_min=5.0, p_max=99.5), name="final_clip")

        wf.connect([
            (inputnode, clip, [("in_anat", "in_file")]),
            (clip, denoise, [("out_file", "input_image")]),
            (denoise, n4_correct, [("output_image", "input_image")]),
            (n4_correct, final_clip, [("output_image", "in_file")]),
            (final_clip, outputnode, [("out_file", "out_anat")]),
        ])
        wf.run()

    dest=input_avg_img
    shutil.copy(src, dest)

def apply_final_prebibsnet_xfms(regn_non_ACPC, regn_ACPC, averaged_imgs,
                                j_args):
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
    :param j_args: Dictionary containing all args
    :return: Dict with paths to either ACPC- or non-ACPC image & .mat files
    """
    out_ACPC = dict()
    out_non_ACPC = dict()

    for t in (1, 2):
        # Apply ACPC-then-registration transforms for this subject session & T
        out_ACPC.update(apply_final_ACPC_xfm(
            regn_ACPC["vars"], regn_ACPC["img_paths"],
            averaged_imgs, out_ACPC, t, j_args
        ))

        # Retrieve path to ACPC full2crop.mat file (to use for non-ACPC xfms)
        full2crop_ACPC = regn_ACPC["vars"][f"mats_T{t}w"]["full2crop"]

        # Apply registration-only transforms for this subject session (and T)
        out_non_ACPC.update(apply_final_non_ACPC_xfm(
            regn_non_ACPC["vars"], regn_non_ACPC["img_paths"],
            averaged_imgs, out_non_ACPC, t, full2crop_ACPC, j_args
        ))

    # Outputs: 1 .mat file for ACPC and 1 for non-ACPC (only retain the -to-T1w .mat file after this point)

    # Return the best of the 2 resized images
    return optimal_realigned_imgs(out_non_ACPC,  # TODO Add 'if' statement to skip eta-squared functionality if T1-/T2-only, b/c only one T means we'll only register to ACPC space
                                  out_ACPC, j_args)


def apply_final_ACPC_xfm(xfm_vars, xfm_imgs, avg_imgs, outputs,
                         t, j_args):
    """
    Apply entire image transformation (from cropped image to BIBSnet format)
    with ACPC transformation to a T1w or T2w image 
    :param xfm_vars: Dict with variables specific to images to transform
    :param xfm_imgs: Dict with paths to image files
    :param avg_imgs: Dictionary mapping ints, (T) 1 or 2, to strings
                     (valid paths to existing image files to resize)
    :param outputs: Dict that will have T1w &/or T2w ACPC transformed images
    :param t: Int, either 1 or 2 (to signify T1w or T2w respectively)
    :param j_args: Dictionary containing all args
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
            j_args, "convert_xfm", "-omat", to_rigidbody_final_mat,
            "-concat", xfm_imgs[f"cropT{t}tocropT1"],
            acpc2rigidbody
        )

    crop2BIBS_mat = os.path.join(xfm_vars["out_dir"],
                                 f"crop_T{t}w_to_BIBS_template.mat")
    if not os.path.exists(crop2BIBS_mat):
        shutil.copy2(to_rigidbody_final_mat, crop2BIBS_mat)
        LOGGER.verbose("Copying {} to {}".format(to_rigidbody_final_mat,
                                                crop2BIBS_mat))
    outputs[f"T{t}w_crop2BIBS_mat"] = crop2BIBS_mat

    # Do the applywarp FSL command from align_ACPC_1_img (for T1w and T2w, for ACPC)
    # applywarp output is optimal_realigned_imgs input
    # Apply registration and ACPC alignment to the T1ws and the T2ws
    run_FSL_sh_script(j_args, "applywarp", "--rel", 
                        "--interp=spline", "-i", avg_imgs[f"T{t}w_avg"],
                        "-r", xfm_vars["ref_img"].format(t),
                        "--premat=" + crop2BIBS_mat, # preBIBS_ACPC_out["T{}w_crop2BIBS_mat".format(t)],
                        "-o", outputs[f"T{t}w"])
    # pdb.set_trace()  # TODO Add "debug" flag?

    return outputs


def apply_final_non_ACPC_xfm(xfm_vars, xfm_imgs, avg_imgs,
                             outputs, t, full2crop_ACPC, j_args):
    """
    Apply entire image transformation (from cropped image to BIBSnet format)
    without ACPC transformation to a T1w or T2w image 
    :param xfm_vars: Dict with variables specific to images to transform
    :param xfm_ACPC_imgs: Dict with paths to image files
    :param avg_imgs: Dictionary mapping ints, (T) 1 or 2, to strings
                     (valid paths to existing image files to resize)
    :param outputs: Dict that will have T1w &/or T2w ACPC transformed images
    :param t: Int, either 1 or 2 (to signify T1w or T2w respectively)
    :param j_args: Dictionary containing all args
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
        j_args, "convert_xfm",
        "-omat", full2crop_mat,
        "-concat", full2crop_ACPC, xfm_imgs["cropT1tocropT1"]
    )
    if t == 1:
        run_FSL_sh_script( 
            j_args, "convert_xfm",
            "-omat", outputs[f"T{t}w_crop2BIBS_mat"],
            "-concat", full2crop_mat, 
            xfm_imgs[f"T{t}w_crop2BIBS_mat"]
        )
    else: # if t == 2:
        crop_and_reg_mat = os.path.join(xfm_vars["out_dir"],
                                        "full2cropT2toT1.mat")
        run_FSL_sh_script( 
            j_args, "convert_xfm", "-omat", crop_and_reg_mat,
            "-concat", xfm_imgs[f"cropT{t}tocropT1"], full2crop_mat 
        )
        run_FSL_sh_script(j_args, "convert_xfm", "-omat",
                          outputs[f"T{t}w_crop2BIBS_mat"], "-concat",
                          xfm_imgs[f"T{t}w_crop2BIBS_mat"], crop_and_reg_mat)

    # Do the applywarp FSL command from align_ACPC_1_img
    # (for T2w and not T1w, for non-ACPC)
    # applywarp output is optimal_realigned_imgs input
    # Apply registration to the T1ws and the T2ws
    run_FSL_sh_script(j_args, "applywarp", "--rel",
                      "--interp=spline", "-i", avg_imgs[f"T{t}w_avg"],
                      "-r", xfm_vars["ref_img"].format(t),
                      "--premat=" + outputs[f"T{t}w_crop2BIBS_mat"],
                      "-o", outputs[f"T{t}w"])
    return outputs

import os

def optimal_realigned_imgs(xfm_imgs_non_ACPC, xfm_imgs_ACPC_and_reg, j_args):
    """
    Check whether the cost function shows that only the registration-T2-to-T1
    or the ACPC-alignment-and-T2-to-T1-registration is better (check whether
    ACPC alignment improves the T2-to-T1 registration; compare the T2-to-T1
    with and without first doing the ACPC registration)
    :param j_args: Dictionary containing all args
    """
    msg = "Using {} T2w-to-T1w registration for resizing."
    eta = dict()
    # ssim = dict()

    # eta squared calculations
    eta["ACPC"] = calculate_eta(xfm_imgs_ACPC_and_reg)
    eta["non-ACPC"] = calculate_eta(xfm_imgs_non_ACPC)

    # ssim calculations
    # ssim["ACPC"] = calculate_ssim(xfm_imgs_ACPC_and_reg)
    # ssim["non-ACPC"] = calculate_ssim(xfm_imgs_non_ACPC)

    LOGGER.verbose(f"Eta-Squared Values: {eta}")
    # LOGGER.verbose(f"SSIM Values: {ssim}")

    # Save results to a text file in parent directory of xfm_imgs_ACPC_and_reg
    try:
        acpc_t1_path = xfm_imgs_ACPC_and_reg['T1w']
        output_dir = os.path.abspath(os.path.join(os.path.dirname(acpc_t1_path), ".."))
        output_path = os.path.join(output_dir, "registration_metrics_acpc-vs-xfms.txt")

        with open(output_path, "w") as f:
            f.write("Eta-Squared Values:\n")
            for k, v in eta.items():
                f.write(f"  {k}: {v:.4f}\n")
            # f.write("\nSSIM Values:\n")
            # for k, v in ssim.items():
            #     f.write(f"  {k}: {v:.4f}\n")
    except Exception as e:
        LOGGER.warning(f"Could not write registration metrics to file: {e}")

    if eta["non-ACPC"] > eta["ACPC"]:
        optimal_resize = xfm_imgs_non_ACPC
        LOGGER.info(msg.format("only"))
        LOGGER.verbose(f"\nT1w: {optimal_resize['T1w']}\nT2w: {optimal_resize['T2w']}")
        LOGGER.verbose(f"Best registration path: non-ACPC/XFMS")
    else:
        optimal_resize = xfm_imgs_ACPC_and_reg
        LOGGER.info(msg.format("ACPC and"))
        LOGGER.verbose(f"\nT1w: {optimal_resize['T1w']}\nT2w: {optimal_resize['T2w']}")
        LOGGER.verbose(f"Best registration path: ACPC")

    return optimal_resize

def calculate_eta(img_paths):
    """
    :param img_paths: Dictionary mapping "T1w" and "T2w" to strings that are
                      valid paths to the existing respective image files
    :return: Float(?), the eta value
    """  
    vectors = dict()
    for t in (1, 2):  
        anat = f"T{t}w"
        vectors[anat] = reshape_volume_to_array(nib.load(img_paths[anat]))

    m_grand = (np.mean(vectors["T1w"]) + np.mean(vectors["T2w"])) / 2  # mean value over all locations in both images
    m_within = (vectors["T1w"] + vectors["T2w"]) / 2  # mean value matrix for each location in the 2 images

    sswithin = sum_of_2_sums_of_squares_of(vectors["T1w"], vectors["T2w"], m_within)
    sstot = sum_of_2_sums_of_squares_of(vectors["T1w"], vectors["T2w"], m_grand) 
    LOGGER.verbose(f"\nVectors: {vectors}\nMean Within: {m_within}\nMean Total: {m_grand}\nSumSq Within: {sswithin}\nSumSq Total: {sstot}")

    return 1 - sswithin / sstot

def calculate_ssim(img_paths):
    """
    Computes the Structural Similarity Index (SSIM) between two MRI images.
    SSIM ranges from -1 to 1, where 1 indicates perfect similarity.
    :param img_paths: Dictionary mapping "T1w" and "T2w" to strings that are
                      valid paths to the existing respective image files
    :return: SSIM value (Float)
    """  
    # Load niftis as arrays and normalize intensity values to [0,1] for SSIM
    vectors = dict()
    for t in (1, 2): 
        anat = f"T{t}w"
        vectors[anat]=nib.load(img_paths[anat]).get_fdata()
        vectors[anat] = (vectors[anat] - np.min(vectors[anat])) / (np.max(vectors[anat]) - np.min(vectors[anat])) # normalize

    # Compute SSIM
    ssim_value = ssim(vectors["T1w"], vectors["T2w"], data_range=1.0)

    return ssim_value

def reshape_volume_to_array(array_img):
    """ 
    :param array_img: nibabel.Nifti1Image (or Nifti2Image?)
    :return: numpy.ndarray (?), array_img's data matrix but flattened
    """
    image_data = array_img.get_fdata()
    return image_data.flatten()


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


def create_anatomical_averages(avg_params):
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
                                       avg_params[f"T{t}w_avg"])


def register_and_average_files(input_file_paths, output_file_path):
    """
    Register all input image files, and if there are multiple of them, then
    create an average of all of them
    :param input_file_paths: List of strings, each a valid path to an existing
                             image file to register
    :param output_file_path: String, valid path to image file to create by
                             averaging all of the input_file_paths images
    """
    reference = input_file_paths[0]
    out_dir=os.path.dirname(output_file_path)
    if len(input_file_paths) > 1:
        registered_files = register_files(input_file_paths, reference,
                                          out_dir)

        create_avg_image(output_file_path, registered_files)
    else:
        shutil.copyfile(reference, output_file_path)


def register_files(input_file_paths, reference, out_dir):
    """
    :param input_file_paths: List of strings, each a valid path to an existing
                             image file to register
    :param reference: String, valid path to existing img to register others to
    :param out_dir: String, valid path to existing directory to save registered
                    images into
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
        LOGGER.info("Now running FSL FLIRT:\n{}".format(flt.cmdline))
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

def synthstrip(input_avg_img, output_crop_dir):
    '''
    Generate brainmasks for T1 and T2 using SynthStrip
    :param input_avg_img: String, valid path to averaged (T1w or T2w) image
    :param output_crop_dir: String, valid folder path to save (T1w or T2w) brainmask image out to
    '''
    # Define skullstripped anat filepath
    skullstripped_img=os.path.join(output_crop_dir, "skullstripped.nii.gz")
    brainmask_img=os.path.join(output_crop_dir, "brainmask.nii.gz")

    # Run SynthStrip and delete output skullstripped_img (not needed for BIBSNet)
    os.system(f'python /opt/freesurfer/bin/mri_synthstrip -i {input_avg_img} -o {skullstripped_img} -m {brainmask_img}')
    os.remove(skullstripped_img)

    brainmask=os.path.join(output_crop_dir, brainmask_img)

    return brainmask

def crop_image(input_avg_img, brainmask_img, output_crop_img):
    """
    Use SynthStrip-derived brainmask to define axial cutting plane as
    10 voxels lower than lower edge of brainmask, crop average image using axial cutting plane, and generate full2crop.mat transform
    :param input_avg_img: String, valid path to averaged (T1w or T2w) image
    :param brainmask_img: String, valid path to (T1w or T2w) brainmask image
    :param output_crop_img: String, valid path to save cropped image file at
    :return: String, path to crop2full.mat file in same dir as output_crop_img
    """
    # Define filenames
    output_crop_dir = os.path.dirname(output_crop_img)
    crop2full = os.path.join(output_crop_dir, "crop2full.mat")

    # Load averaged image file and SynthSeg-derived brainmask data
    av_img_path = input_avg_img
    mask_img_path = brainmask_img

    av_img = nib.load(av_img_path)
    mask_img = nib.load(mask_img_path)

    av_img_data = av_img.get_fdata()
    mask_img_data = mask_img.get_fdata()

    # Find the first lower axial slices (z-plane) with voxel values of 1 to identify the bottom of the brainmask in axial plane 
    z_dim_size = mask_img_data.shape[2]
    lower_mask_frame = None

    # Iterate over frames starting at bottom of image in z-direction to find lower frame of brainmask
    for z in range(z_dim_size):
        axial_slice = mask_img_data[:, :, z]
        if np.any(axial_slice == 1):
            lower_mask_frame = z
            break

    # Add buffer to lower cutting plane
    voxel_buffer = 10
    lower_axial_crop_plane = lower_mask_frame - voxel_buffer 

    # Crop average image
    print(f'Now cropping average image at z-coordinate plane {lower_axial_crop_plane}')
    cropped_img_data = av_img_data[:, :, lower_axial_crop_plane:]

    cropped_img = nib.Nifti1Image(cropped_img_data, av_img.affine, av_img.header)
    cropped_file = output_crop_img
    nib.save(cropped_img, cropped_file)

    # Create crop2full transformation matrix: create identity matrix and update translation (last column) by the cropping offset in z-direction
    crop2full_matrix=np.eye(4) 
    z_translation = lower_axial_crop_plane * av_img.header.get_zooms()[2]
    crop2full_matrix[2,3] = z_translation

    # Save crop2full.mat as FSL .mat file for use in future stages
    np.savetxt(crop2full, crop2full_matrix, fmt='%.8f')

    return crop2full


def get_and_make_preBIBSnet_work_dirs(j_args):
    """ 
    :param j_args: Dictionary containing all args
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
        LOGGER.debug(f"preBIBSnet_paths: {preBIBSnet_paths}")
        list_files(j_args["common"]["work_dir"])

    return preBIBSnet_paths

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


def register_preBIBSnet_imgs_ACPC(cropped_imgs, output_dir, xfm_non_ACPC_vars,
                                  crop2full, averaged_imgs, j_args):
    """
    :param cropped_imgs: Dictionary mapping ints, (T) 1 or 2, to strings (valid
                         paths to existing image files to resize)
    :param output_dir: String, valid path to a dir to save resized images into
    :param xfm_non_ACPC_vars: Dict TODO Fix this function description
    :param crop2full: String, valid path to existing crop2full.mat file
    :param averaged_imgs: Dictionary mapping ints, (T) 1 or 2, to strings
                          (valid paths to existing image files to resize)
    :param j_args: Dictionary containing all args
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
            j_args, xfm_ACPC_vars, crop2full[t], reg_in_var, t,
            averaged_imgs[f"T{t}w_avg"]
        )

    # T2w-T1w alignment of ACPC-aligned images
    xfm_ACPC_and_reg_imgs = registration_T2w_to_T1w(
        j_args, xfm_ACPC_vars, reg_in_var, acpc=True
    )

    # pdb.set_trace()  # TODO Add "debug" flag?

    return {"vars": xfm_ACPC_vars, "img_paths": xfm_ACPC_and_reg_imgs}


def align_ACPC_1_img(j_args, xfm_ACPC_vars, crop2full, output_var, t,
                     averaged_image):
    """ 
    Functionality copied from the DCAN Infant Pipeline:
    github.com/DCAN-Labs/dcan-infant-pipeline/blob/master/PreFreeSurfer/scripts/ACPCAlignment_with_crop.sh
    :param j_args: Dictionary containing all args
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
    
    run_FSL_sh_script(j_args, "flirt", "-interp", "spline",  
                      "-ref", mni_ref_img_path, "-in", input_img,
                      "-omat", mats["crop2acpc"], # "-out", os.path.join(work_dir, "T{}w_acpc_final.nii.gz".format(t)),
                      "-searchrx", "-45", "45", "-searchry", "-30", "30",
                      "-searchrz", "-30", "30")

    # Invert crop2full to get full2crop
    run_FSL_sh_script(j_args, "convert_xfm", "-inverse", crop2full,
                      "-omat", mats["full2crop"])  # TODO Move this to right after making crop2full to use it in both T?w-only and here

    run_FSL_sh_script(  # Combine ACPC-alignment with robustFOV output
        j_args, "convert_xfm", "-omat", mats["full2acpc"],
        "-concat", mats["crop2acpc"], mats["full2crop"]
    )

    # Transform 12 dof matrix to 6 dof approximation matrix
    run_FSL_sh_script(j_args, "aff2rigid", mats["full2acpc"],
                      mats["acpc2rigidbody"])

    # Apply ACPC alignment to the data
    # Create a resampled image (ACPC aligned) using spline interpolation  # TODO Only run this command in debug mode
    # if j_args["common"]["debug"]:
    run_FSL_sh_script(j_args, "applywarp", "--rel", "--interp=spline",  
                      "-i", averaged_image, "-r", mni_ref_img_path,  # Changed input_img to average_image 2022-06-16
                      "--premat=" + mats["acpc2rigidbody"], "-o", output_img)
    # pdb.set_trace()  # TODO Add "debug" flag?
    return mats

def compute_eta2_between_images(img_path_a, img_path_b, mask=None):
    """
    Compute eta^2-like metric between two images.
    Here we compute Pearson r between nonzero voxels and return r**2.
    Returns float in [0,1]. Higher -> more similar.
    """
    a = nib.load(img_path_a).get_fdata(dtype=np.float64)
    b = nib.load(img_path_b).get_fdata(dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Image shapes differ: {a.shape} vs {b.shape}")

    # Build mask: nonzero in both images (and user-supplied mask)
    mask_data = (np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0))
    if mask is not None:
        mask_img = nib.load(mask).get_fdata(dtype=bool)
        if mask_img.shape == mask_data.shape:
            mask_data &= mask_img.astype(bool)
    vals_a = a[mask_data].ravel()
    vals_b = b[mask_data].ravel()

    if vals_a.size < 10:
        # Too few voxels to get a reliable estimate; fallback to whole-image finite vals
        mask_data = (np.isfinite(a) & np.isfinite(b))
        vals_a = a[mask_data].ravel()
        vals_b = b[mask_data].ravel()

    if vals_a.size == 0:
        return 0.0

    # Remove constant signals
    if np.nanstd(vals_a) == 0 or np.nanstd(vals_b) == 0:
        return 0.0

    r = np.corrcoef(vals_a, vals_b)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(r**2)

def register_preBIBSnet_imgs_non_ACPC(cropped_imgs, output_dir, ref_image, 
                                      ident_mx, resolution, j_args):
    """
    Registers T2w to T1w using non-ACPC method, both with and without
    restricted search space parameters, then chooses the better result
    based on eta-squared (and optionally SSIM). Finally, applies chosen 
    result to make *_to_BIBS and *_crop2BIBS_mat transform matrices.
    :param cropped_imgs: Cropped T1 and T2 image filepaths (dictionary)
    :param output_dir: Output folder filepath
    :param ref_image: MNI template paths (dictionary)
    :param ident_mx: identity matrix .mat filepath
    :param resolution: Output resolution
    :param j_args: Dictionary containing all args
    """
    # Output dirs for free and restricted registrations
    xfm_out_dir_free = os.path.join(output_dir, "xfms_free")
    xfm_out_dir_restrict = os.path.join(output_dir, "xfms_restrict")
    os.makedirs(xfm_out_dir_free, exist_ok=True)
    os.makedirs(xfm_out_dir_restrict, exist_ok=True)

    # Helper lambdas for variable names
    reg_in_var = lambda t: f"reg_input_T{t}w_img"
    out_var = lambda t: f"output_T{t}w_img"

    # Build base variables
    xfm_vars_free = {"out_dir": xfm_out_dir_free,
                    "resolution": resolution,
                    "ident_mx": ident_mx,
                    "ref_img": ref_image}
    xfm_vars_restrict = {"out_dir": xfm_out_dir_restrict,
                         "resolution": resolution,
                         "ident_mx": ident_mx,
                         "ref_img": ref_image}
    
    # Assign inputs
    for t, crop_img_path in cropped_imgs.items():
        img_ext = split_2_exts(crop_img_path)[-1]
        for xfm_vars in (xfm_vars_free, xfm_vars_restrict):
            xfm_vars[reg_in_var(t)] = crop_img_path

            if t == 2:
                # Only create an output for T2 (T1 is not registered/transformed in this function)
                xfm_vars[out_var(t)] = os.path.join(
                    xfm_vars["out_dir"], f"T{t}w_registered_to_T1w{img_ext}"
                )

    # Always set T1w to the input cropped T1 (no separate registered T1 file)
    xfm_vars_free["T1w"] = xfm_vars_free[reg_in_var(1)]
    xfm_vars_restrict["T1w"] = xfm_vars_restrict[reg_in_var(1)]

    # Run free-space registration and cost function set to default (correlation ratio)
    mat_free = os.path.join(xfm_out_dir_free, "cropT2tocropT1.mat")
    run_FSL_sh_script(
        j_args, "flirt",
        "-ref", xfm_vars_free[reg_in_var(1)],
        "-in",  xfm_vars_free[reg_in_var(2)],
        "-omat", mat_free,
        "-out",  xfm_vars_free["output_T2w_img"],
        "-dof", "6"
    )
    xfm_vars_free["T2w"] = xfm_vars_free["output_T2w_img"]
    xfm_vars_free["cropT2tocropT1"] = mat_free

    # Run restricted-space registration
    mat_restrict = os.path.join(xfm_out_dir_restrict, "cropT2tocropT1.mat")
    run_FSL_sh_script(
        j_args, "flirt",
        "-ref", xfm_vars_restrict[reg_in_var(1)],
        "-in",  xfm_vars_restrict[reg_in_var(2)],
        "-omat", mat_restrict,
        "-out",  xfm_vars_restrict["output_T2w_img"],
        "-cost", "mutualinfo",
        "-searchrx", "-15", "15",
        "-searchry", "-15", "15",
        "-searchrz", "-15", "15",
        "-dof", "6"
    )
    xfm_vars_restrict["T2w"] = xfm_vars_restrict["output_T2w_img"]
    xfm_vars_restrict["cropT2tocropT1"] = mat_restrict

    # Calculate ETA2 and SSIM for each workflows
    eta = {
        "free":       calculate_eta(xfm_vars_free), 
        "restricted": calculate_eta(xfm_vars_restrict)
    }
    ssim = {
        "free":       calculate_ssim(xfm_vars_free),
        "restricted": calculate_ssim(xfm_vars_restrict)
    }
    LOGGER.verbose(f"Eta-Squared Values: {eta}")
    LOGGER.verbose(f"SSIM Values: {ssim}")

    # Save results to text file
    try:
        output_path = os.path.join(output_dir, "registration_metrics_xfms.txt")
        with open(output_path, "w") as f:
            f.write("Eta-Squared Values:\n")
            for k, v in eta.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\nSSIM Values:\n")
            for k, v in ssim.items():
                f.write(f"  {k}: {v:.4f}\n")
    except Exception as e:
        LOGGER.warning(f"Could not write registration metrics to file: {e}")

    # Choose best registration
    if eta["free"] > eta["restricted"]:
        chosen = xfm_vars_free
        mode = "free"
        LOGGER.info("Using free-space search T2w-to-T1w registration.")
    else:
        chosen = xfm_vars_restrict
        mode = "restricted"
        LOGGER.info("Using restricted search T2w-to-T1w registration.")

    # Create BIBS outputs and *_crop2BIBS_mat
    registration_outputs = {
        "cropT1tocropT1": ident_mx,
        "cropT2tocropT1": chosen["cropT2tocropT1"],
        "T1w": chosen["T1w"],
        "T2w": chosen["T2w"],
        "chosen_T2_mode": mode,
        "eta_free": eta["free"],
        "eta_restricted": eta["restricted"],
    }

    # Loop through T1, T2 to apply transformations
    for t in (1, 2):
        t_key = f"T{t}w"
        registration_outputs[f"{t_key}_crop2BIBS_mat"] = os.path.join(
            chosen["out_dir"], f"crop_{t_key}_to_BIBS_template.mat"
        )
        registration_outputs[f"{t_key}_to_BIBS"] = os.path.join(
            chosen["out_dir"], f"{t_key}_to_BIBS.nii.gz"
        )

        input_img = chosen[reg_in_var(t)] if t == 1 else registration_outputs["T2w"]
        transform_image_T(t, input_img, chosen, registration_outputs, j_args)

        run_FSL_sh_script(
            j_args, "flirt",
            "-in", input_img,
            "-ref", chosen["ref_img"][t],
            "-applyisoxfm", chosen["resolution"],
            "-init", chosen["ident_mx"],
            "-o", registration_outputs[f"{t_key}_to_BIBS"],
            "-omat", registration_outputs[f"{t_key}_crop2BIBS_mat"]
        )

    return {"vars": chosen, "img_paths": registration_outputs}

def registration_T2w_to_T1w(j_args, xfm_vars, reg_input_var, acpc):
    """
    T2w to T1w registration for use in preBIBSnet
    :param j_args: Dictionary containing all args
    :param xfm_vars: Dictionary containing paths to files used in registration
    :param reg_input_var: String naming the key in xfm_vars mapped to the path
                          to the image to use as an input for registration
    :return: Dictionary mapping "T1w" and "T2w" to their respective newly
             registered image file paths
    """
    # String for naming the key in xfm_vars mapped to the path
    # to the image to use as an input for registration
    inputs_msg = "\n".join(["T{}w: {}".format(t, xfm_vars[reg_input_var.format(t)])
                            for t in only_Ts_needed_for_bibsnet_model(j_args["ID"])])
    LOGGER.verbose("Input images for T1w registration:\n" + inputs_msg)

    # Define paths to registration output matrices and images
    registration_outputs = {"cropT1tocropT1": xfm_vars["ident_mx"],
                            "cropT2tocropT1": os.path.join(xfm_vars["out_dir"],
                                                           "cropT2tocropT1.mat")}
    """
    ACPC Order:
    1. T1w Save cropped and aligned T1w image 
    2. T2w Make T2w-to-T1w matrix
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
            run_FSL_sh_script(j_args, "flirt",
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

    return registration_outputs


def transform_image_T(t, cropped_in_img, xfm_vars, regn_outs, j_args):
    """
    Run FSL command on a cropped input image to apply a .mat file transform 
    :param t: Int, either 1 or 2 (to signify T1w or T2w respectively)
    :param cropped_in_img: String, valid path to cropped T1w or T2w image
    :param xfm_vars: Dict with paths to reference image & identity matrix files
    :param regn_outs: Dict with paths to transformed output images to make
    :param j_args: Dictionary containing all args
    """
    run_FSL_sh_script(  # TODO Should the output image even be created here, or during applywarp?
        j_args, "flirt",
        "-in", cropped_in_img, # xfm_vars[reg_input_var.format(t)] if t == 1 else registration_outputs["T2w"],  # Input: Cropped image
        "-ref", xfm_vars["ref_img"].format(t),
        "-applyisoxfm", xfm_vars["resolution"],
        "-init", xfm_vars["ident_mx"], # registration_outputs["cropT{}tocropT1".format(t)],
        "-o", regn_outs[f"T{t}w_to_BIBS"], # registration_outputs["T{}w".format(t)],  # TODO Should we eventually exclude the (unneeded?) -o flags?
        "-omat", regn_outs[f"T{t}w_crop2BIBS_mat"]
    )

