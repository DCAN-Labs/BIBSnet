import os
import shutil

from src.logger import LOGGER

from src.utilities import (
    apply_final_prebibsnet_xfms, 
    create_anatomical_averages, crop_image,
    get_and_make_preBIBSnet_work_dirs,
    get_preBIBS_final_img_fpath_T, get_subj_ID_and_session,
    only_Ts_needed_for_bibsnet_model, register_preBIBSnet_imgs_ACPC, 
    register_preBIBSnet_imgs_non_ACPC, run_FSL_sh_script
) 


SCRIPT_DIR = os.path.dirname(os.path.dirname(__file__))


def run_preBIBSnet(j_args):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :return: j_args, but with preBIBSnet working directory names added
    """
    completion_msg = "The anatomical images have been {} for use in BIBSnet"
    preBIBSnet_paths = get_and_make_preBIBSnet_work_dirs(j_args)
    sub_ses = get_subj_ID_and_session(j_args)

    # If there are multiple T1ws/T2ws, then average them
    create_anatomical_averages(preBIBSnet_paths["avg"], LOGGER)  # TODO make averaging optional with later BIBSnet model?

    # Crop T1w and T2w images
    cropped = dict()
    crop2full = dict()
    for t in only_Ts_needed_for_bibsnet_model(j_args["ID"]):
        cropped[t] = preBIBSnet_paths[f"crop_T{t}w"]
        crop2full[t] = crop_image(preBIBSnet_paths["avg"][f"T{t}w_avg"],
                                  cropped[t], j_args, LOGGER)
    LOGGER.info(completion_msg.format("cropped"))

    # Resize T1w and T2w images if running a BIBSnet model using T1w and T2w
    # TODO Make ref_img an input parameter if someone wants a different reference image?
    # TODO Pipeline should verify that reference_img files exist before running
    reference_img = os.path.join(SCRIPT_DIR, "data", "MNI_templates",
                                 "INFANT_MNI_T{}_1mm.nii.gz") 
    id_mx = os.path.join(SCRIPT_DIR, "data", "identity_matrix.mat")
    # TODO Resolution is hardcoded; infer it or get it from the command-line
    resolution = "1"  
    if j_args["ID"]["has_T1w"] and j_args["ID"]["has_T2w"]:
        msg_xfm = "Arguments for {}ACPC image transformation:\n{}"

        # Non-ACPC
        regn_non_ACPC = register_preBIBSnet_imgs_non_ACPC(
            cropped, preBIBSnet_paths["resized"], reference_img, 
            id_mx, resolution, j_args
        )
        if j_args["common"]["verbose"]:
            LOGGER.info(msg_xfm.format("non-", regn_non_ACPC["vars"]))

        # ACPC
        regn_ACPC = register_preBIBSnet_imgs_ACPC(
            cropped, preBIBSnet_paths["resized"], regn_non_ACPC["vars"],
            crop2full, preBIBSnet_paths["avg"], j_args
        )
        if j_args["common"]["verbose"]:
            LOGGER.info(msg_xfm.format("", regn_ACPC["vars"]))

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
            "-init", id_mx, # TODO Should this be a matrix that does a transformation?
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

        # - Concatenate crop .mat to out_mat (in that order) and apply the
        #   concatenated .mat to the averaged image as the output
        # - Treat that concatenated output .mat as the output to pass
        #   along to postBIBSnet, and the image output to BIBSnet
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

    # TODO Copy this whole block to postBIBSnet, so it copies everything it needs first
    # Copy preBIBSnet outputs into BIBSnet input dir
    for t in only_Ts_needed_for_bibsnet_model(j_args["ID"]): 
        # Copy image files
        out_nii_fpath = j_args["optimal_resized"][f"T{t}w"]
        os.makedirs(os.path.dirname(out_nii_fpath), exist_ok=True)
        if j_args["common"]["overwrite"]:  # TODO Should --overwrite delete old image file(s)?
            os.remove(out_nii_fpath)
        if not os.path.exists(out_nii_fpath): 
            shutil.copy2(transformed_images[f"T{t}w"], out_nii_fpath)

        # Copy .mat into postbibsnet dir with the same name regardless of which
        # is chosen, so postBIBSnet can use the correct/chosen .mat file
        concat_mat = transformed_images[f"T{t}w_crop2BIBS_mat"]
        out_mat_fpath = os.path.join(  # TODO Pass this in (or out) from the beginning so we don't have to build the path twice (once here and once in postBIBSnet)
            j_args["optional_out_dirs"]["postbibsnet"],
            *sub_ses, "preBIBSnet_" + os.path.basename(concat_mat)
        )
        if not os.path.exists(out_mat_fpath):
            shutil.copy2(concat_mat, out_mat_fpath)
            if j_args["common"]["verbose"]:
                LOGGER.info(f"Copying {concat_mat} to {out_mat_fpath}")
    LOGGER.info("PreBIBSnet has completed")
    return j_args

