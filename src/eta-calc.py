#!/usr/bin/env python3
import nibabel as nib
import numpy as np
import argparse
import sys

def reshape_volume_to_array(array_img):
    """Return a flattened float64 array from a NIfTI image."""
    image_data = array_img.get_fdata(dtype=np.float64)
    return image_data.ravel()

def sum_of_2_sums_of_squares_of(np_vector1, np_vector2, a_mean):
    """
    a_mean can be a scalar (grand mean) or a vector (per-voxel mean).
    Returns scalar sum of squared deviations for both vectors.
    """
    total = 0.0
    for each_vec in (np_vector1, np_vector2):
        total += np.sum((each_vec - a_mean) ** 2)
    return float(total)

def calculate_eta(img_paths):
    """
    img_paths: dict with keys "T1w" and "T2w" -> paths to images.
    Returns 1 - SS_within / SS_total for two images.
    """
    v = {}
    for t in (1, 2):
        anat = f"T{t}w"
        v[anat] = reshape_volume_to_array(nib.load(img_paths[anat]))

    if v["T1w"].shape != v["T2w"].shape:
        raise ValueError(f"Vector lengths differ: {v['T1w'].shape} vs {v['T2w'].shape}")

    m_grand = (np.mean(v["T1w"]) + np.mean(v["T2w"])) / 2.0
    m_within = (v["T1w"] + v["T2w"]) / 2.0

    sswithin = sum_of_2_sums_of_squares_of(v["T1w"], v["T2w"], m_within)
    sstot   = sum_of_2_sums_of_squares_of(v["T1w"], v["T2w"], m_grand)

    if sstot == 0:
        return 0.0
    return 1.0 - (sswithin / sstot)

def compute_eta2_between_images(img_path_a, img_path_b, mask=None):
    """
    Pearson r^2 between the two images over nonzero, finite voxels (and optional mask).
    """
    a = nib.load(img_path_a).get_fdata(dtype=np.float64)
    b = nib.load(img_path_b).get_fdata(dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Image shapes differ: {a.shape} vs {b.shape}")

    mask_data = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    if mask is not None:
        mask_img = nib.load(mask).get_fdata().astype(bool)
        if mask_img.shape == mask_data.shape:
            mask_data &= mask_img

    vals_a = a[mask_data].ravel()
    vals_b = b[mask_data].ravel()

    if vals_a.size < 10:
        mask_data = np.isfinite(a) & np.isfinite(b)
        vals_a = a[mask_data].ravel()
        vals_b = b[mask_data].ravel()

    if vals_a.size == 0:
        return 0.0

    if np.nanstd(vals_a) == 0 or np.nanstd(vals_b) == 0:
        return 0.0

    r = np.corrcoef(vals_a, vals_b)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(r ** 2)

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

def main():
    parser = argparse.ArgumentParser(description="Compare two NIfTI images using eta metrics.")
    parser.add_argument("img1", help="Path to first NIfTI image")
    parser.add_argument("img2", help="Path to second NIfTI image")
    parser.add_argument("--mask", help="Optional mask image path", default=None)
    args = parser.parse_args()

    try:
        eta_method1 = calculate_eta({"T1w": args.img1, "T2w": args.img2})
        eta_method2 = compute_eta2_between_images(args.img1, args.img2, mask=args.mask)
        ssim = calculate_ssim({"T1w": args.img1, "T2w": args.img2})
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Eta (method 1): {eta_method1:.6f}")
    print(f"Eta (method 2, r^2): {eta_method2:.6f}")
    print(f"SSIM: {ssim:.6f}")

if __name__ == "__main__":
    main()
