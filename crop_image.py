import os
import nibabel as nib
import numpy as np

wd='/your/path'
os.chdir(wd)

# Load averaged image file and SynthSeg-derived brainmask data
av_img_path = 'average_image.nii.gz'
mask_img_path = 'average_image_MASK.nii.gz'

av_img = nib.load(av_img_path)
mask_img = nib.load(mask_img_path)

av_img_data = av_img.get_fdata()
mask_img_data = mask_img.get_fdata()

# Find the first (1) lower and (2) upper axial slices (z-plane) with voxel values of 1 to identify the bottom and top of the brainmask in axial plane 
z_dim_size = mask_img_data.shape[2]
lower_mask_frame = None
upper_mask_frame = None

# Note that for standardly oriented neuroimaging data, the bottom of the image in the axial plane
# corresponds to the lowest value in the z-plane plane of the image data array

# Iterate over frames starting at bottom of image in z-direction to find lower frame of brainmask
for z in range(z_dim_size):
    axial_slice = mask_img_data[:, :, z]
    if np.any(axial_slice == 1):
        lower_mask_frame = z
        break

# Iterate over frames starting at top of image in z-direction to find upper frame of brainmask
for z in range(z_dim_size - 1, -1, -1): 
    axial_slice = mask_img_data[:, :, z]
    if np.any(axial_slice == 1):
        upper_mask_frame = z
        break

# Add buffer to lower cutting plane
voxel_buffer = 10
lower_axial_crop_plane = lower_mask_frame - voxel_buffer #lowers the lower axial cutting plane to prevent possible overcropping
upper_axial_crop_plane = upper_mask_frame + lower_mask_frame #offset crop from lower_axial_crop_plane

print(f'Now cropping average image at z-coordinate planes {lower_axial_crop_plane} and {upper_axial_crop_plane}')

cropped_img_data = av_img_data[:, :, lower_axial_crop_plane:]
cropped_img_data = cropped_img_data[:, :, :upper_axial_crop_plane+1]

cropped_img = nib.Nifti1Image(cropped_img_data, av_img.affine, av_img.header)
cropped_file = 'average_image_cropped.nii.gz'
nib.save(cropped_img, cropped_file)
