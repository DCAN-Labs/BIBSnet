import nibabel as nib

from img_processing.chirality_constants import LEFT, RIGHT, BILATERAL
from util.look_up_tables import get_id_to_region_mapping

# TODO Un-hardcode, create main function, etc.
nifti_input_file_path = '../../temp/aseg_acpc_final.nii.gz'
nifti_output_file_path = '../../data/chirality_masks/1_month.nii.gz'

img = nib.load(nifti_input_file_path)
data = img.get_data()
data_shape = img.header.get_data_shape()
width = data_shape[0]
height = data_shape[1]
depth = data_shape[2]
chirality_mask = data.copy()
segment_lookup_table = '../../data/look_up_tables/FreeSurferColorLUT.txt'
free_surfer_label_to_region = get_id_to_region_mapping(segment_lookup_table)
for i in range(width):
    for j in range(height):
        for k in range(depth):
            region_id = data[i][j][k]
            if region_id == 0:
                continue
            region_name = free_surfer_label_to_region[region_id]
            if region_name.startswith('Left-'):
                chirality_mask[i][j][k] = LEFT
            elif region_name.startswith('Right-'):
                chirality_mask[i][j][k] = RIGHT
            else:
                chirality_mask[i][j][k] = BILATERAL
mask_img = nib.Nifti1Image(chirality_mask, img.affine, img.header)
nib.save(mask_img, nifti_output_file_path)
