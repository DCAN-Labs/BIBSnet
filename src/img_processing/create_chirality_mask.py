"""
Create chirality mask.

Usage:
  create_chirality_mask <nifti_input_file_path> <segment_lookup_table> <nifti_output_file_path>
  create_chirality_mask -h | --help

Options:
  -h --help     Show this screen.
"""
from docopt import docopt
import nibabel as nib

from img_processing.chirality_constants import LEFT, RIGHT, BILATERAL
from util.look_up_tables import get_id_to_region_mapping


def correct_chirality_mask(nifti_input_file_path, segment_lookup_table, nifti_output_file_path):
    img = nib.load(nifti_input_file_path)
    data = img.get_data()
    data_shape = img.header.get_data_shape()
    width = data_shape[0]
    height = data_shape[1]
    depth = data_shape[2]
    chirality_mask = data.copy()
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


if __name__ == '__main__':
    args = docopt(__doc__)
    correct_chirality_mask(
        args['<nifti_input_file_path>'], args['<segment_lookup_table>'], args['<nifti_output_file_path>'])
