"""
Create chirality mask.

Usage:
  create_chirality_mask <nifti_input_file_path> <segment_lookup_table> <nifti_output_file_path>
  create_chirality_mask -h | --help

Options:
  -h --help     Show this screen.
"""
import os

from docopt import docopt
import nibabel as nib

from nipype.interfaces import fsl


UNKNOWN = 0
LEFT = 1
RIGHT = 2
BILATERAL = 3


def get_id_to_region_mapping(mapping_file_name, separator=None):
    file = open(mapping_file_name, 'r')
    lines = file.readlines()

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


def correct_chirality_mask(nifti_input_file_path, segment_lookup_table, nifti_output_file_path):
    create_initial_mask(nifti_input_file_path, nifti_output_file_path, segment_lookup_table)

    fill_in_holes(nifti_output_file_path)


def fill_in_holes(nifti_output_file_path):
    os.system('module load fsl')
    # separate mask into L and R files
    anatfile = nifti_output_file_path
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-thr 1 -uthr 1 -bin',
                           out_file='Lmask.nii.gz')
    maths.run()
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-thr 2 -uthr 2 -bin',
                           out_file='Rmask.nii.gz')
    maths.run()
    # dilate, fill, and erode each mask in order to get rid of holes
    anatfile = 'Lmask.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-dilM -dilM -dilM -dilM -fillh -ero -ero -ero -ero',
                           out_file='L_mask_holes_filled.nii.gz')
    maths.run()
    anatfile = 'Rmask.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-dilM -dilM -dilM -dilM -fillh -ero -ero -ero -ero',
                           out_file='R_mask_holes_filled.nii.gz')
    maths.run()
    # recombine new L and R mask files
    anatfile_left = 'L_mask_holes_filled.nii.gz'
    anatfile_right = 'R_mask_holes_filled.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile_left, op_string='-add {}'.format(anatfile_right),
                           out_file='recombined_mask.nii.gz')
    maths.run()


def create_initial_mask(nifti_input_file_path, nifti_output_file_path, segment_lookup_table):
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
