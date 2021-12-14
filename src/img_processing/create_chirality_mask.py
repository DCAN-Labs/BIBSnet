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
import shutil
import numpy as np

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

    fix_overlap_values(nifti_output_file_path)


def fill_in_holes(nifti_output_file_path):
    os.system('module load fsl')

    #create working directory to store intermediate outputs that can be deleted after
    if not os.path.exists('wd'):
        os.mkdir('wd')

    # separate mask into L (1), R (2), and middle (3) files.

    anatfile = nifti_output_file_path
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-thr 1 -uthr 1',
                           out_file='wd/Lmask.nii.gz')
    maths.run()

    maths = fsl.ImageMaths(in_file=anatfile, op_string='-thr 2 -uthr 2',
                           out_file='wd/Rmask.nii.gz')
    maths.run()

    maths.run()
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-thr 3 -uthr 3',
                           out_file='wd/Mmask.nii.gz')
    maths.run()

    # dilate, fill, and erode each mask in order to get rid of holes (also binarize L and M images in order to perform binary operations)
    anatfile = 'wd/Lmask.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-dilM -dilM -dilM -fillh -ero -ero -ero',
                           out_file='wd/L_mask_holes_filled.nii.gz')
    maths.run()

    anatfile = 'wd/Rmask.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-bin -dilM -dilM -dilM -fillh -ero -ero -ero',
                           out_file='wd/R_mask_holes_filled.nii.gz')
    maths.run()

    anatfile = 'wd/Mmask.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-bin -dilM -dilM -dilM -fillh -ero -ero -ero',
                           out_file='wd/M_mask_holes_filled.nii.gz')
    maths.run()

    # Reassign values of 2 and 3 to R and middle masks
    anatfile = 'wd/R_mask_holes_filled.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-mul 2',
                           out_file='wd/R_mask_holes_filled_label2.nii.gz')
    maths.run()

    anatfile = 'wd/M_mask_holes_filled.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-mul 3',
                           out_file='wd/M_mask_holes_filled_label3.nii.gz')
    maths.run()

    # recombine new L and R mask files
    anatfile_left = 'wd/L_mask_holes_filled.nii.gz'
    anatfile_right = 'wd/R_mask_holes_filled_label2.nii.gz'
    anatfile_mid = 'wd/M_mask_holes_filled_label3.nii.gz'
    maths = fsl.ImageMaths(in_file=anatfile_left, op_string='-add {}'.format(anatfile_right),
                           out_file='wd/recombined_mask_LR.nii.gz')
    maths.run()

    maths = fsl.ImageMaths(in_file=anatfile_mid, op_string='-add wd/recombined_mask_LR.nii.gz',
                           out_file='filled_mask.nii.gz')
    maths.run()
    #os.replace('filled_mask.nii.gz', nifti_output_file_path)

    shutil.rmtree('wd')

def fix_overlap_values(nifti_output_file_path):
    # load original and filled LR mask data
    orig_LRmask_img = nib.load(nifti_output_file_path)
    orig_LRmask_data = orig_LRmask_img.get_fdata()

    fill_LRmask_img = nib.load('filled_mask.nii.gz')
    fill_LRmask_data = fill_LRmask_img.get_fdata()

    # Flatten numpy arrays
    orig_LRmask_data_2D = orig_LRmask_data.reshape((182, 39676), order='C')
    orig_LRmask_data_1D = orig_LRmask_data_2D.reshape((7221032), order='C')

    fill_LRmask_data_2D = fill_LRmask_data.reshape((182, 39676), order='C')
    fill_LRmask_data_1D = fill_LRmask_data_2D.reshape((7221032), order='C')

    #grab index values of voxels with a value of 4, 5, or 6 in filled L/R mask
    val_0 = np.where(fill_LRmask_data_1D == 0.0)
    num_nonzeros = fill_LRmask_data_1D.shape[0] - len(val_0[0])

    val_4 = np.where(fill_LRmask_data_1D == 4.0)
    val_5 = np.where(fill_LRmask_data_1D == 5.0)
    val_6 = np.where(fill_LRmask_data_1D == 6.0)
    total = len(val_4[0]) + len(val_5[0]) + len(val_6[0])

    percentage_mislabeled = total / num_nonzeros
    percentage_mislabeled = round(percentage_mislabeled, 5)
    print("{} out of {} voxels ({}%) have a value of 4, 5, or 6".format(total, num_nonzeros, percentage_mislabeled))
    print("Replacing voxel values of 4, 5, or 6 with equivalent voxel label alues from initial L/R mask created before filling holes...")

    #Replace overlapping label values with corresponding label values from initial mask
    index_vals = np.concatenate([val_4, val_5, val_6], axis=1)
    for i in index_vals[:]:
        fill_LRmask_data_1D[i] = orig_LRmask_data_1D[i]

    # reshape numpy array
    fill_LRmask_data_2D = fill_LRmask_data_1D.reshape((182, 39676), order='C')
    fill_LRmask_data_3D = fill_LRmask_data_2D.reshape((182, 218, 182), order='C')

    #save new numpy array as image
    empty_header = nib.Nifti1Header()
    out_img = nib.Nifti1Image(fill_LRmask_data_3D, orig_LRmask_img.affine, empty_header)
    nib.save(out_img, nifti_output_file_path)

    os.remove('filled_mask.nii.gz')

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
