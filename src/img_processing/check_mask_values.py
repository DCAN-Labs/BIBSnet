"""
Check that fixed chirality masks don't contain label values of 4 or 5.

Usage:
  check_mask_values <dilated_LRmask>
  check_mask_values -h | --help

Options:
  -h --help     Show this screen.
"""

import nibabel as nib
import numpy as np
from docopt import docopt
import pandas as pd


def check_mask_values(dilated_LRmask):
    mask_img = nib.load(dilated_LRmask)
    mask_data = mask_img.get_fdata()

    #reshape numpy array
    mask_data_2D = mask_data.reshape((182, 39676), order='C')
    mask_data_1D = mask_data_2D.reshape((7221032), order='C')

    val_0 = np.where(mask_data_1D == 0.0)
    num_nonzeros = mask_data_1D.shape[0] - len(val_0[0])

    val_4 = np.where(mask_data_1D == 4.0)
    val_5 = np.where(mask_data_1D == 5.0)
    val_6 = np.where(mask_data_1D == 6.0)
    total = len(val_4[0]) + len(val_5[0]) + len(val_6[0])

    percentage_mislabeled = total/num_nonzeros
    percentage_mislabeled = round(percentage_mislabeled, 5)
    print("{} out of {} voxels ({}%) have a value of 4, 5, or 6".format(total, num_nonzeros, percentage_mislabeled))


if __name__ == '__main__':
    args = docopt(__doc__)
    check_mask_values(args['<dilated_LRmask>'])