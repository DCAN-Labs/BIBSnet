"""
Left-right registration correction.

Usage:
  left_right_registration_correction <subject_head> <template_head> <template_mask> <nifti_input_file_path> <nifti_output_file_path> <segment_lookup_table>
  left_right_registration_correction -h | --help

Options:
  -h --help     Show this screen.
"""

import subprocess
from docopt import docopt

from img_processing.correct_chirality import correct_chirality

if __name__ == '__main__':
    args = docopt(__doc__)
    # 1. LR_mask_registration.sh
    subject_head = args['<subject_head>']
    template_head = args['<template_head>']
    template_mask = args['<template_mask>']
    nifti_input_file_path = args['<nifti_input_file_path>']
    segment_lookup_table = args['<segment_lookup_table>']
    command = '../../bin/LR_mask_registration.sh {} {} {}'.format(subject_head, template_head, template_mask)
    nifti_output_file_path = args['<nifti_output_file_path>']
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    return_code = process.returncode
    print(return_code)
    if return_code == 0:
        # 2. correct_chirality.py
        correct_chirality(nifti_input_file_path, segment_lookup_table, 'LRmask.nii.gz', nifti_output_file_path)
    else:
        print('Error occurred during call to LR_mask_registration.')
