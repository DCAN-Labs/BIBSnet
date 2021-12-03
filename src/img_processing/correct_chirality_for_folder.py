# Author: Paul Reiners

import os
import sys

from img_processing.correct_chirality import correct_chirality


def correct_chirality_for_folder(input_folder, lut_file, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".nii.gz"):
            file_path = os.path.join(input_folder, filename)
            print(file_path)
            correct_chirality(file_path, lut_file, os.path.join(output_folder, filename))
        else:
            continue


if __name__ == '__main__':
    correct_chirality_for_folder(sys.argv[1], sys.argv[2], sys.argv[3])
