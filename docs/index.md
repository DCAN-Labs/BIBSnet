# BIBSnet

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7019701.svg)](https://doi.org/10.5281/zenodo.7019701)

[BIBSnet Github Repository](https://github.com/DCAN-Labs/BIBSnet) \
[BIBSnet dockerhub Repository](https://hub.docker.com/repository/docker/dcanumn/bibsnet/)

This [BIDS App](https://bids-apps.neuroimaging.io/about/) provides the utility of creating a [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) anatomical MRI segmentation and mask with a infant brain trained model for the purposes of circumventing JLF within [Nibabies](https://nibabies.readthedocs.io/en/latest/index.html). 

Please post questions or issues to our [Github](https://github.com/DCAN-Labs/BIBSnet)! 

## Stages
![BIBSnet - Stages for MRI Processing](BIBSNetWorkflowDiagram.drawio.png)

### Stage 1 - PreBIBSnet 
Prepares the input T1w and/or T2w anatomical image(s) for BIBSnet segmentation generation with the following steps:

1. Renames T1 and T2 to fit nnU-Net naming conventions (0000 and 0001 for T1 and T2, respectively)
2. Creates average T1w/T2w images: images are registered to the first run and averaged
3. Crops the neck and shoulders using a SynthStrip-derived brain mask to identify the optimal axial cropping plane
4. Performs T2-to-T1 registration either directly or following ACPC-alignment of both T1w and T2w. The optimal registration method is chosen based on based on eta-squared calculation

### Stage 2 - BIBSnet
Quickly and accurately segments an optimally-aligned T1 and T2 pair with a deep neural network trained via nnU-Net and SynthSeg with 0-8 month old infant MRI brain dataset.

### Stage 3 - PostBIBSnet
Transforms segmentation back to native space for both T1w and T2w, generates brain masks derived from the segmentation, and creates derivative outputs including sidecar jsons. The working directories for pre- through postBIBSNet are removed if user did not specify a working directory.
