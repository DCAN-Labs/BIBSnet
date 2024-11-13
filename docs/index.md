# BIBSnet

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7019701.svg)](https://doi.org/10.5281/zenodo.7019701)

We introduce BIBSNet (Baby and Infant Brain Segmentation Neural Network), an open-source, community-driven deep learning model. Provided as a [BIDS App](https://bids-apps.neuroimaging.io/about/) container, BIBSNet leverages data augmentation and a large, manually annotated infant dataset to produce robust and generalizable brain segmentations. The model outputs native-space brain segmentations, brain masks, and sidecar JSON files as [BIDS derivatives](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html).

Latest version: [3.4.2](https://github.com/DCAN-Labs/BIBSnet/releases/tag/3.4.2):    
[BIBSnet Github Repository](https://github.com/DCAN-Labs/BIBSnet)<br>
[BIBSnet dockerhub Repository](https://hub.docker.com/repository/docker/dcanumn/bibsnet/)



## Pipeline Workflow
![BIBSnet - Stages for MRI Processing](BIBSNetWorkflowDiagram.drawio.png)

### Stage 1 - PreBIBSnet 
Prepares the input T1w and/or T2w anatomical image(s) for BIBSnet:
* T1w and T2w images are renamed to fit nnU-Net naming conventions (_0000 and _0001 respectively) and if there are multiple T1w or T2w, they are registered to the first run and averaged
* The neck and shoulders are cropped from the average images using a [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/)-derived brain mask to identify the optimal axial cropping plane
* T2w-to-T1w registration is performed via multiple workflows (either directly or following ACPC-alignment of both T1w and T2w), eta-squared is used to choose the optimal registration method, and the resulting best pair is fed into the next stage for segmentation

### Stage 2 - BIBSnet
Quickly and accurately segments an optimally-aligned T1 and T2 pair with a deep neural network trained via nnU-Net and SynthSeg with 0-8 month old infant MRI brain dataset.

### Stage 3 - PostBIBSnet
Transforms segmentation back to native space for both T1w and T2w, generates brain masks derived from the segmentation, and creates derivative outputs including sidecar jsons. The working directories for pre- through postBIBSNet are removed if user did not specify a working directory.
