## 1. PreBIBSnet

Prepares the anatomical BIDS images for BIBSnet segmentation generation.

1. Creates average T1w/T2w images: images are registered to the first run and averaged
2. Renames T1 and T2 to fit nnU-Net naming conventions (0000 and 0001 for T1 and T2, respectively)
3. Crops the neck and shoulders out of the images using SynthStrip to identify optimal axial cropping plane
4. Registers the T2 to the T1 either directly or following ACPC-alignment of both T1w and T2w. The optimal method is chosen based on whichever results in the best T2-to-T1 registration asw via eta-squared calculation

<br />

## 2. BIBSnet

Quickly and accurately segments an optimally-aligned T1 and T2 pair with a deep neural network trained via nnU-Net and SynthSeg with 0-8 month old infant MRI brain dataset.

<br />

**NOTE:** For running the T1w- and T2w-only models, you need to remove the unused anatomical image. For example, if you're running the T1w-only model, remove the T2w images from the subject folder and vice versa. 

## 3. PostBIBSnet

Generates T1w/T2w-derived brainmask(s) from segmentation(s), registers the segmentation and brainmask to native space, creates derivative outputs, and removes prebibsnet through postbibsnet working directories if the user does not specify a working directory.

<br />
