## 1. PreBIBSnet

Prepares the anatomical BIDS images for BIBSnet segmentation generation.

1. Averages multiple T1/T2 weighted images into one T1/T2 weighted image (all images are registered to the first run)
2. Renames T1 and T2 to fit nnU-Net naming (0000 and 0001 for T1 and T2, respectively)
3. Crops the neck and shoulders out of the images
4. Registers the T2 to the T1 either via traditional transform or ACPC-alignment. Optimal method gets chosen via eta-squared calculation

<br />

## 2. BIBSnet

Quickly and accurately segments an optimally-aligned T1 and T2 pair with a deep neural network trained via nnU-Net and SynthSeg with a large 0 to 8 month old infant MRI brain dataset.

### BIBSnet Segmentation Models

`data/models.csv` lists all available BIBSnet models to run. Below are the default BIBSnet models, all trained on manually-segmented 0- to 8-month-old BCP subjects' segmentations. 

| Model | Description |
|:-:|:--|
| 514 | Default T1w-only model |
| 515 | Default T2w-only model |
| 526 | Default T1w and T2w model |
| 552 | Old T1w and T2w model |

<br />

**NOTE:** For running the T1w- and T2w-only models, you need to remove the unused anatomical image. For example, if you're running the T1w-only model, remove the T2w images from the subject folder and vice versa. 

## 3. PostBIBSnet

1. Generates a left-right mask and dilates it
2. Uses dilated left-right mask to correct segmentation chirality
3. Registers the segmentation back into native T1 space using transform produced via optimal registration in preBIBSnet
4. Generates a mask of the segmentation from the native T1 space segmentation
5. Renames the native T1 space segmentation and mask to BIDS standard naming conventions to feed into Nibabies
6. Creates a "bibsnet" derivatives directory for input into Nibabies, containing the final segmentation and mask along with the `dataset_description.file`
7. Removes prebibsnet through postbibsnet working directories if the user does not specify a working directory.

<br />
