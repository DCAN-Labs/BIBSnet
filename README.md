# CABINET

##Stages
1. PreBIBSnet
2. BIBSnet
3. PostBIBSnet
4. Nibabies
5. ABCD-XCP

## 1. PreBIBSnet

Crops the neck and shoulders out of infant images and resizes them.

## 2. BIBSnet

Produces a segmentation created by nnU-Net from a model trained on 0-8 month old infants.

## 3. PostBIBSnet

Checks the chirality of the segmentation and flips the regions if needed.

## 4. Nibabies

Infant fmriprep image processing

## 5. ABCD-XCP

DCANBOLDProc and Executive Summary
