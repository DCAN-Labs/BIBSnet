#!/bin/bash
set -e

Reference=$1;;shift
TargetNifti=$1;;shift
ReferenceMask=$1;;shift

# this script will register a LR mask from a "template" brain (T1 OR T2) to a "target" brain T1/T2 from a different case

# functions for registering the target brain to the template
${ANTSPATH}${ANTSPATH:+/}ANTS 3 -m CC["$TargetNifti","$Reference",1,5] -t SyN[0.25] -r Gauss[3,0] -o "$WD"/antsreg -i 60x50x20 --use-Histogram-Matching  --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000

# apply transformation to template mask
${ANTSPATH}${ANTSPATH:+/}antsApplyTransforms -d 3 \
        --output "$WD"/LRmask.nii.gz \
        --reference-image "$TargetNifti" \
        --transform "$WD"/antsregWarp.nii.gz "$WD"/antsregAffine.txt \
        --input "$ReferenceMask"