#!/bin/bash
set -e

# This script generates a L/R mask for a subject. It first registers a template head (eg "1mo_T1w_acpc_dc_restore.nii.gz") to the subject's head. Next it applies
# the resulting transformation matrix ("antsregAffine.txt") using Nearest Neighbor interpolation to register the template LR mask (eg 
# "1mo_template_LRmask.nii.gz") to the subject head.

# Important notes:
# (1) Templates are age-specific, so make sure to select a template matching the age range of the subject. The 1mo template can be used for neonates.
# (2) If available, it is recommended that one use T2w subject/template anatomicals for subjects 0-21 months of age and T1w subject/template anatomicals for 
# subjects over 21 months old. The contrast in infant T2w scans makes the pial surface more visible and increases the quality of registration.

SubjectHead=$1;shift
TemplateHead=$1;shift
TemplateMask=$1;shift

module load ants
WD="./wd"
mkdir "$WD"

# Register the template head to the subject head
ANTS 3 -m CC["$SubjectHead","$TemplateHead",1,5] -t SyN[0.25] -r Gauss[3,0] -o "$WD"/antsreg -i 60x50x20 --use-Histogram-Matching  --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000

# Apply resulting transformation to template L/R mask to generate subject L/R mask
antsApplyTransforms -d 3 \
        --output LRmask.nii.gz \
        --reference-image "$SubjectHead" \
        --transform "$WD"/antsregWarp.nii.gz "$WD"/antsregAffine.txt \
        --input "$TemplateMask" \
	--interpolation NearestNeighbor

#delete wd
rm -r "$WD"
