#!/bin/bash
set -e

SubjectHead=$1;shift
TemplateHead=$1;shift
TemplateMask=$1;shift

# this script will register a LR mask from a "template" brain (T1 OR T2) to a "target" brain T1/T2 from a different case

#module load ANTS
WD="./wd"
mkdir "$WD"

# Register the template head to the subject head
ANTS 3 -m CC["$SubjectHead","$TemplateHead",1,5] -t SyN[0.25] -r Gauss[3,0] -o "$WD"/antsreg -i 60x50x20 --use-Histogram-Matching  --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000

# apply transformation to template mask
antsApplyTransforms -d 3 \
        --output LRmask.nii.gz \
        --reference-image "$SubjectHead" \
        --transform "$WD"/antsregWarp.nii.gz "$WD"/antsregAffine.txt \
        --input "$TemplateMask" \
	--interpolation NearestNeighbor

#delete wd
rm -r "$WD"
