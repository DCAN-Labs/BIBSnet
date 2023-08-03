#!/bin/bash -l

#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks 20
#SBATCH --gres=gpu:1
#SBATCH --mem=80gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tikal004@umn.edu
#SBATCH -p v100
#SBATCH -o /home/feczk001/shared/projects/segpipeline_testing/Barry_test/roo_data/logs/%A_bibsnet.out
#SBATCH -e /home/feczk001/shared/projects/segpipeline_testing/Barry_test/roo_data/logs/%A_bibsnet.err
#SBATCH -J cabinet
#SBATCH -A faird

module load singularity

singularity=`which singularity`

# conda activate /home/support/public/pytorch_1.11.0_agate

/usr/bin/singularity run --nv \
-B /home/feczk001/shared/projects/segpipeline_testing/Barry_test/roo_data/input/:/input \
-B /home/feczk001/shared/projects/segpipeline_testing/Barry_test/roo_data/derivatives/:/output \
-B /home/feczk001/shared/projects/segpipeline_testing/Barry_test/roo_data/work/:/work \
/home/faird/shared/code/internal/pipelines/cabinet_container/cabinet_derivative-json.sif \
/input /output participant -jargs /home/cabinet/parameter-file-container.json -start prebibsnet -end postbibsnet -v \
--participant-label M1003 \
-w /work
