#!/bin/bash -l

#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks 20
#SBATCH --gres=gpu:1
#SBATCH --mem=80gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tikal004@umn.edu
#SBATCH -p v100
#SBATCH -o /home/feczk001/shared/projects/segpipeline_testing/Barry_test/cbaw-test/logs/%A_cbaw.out
#SBATCH -e /home/feczk001/shared/projects/segpipeline_testing/Barry_test/cbaw-test/logs/%A_cbaw.err
#SBATCH -J cabinet
#SBATCH -A feczk001

module load singularity
module load python

singularity=`which singularity`

# conda activate /home/support/public/pytorch_1.11.0_agate

./run.py parameter-jsons/bibsnet-nibabies-xcpd.json
