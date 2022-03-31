#!/bin/sh

# Only run if the user provided exactly 1 input argument: the parameter file path
if [ "$#" -eq 1 ]; then 

    # Get the code directory where test_run.sh lives, assumed to be the same CABINET directory where run.py lives
    this_dir=$(dirname ${0})

    # Get parameter file as input variable from user
    param_file=${1}

    # Activate cabinet environment
    source /home/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
    conda activate cabinet
    echo Activated cabinet conda environment

    # Run preBIBSnet
    python3 /home/faird/shared/code/internal/pipelines/CABINET/run.py ${param_file} -start preBIBSnet -end preBIBSnet 

    # Deactivate cabinet environment
    conda deactivate
    echo Deactivated cabinet conda environment

    # Set paths to run BIBSnet
    path_BIBSnet="/home/faird/shared/code/internal/pipelines/bibsnet/"
    # export PATH="${PATH}:/home/faird/shared/code/internal/pipelines/bibsnet/"
    export nnUNet_raw_data_base="/home/feczk001/shared/data/nnUNet/nnUNet_raw_data_base/"
    export nnUNet_preprocessed="/home/feczk001/shared/data/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed"
    export RESULTS_FOLDER="/home/feczk001/shared/data/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models"

    # Activate PyTorch/nnU-Net environment to run BIBSnet
    module load gcc cuda/11.2
    source /panfs/roc/msisoft/anaconda/anaconda3-2018.12/etc/profile.d/conda.sh
    conda activate /home/support/public/torch_cudnn8.2
    echo Activated BIBSnet PyTorch conda environment

    # Get BIBSnet parameters from parameter file

    # Run BIBSnet
    python3 -c "import os; import sys; import json; import subprocess; sys.path.append('${path_BIBSnet}'); from BIBSnet.run import run_nnUNet_predict; param_file=open('${param_file}'); params=json.load(param_file); param_file.close(); print(params); sub = 'sub-' + str(params['common']['participant_label']); sub_ses = [sub, 'ses-' + str(params['common']['session'])] if params['common']['session'] else [sub]; print('sub_ses is {}'.format(sub_ses)); print('BIBSnet out_dir is {}'.format(params['optional_out_dirs']['derivatives'])); dir_BIBS = os.path.join(params['optional_out_dirs']['derivatives'], 'BIBSnet', *sub_ses, '{}put'); print('dir_BIBS is {}'.format(dir_BIBS)); run_nnUNet_predict({'model': params['BIBSnet']['model'], 'nnUNet': params['BIBSnet']['nnUNet_predict_path'], 'input': dir_BIBS.format('in'), 'output': dir_BIBS.format('out'), 'task': str(params['BIBSnet']['task'])})"

    # Deactivate PyTorch/nnU-Net environment 
    conda deactivate
    echo Deactivated BIBSnet PyTorch conda environment

    # Activate cabinet environment 
    source /home/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
    conda activate cabinet
    echo Activated cabinet conda environment

    # Run postBIBSnet
    python3 /home/faird/shared/code/internal/pipelines/CABINET/run.py ${param_file} -start postBIBSnet -end postBIBSnet

else 

    echo "Usage: test_run.sh \$param_file_path"

fi
