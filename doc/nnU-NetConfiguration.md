# nnU-Net configuration

## Environment variables

nnU-Net needs to know where you intend to save raw data, preprocessed data and trained models. For
it to know this, you need to set three environment variables:

* nnUNet_raw_data_base: this folder is for training data.  To my knowledge, it is not used at all during inference.
* nnUNet_preprocessed: this folder is written to by nnU-Net.  I imagine it might be used for both training and inference.
* RESULTS_FOLDER: this folder is where models are written to by nnU-Net in the case of training, and where models are loaded from in the case of inference.

Here is an example:

    export nnUNet_raw_data_base="/home/feczk001/shared/data/nnUNet/nnUNet_raw_data_base/"
    export nnUNet_preprocessed="/home/feczk001/shared/data/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed"
    export RESULTS_FOLDER="/home/feczk001/shared/data/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models"

During development of CABINET, I recommend setting all three of these environment variables to existing folders, **even if you think a particular variable might not be needed by inference**.
Once the nnU-Net part of CABINET is working without error, then you can try removing variables that you do not think are necessary.

For more details, see [this section]()